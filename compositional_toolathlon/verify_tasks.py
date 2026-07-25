from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from .config import PACKAGE_DIR, load_experiment_config
from .generate_tasks import validate_task_candidate, write_jsonl
from .manifest import canonical_json, load_manifest
from .model_client import OpenAICompatibleClient, read_prompt
from .synthetic_workspace import verify_task_assets


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            records.append(record)
    return records


def load_usable_tools(
    path: str | Path,
    expected_manifest_hash: str,
) -> set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        tools = payload
    elif isinstance(payload, dict):
        if payload.get("tool_manifest_hash") != expected_manifest_hash:
            raise ValueError("usable-tools file belongs to another manifest")
        tools = payload.get("usable_tool_ids")
    else:
        raise ValueError("usable-tools file must be a list or object")
    if not isinstance(tools, list) or any(not isinstance(item, str) for item in tools):
        raise ValueError("usable_tool_ids must be a list of strings")
    return set(tools)


def _word_ngrams(text: str, width: int = 5) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {
        tuple(tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    }


def protected_ngram_audit(
    instruction: str,
    protected_task_root: str | Path,
    task_ids: list[str],
    *,
    threshold: float,
) -> dict[str, Any]:
    candidate = _word_ngrams(instruction)
    maximum = 0.0
    files_checked = 0
    root = Path(protected_task_root)
    for task_id in task_ids:
        for filename in ("docs/task.md", "docs/task_cn.md"):
            path = root / task_id / filename
            if not path.is_file():
                continue
            protected = _word_ngrams(path.read_text(encoding="utf-8"))
            union = candidate | protected
            score = len(candidate & protected) / len(union) if union else 0.0
            maximum = max(maximum, score)
            files_checked += 1
    if files_checked == 0:
        raise ValueError("protected leakage audit did not find any task prompt files")
    return {
        "passed": maximum < threshold,
        "metric": "word_5gram_jaccard",
        "max_similarity": maximum,
        "threshold": threshold,
        "protected_files_checked": files_checked,
        "matched_text_not_recorded": True,
    }


async def verify_candidates(
    *,
    candidates: list[dict[str, Any]],
    manifest: dict[str, Any],
    usable_tool_ids: set[str],
    protected_task_root: str | Path,
    config: dict[str, Any],
    client: OpenAICompatibleClient,
    ngram_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system_prompt = read_prompt(PACKAGE_DIR / "prompts" / config["verifier"]["prompt"])
    model = config["models"]["verifier"]
    accepted = []
    rejected = []
    for candidate in candidates:
        reasons = []
        program_checks: dict[str, Any] = {}
        try:
            validate_task_candidate(candidate, manifest)
            unavailable = sorted(set(candidate["available_tools"]) - usable_tool_ids)
            if unavailable:
                reasons.append(
                    "available tools lack a successful environment smoke check: "
                    + ", ".join(unavailable)
                )
            program_checks["assets"] = verify_task_assets(candidate)
            if not program_checks["assets"]["passed"]:
                reasons.extend(program_checks["assets"]["reasons"])
            program_checks["leakage"] = protected_ngram_audit(
                candidate["instruction"],
                protected_task_root,
                list(load_experiment_config().task_ids),
                threshold=ngram_threshold,
            )
            if not program_checks["leakage"]["passed"]:
                reasons.append("instruction failed protected-task n-gram rejection gate")
        except Exception as exc:
            reasons.append(f"program verifier: {type(exc).__name__}: {exc}")

        model_verdict: dict[str, Any] | None = None
        if not reasons:
            model_payload = {
                "task": candidate,
                "manifest_tool_ids": [
                    record["stable_id"] for record in manifest["tools"]
                ],
                "program_checks": program_checks,
            }
            try:
                model_verdict = await client.request_json(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=canonical_json(model_payload),
                    max_completion_tokens=int(
                        config["verifier"]["max_completion_tokens"]
                    ),
                )
                if model_verdict.get("verdict") != "PASS":
                    verifier_reasons = model_verdict.get("reasons", [])
                    reasons.append(
                        "independent model verifier rejected: "
                        + "; ".join(str(reason) for reason in verifier_reasons)
                    )
            except Exception as exc:
                reasons.append(f"model verifier: {type(exc).__name__}: {exc}")

        result = dict(candidate)
        result["verification"] = {
            "program": program_checks,
            "model": model_verdict,
            "verifier_model": model,
            "verifier_prompt_version": config["verifier"]["prompt_version"],
        }
        result["verified"] = not reasons
        result["verification_reasons"] = reasons
        (accepted if not reasons else rejected).append(result)
    return accepted, rejected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply program, leakage, usability, and independent-LLM task gates"
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--usable-tools", required=True)
    parser.add_argument("--protected-task-root", required=True)
    parser.add_argument("--accepted-output", required=True)
    parser.add_argument("--rejected-output", required=True)
    parser.add_argument(
        "--generation-config",
        default=str(PACKAGE_DIR / "configs" / "generation.json"),
    )
    parser.add_argument("--ngram-threshold", type=float, default=0.20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.ngram_threshold <= 1.0:
        raise ValueError("--ngram-threshold must be in [0, 1]")
    manifest = load_manifest(args.manifest)
    usable = load_usable_tools(args.usable_tools, manifest["manifest_hash"])
    config = json.loads(Path(args.generation_config).read_text(encoding="utf-8"))
    accepted, rejected = asyncio.run(
        verify_candidates(
            candidates=read_jsonl(args.candidates),
            manifest=manifest,
            usable_tool_ids=usable,
            protected_task_root=args.protected_task_root,
            config=config,
            client=OpenAICompatibleClient.from_env(),
            ngram_threshold=args.ngram_threshold,
        )
    )
    write_jsonl(Path(args.accepted_output), accepted)
    write_jsonl(Path(args.rejected_output), rejected)
    print(f"accepted={len(accepted)} rejected={len(rejected)}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
