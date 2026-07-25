from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .config import PACKAGE_DIR
from .manifest import canonical_json, load_manifest
from .model_client import OpenAICompatibleClient, read_prompt
from .synthetic_workspace import validate_task_spec_structure


DEFAULT_GENERATION_CONFIG = PACKAGE_DIR / "configs" / "generation.json"
ALLOWED_SPLITS = {"train", "validation", "synthetic_test"}
TASK_SPEC_CONTRACT = {
    "workspace_recipe": {
        "directories": ["relative/path"],
        "files": [
            {
                "path": "relative/path.ext",
                "format": "text|json|csv|xlsx|pdf|base64",
                "content": "format-specific JSON value",
            }
        ],
        "remove": ["relative/path"],
    },
    "evaluator": {
        "type": "workspace_assertions_v1",
        "assertions": [
            {
                "op": (
                    "exists|absent|text_equals|text_contains|json_equals|"
                    "csv_equals|xlsx_cells_equal|pdf_text_contains|sha256|"
                    "file_size_at_least|tree_equals"
                ),
                "path": "relative/path unless tree_equals",
            }
        ],
    },
}


def load_generation_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported generation config schema version")
    sessions = config.get("generator_sessions")
    if not isinstance(sessions, list) or len(sessions) < 3:
        raise ValueError("at least three generator sessions are required")
    prompts = [session.get("prompt") for session in sessions]
    if len(prompts) != len(set(prompts)):
        raise ValueError("generator sessions must use different prompt files")
    return config


def compact_manifest_for_generator(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "stable_id": record["stable_id"],
            "description": record.get("description", ""),
            "input_schema": record["input_schema"],
        }
        for record in manifest["tools"]
    ]


def validate_task_candidate(
    candidate: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> None:
    required = (
        "task_id",
        "task_family",
        "template_id",
        "asset_seed",
        "split",
        "instruction",
        "available_tools",
        "intended_required_tools",
        "distractor_tools",
        "initial_workspace",
        "oracle_final_state",
        "evaluator",
        "generation_provenance",
    )
    missing = [field for field in required if field not in candidate]
    if missing:
        raise ValueError(f"generated task is missing required fields: {missing}")
    if not isinstance(candidate["instruction"], str) or not candidate["instruction"].strip():
        raise ValueError("generated task instruction must be non-empty")
    for field in ("task_id", "template_id"):
        if not isinstance(candidate[field], str) or not candidate[field].strip():
            raise ValueError(f"generated task requires non-empty {field}")
    if not isinstance(candidate["task_family"], str) or not candidate["task_family"].strip():
        raise ValueError("generated task requires non-empty task_family")
    if candidate["split"] not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported generated split: {candidate['split']!r}")
    if not isinstance(candidate["asset_seed"], int):
        raise ValueError("generated task asset_seed must be an integer")
    if not isinstance(candidate["generation_provenance"], dict):
        raise ValueError("generated task requires generation_provenance")
    tool_sets = {}
    for field in ("available_tools", "intended_required_tools", "distractor_tools"):
        value = candidate[field]
        if (
            not isinstance(value, list)
            or any(not isinstance(tool_id, str) or not tool_id for tool_id in value)
            or len(value) != len(set(value))
        ):
            raise ValueError(f"{field} must contain unique non-empty tool IDs")
        tool_sets[field] = set(value)
    if not tool_sets["intended_required_tools"]:
        raise ValueError("generated task requires at least one intended tool")
    if len(tool_sets["distractor_tools"]) < 3:
        raise ValueError("generated task requires at least three distinct distractor tools")
    available = tool_sets["available_tools"]
    required_tools = tool_sets["intended_required_tools"]
    distractors = tool_sets["distractor_tools"]
    if not required_tools <= available or not distractors <= available:
        raise ValueError("required tools and distractors must be in available_tools")
    if required_tools & distractors:
        raise ValueError("required tools and distractors must be disjoint")
    if manifest is not None:
        known_tools = {record["stable_id"] for record in manifest["tools"]}
        unknown = sorted(available - known_tools)
        if unknown:
            raise ValueError(f"generated task references tools absent from manifest: {unknown}")
    if any(key in candidate for key in ("trajectory", "messages", "solution")):
        raise ValueError("generator must not emit a solution trajectory")
    validate_task_spec_structure(candidate)


async def generate_candidates(
    *,
    client: OpenAICompatibleClient,
    manifest: dict[str, Any],
    config: dict[str, Any],
    task_family: str,
    count_per_session: int,
    base_seed: int,
    split: str,
) -> list[dict[str, Any]]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported split: {split!r}")
    model = config["models"]["generator"]
    compact_manifest = compact_manifest_for_generator(manifest)
    requests = []
    provenance = []
    for session in config["generator_sessions"]:
        prompt_path = PACKAGE_DIR / "prompts" / session["prompt"]
        system_prompt = read_prompt(prompt_path)
        for item_index in range(count_per_session):
            seed = base_seed + int(session["seed_offset"]) + item_index
            user_payload = {
                "task_family": task_family,
                "seed": seed,
                "tool_manifest_hash": manifest["manifest_hash"],
                "tools": compact_manifest,
                "required_split": split,
                "task_spec_contract": TASK_SPEC_CONTRACT,
            }
            requests.append(
                client.request_json(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=canonical_json(user_payload),
                    max_completion_tokens=int(session["max_completion_tokens"]),
                )
            )
            provenance.append(
                {
                    "generator_model": model,
                    "generator_prompt_version": prompt_path.stem,
                    "session_id": session["session_id"],
                    "seed": seed,
                    "tool_manifest_hash": manifest["manifest_hash"],
                }
            )

    results = await asyncio.gather(*requests)
    candidates = []
    for result, generation_provenance in zip(results, provenance):
        candidate = dict(result)
        candidate["task_family"] = task_family
        candidate["split"] = split
        candidate["asset_seed"] = generation_provenance["seed"]
        candidate["generation_provenance"] = generation_provenance
        validate_task_candidate(candidate, manifest)
        candidates.append(candidate)
    task_ids = [candidate["task_id"] for candidate in candidates]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("generator sessions emitted duplicate task_id values")
    return candidates


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record))
            handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate original Toolathlon-style synthetic task candidates"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-family", required=True)
    parser.add_argument("--count-per-session", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=sorted(ALLOWED_SPLITS),
        default="train",
    )
    parser.add_argument("--generation-config", default=str(DEFAULT_GENERATION_CONFIG))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.count_per_session <= 0:
        raise ValueError("--count-per-session must be positive")
    manifest = load_manifest(args.manifest)
    config = load_generation_config(args.generation_config)
    client = OpenAICompatibleClient.from_env()
    candidates = asyncio.run(
        generate_candidates(
            client=client,
            manifest=manifest,
            config=config,
            task_family=args.task_family,
            count_per_session=args.count_per_session,
            base_seed=args.base_seed,
            split=args.split,
        )
    )
    write_jsonl(Path(args.output), candidates)
    print(
        f"Wrote {len(candidates)} unverified task candidates to {args.output}; "
        "do not train on them before execution and verifier gates pass."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
