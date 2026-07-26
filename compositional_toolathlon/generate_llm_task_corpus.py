from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .config import PACKAGE_DIR
from .generate_tasks import (
    ALLOWED_SPLITS,
    generate_candidates,
    load_generation_config,
    resolve_tool_names,
    write_jsonl,
)
from .manifest import load_manifest
from .model_client import OpenAICompatibleClient


DEFAULT_COVERAGE_GROUPS = (
    PACKAGE_DIR / "configs" / "llm_v1_coverage_groups.json"
)
DEFAULT_GENERATION_CONFIG = PACKAGE_DIR / "configs" / "generation_llm_v1.json"
DEFAULT_MANIFEST = (
    PACKAGE_DIR / "data" / "llm_v1" / "manifests" / "tool_manifest.json"
)
DEFAULT_OUTPUT_ROOT = PACKAGE_DIR / "data" / "llm_v1"


def load_coverage_groups(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)["groups"]


async def generate_task_corpus(
    *,
    client: Any,
    manifest: dict[str, Any],
    config: dict[str, Any],
    coverage_groups: list[dict[str, Any]],
    output_root: str | Path,
    group_ids: list[str],
    base_seed: int,
    split: str,
) -> list[dict[str, Any]]:
    selected_ids = set(group_ids)
    available_ids = {group["group_id"] for group in coverage_groups}
    unknown_ids = sorted(selected_ids - available_ids)
    if unknown_ids:
        raise ValueError(f"unknown coverage group IDs: {unknown_ids}")

    candidates_dir = Path(output_root) / "tasks" / "candidates"
    all_candidates: list[dict[str, Any]] = []
    for group_index, group in enumerate(coverage_groups):
        group_id = group["group_id"]
        if selected_ids and group_id not in selected_ids:
            continue
        required_tool_ids = resolve_tool_names(
            manifest,
            group["must_require_tool_names"],
        )
        candidates = await generate_candidates(
            client=client,
            manifest=manifest,
            config=config,
            task_family=group["task_family"],
            count_per_session=1,
            base_seed=base_seed + group_index,
            split=split,
            must_require_tool_ids=required_tool_ids,
        )
        for candidate in candidates:
            candidate["generation_provenance"]["coverage_group_id"] = group_id
        write_jsonl(candidates_dir / f"{group_id}.jsonl", candidates)
        all_candidates.extend(candidates)

    write_jsonl(candidates_dir / "all.jsonl", all_candidates)
    return all_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the grouped LLM task corpus"
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--coverage-groups",
        default=str(DEFAULT_COVERAGE_GROUPS),
    )
    parser.add_argument(
        "--generation-config",
        default=str(DEFAULT_GENERATION_CONFIG),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--group-id",
        action="append",
        default=[],
        help="generate only this coverage group; repeat to select more groups",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=sorted(ALLOWED_SPLITS),
        default="train",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = load_manifest(args.manifest)
    config = load_generation_config(args.generation_config)
    coverage_groups = load_coverage_groups(args.coverage_groups)
    client = OpenAICompatibleClient.from_env()
    candidates = asyncio.run(
        generate_task_corpus(
            client=client,
            manifest=manifest,
            config=config,
            coverage_groups=coverage_groups,
            output_root=args.output_root,
            group_ids=args.group_id,
            base_seed=args.base_seed,
            split=args.split,
        )
    )
    print(
        f"Wrote {len(candidates)} unverified task candidates under "
        f"{Path(args.output_root) / 'tasks' / 'candidates'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
