from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import PACKAGE_DIR
from .manifest import canonical_json, load_manifest
from .model_client import OpenAICompatibleClient, read_prompt
from .synthetic_workspace import validate_task_spec_structure


DEFAULT_GENERATION_CONFIG = PACKAGE_DIR / "configs" / "generation.json"
TASK_SPEC_RESPONSE_SCHEMA = (
    PACKAGE_DIR / "configs" / "codex_task_spec_schema.json"
)
ALLOWED_SPLITS = {"train", "validation", "synthetic_test"}
TASK_SPEC_CONTRACT = {
    "top_level_required": {
        "task_id": "unique lowercase-kebab-case string",
        "template_id": "template-family identifier",
        "instruction": "complete natural-language user request",
        "available_tools": [
            "stable tool IDs, including required_terminal_tool_id"
        ],
        "intended_required_tools": [
            "stable IDs genuinely needed to solve the task"
        ],
        "distractor_tools": [
            "at least 3 available stable IDs, disjoint from intended tools"
        ],
        "initial_workspace": "workspace_recipe",
        "oracle_final_state": "workspace_recipe",
        "evaluator": "workspace_assertions_v1",
    },
    "workspace_recipe": {
        "directories": ["relative/path"],
        "files": [
            {
                "path": "relative/path.ext",
                "format": "text|csv|xlsx|pdf|base64",
                "content": (
                    "text=string; csv=list of row lists; "
                    "pdf=list of page strings; base64=base64 string; "
                    "xlsx={sheets:[{name,rows:list of row lists,"
                    "styles:[{start_cell,end_cell,bold,font_color,bg_color,"
                    "alignment}],tables:[{data_range,table_name,table_style}],"
                    "charts:[{data_range,chart_type:bar|line|pie|area,"
                    "target_cell,title,x_axis,y_axis}],"
                    "merged_ranges:[A1:D1,...]}]}"
                ),
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
                    "csv_equals|xlsx_cells_equal|xlsx_sheet_names_equal|"
                    "xlsx_table_names_equal|xlsx_table_properties|"
                    "xlsx_table_definitions_equal|"
                    "xlsx_chart_count|xlsx_chart_properties|xlsx_cell_style|"
                    "xlsx_merged_ranges_equal|pdf_text_contains|pdf_page_count|"
                    "pdf_page_text_contains|pdf_page_text_equals|sha256|"
                    "file_size_at_least|tree_equals"
                ),
                "path": "relative/path unless tree_equals",
            }
        ],
    },
    "assertion_arguments": {
        "exists": "{op,path,kind?:file|directory}",
        "absent": "{op,path}",
        "text_equals|text_contains|json_equals|sha256": "{op,path,value:string}",
        "csv_equals": "{op,path,rows:list of row lists}",
        "file_size_at_least": "{op,path,bytes:nonnegative integer}",
        "tree_equals": "{op,paths:list of relative paths}",
        "xlsx_cells_equal": (
            "{op,path,sheet,cells:[{cell:A1,value:...},...]}"
        ),
        "xlsx_sheet_names_equal": "{op,path,sheets:[name,...]}",
        "xlsx_table_names_equal": "{op,path,sheet,tables:[name,...]}",
        "xlsx_table_properties": (
            "{op,path,sheet,table_name,data_range,table_style}"
        ),
        "xlsx_table_definitions_equal": (
            "{op,path,sheet,tables:[{data_range,table_style},...]}"
        ),
        "xlsx_chart_count": "{op,path,sheet,count}",
        "xlsx_chart_properties": (
            "{op,path,sheet,index,properties:{chart_type,data_range,"
            "target_cell,title,x_axis,y_axis}}"
        ),
        "xlsx_cell_style": "{op,path,sheet,cell,properties}",
        "xlsx_merged_ranges_equal": "{op,path,sheet,ranges:[A1:B2,...]}",
        "pdf_text_contains": "{op,path,value}",
        "pdf_page_count": "{op,path,count}",
        "pdf_page_text_contains|pdf_page_text_equals": (
            "{op,path,page:1-indexed integer,value:string}"
        ),
    },
}


def normalize_structured_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Convert strict-schema wire shapes into the stored task-spec shapes."""

    normalized = copy.deepcopy(candidate)
    for recipe_name in ("initial_workspace", "oracle_final_state"):
        recipe = normalized.get(recipe_name)
        if not isinstance(recipe, dict):
            continue
        for file_spec in recipe.get("files", []):
            if file_spec.get("format") != "xlsx":
                continue
            for sheet in file_spec.get("content", {}).get("sheets", []):
                for style in sheet.get("styles", []):
                    for key in ("bold", "font_color", "bg_color", "alignment"):
                        if style.get(key) is None:
                            style.pop(key, None)
                for chart in sheet.get("charts", []):
                    for key in ("title", "x_axis", "y_axis"):
                        if chart.get(key) is None:
                            chart.pop(key, None)

    evaluator = normalized.get("evaluator")
    if isinstance(evaluator, dict):
        for assertion in evaluator.get("assertions", []):
            if assertion.get("op") in {
                "xlsx_cells_equal",
                "xlsx_nonempty_cells_equal",
            } and isinstance(assertion.get("cells"), list):
                coordinates = [
                    item["cell"].upper()
                    for item in assertion["cells"]
                ]
                if len(coordinates) != len(set(coordinates)):
                    raise ValueError(
                        "structured xlsx cells contain duplicate coordinates"
                    )
                assertion["cells"] = {
                    item["cell"]: item["value"]
                    for item in assertion["cells"]
                }
            if assertion.get("op") == "exists" and assertion.get("kind") is None:
                assertion.pop("kind", None)
            if assertion.get("op") == "xlsx_cell_style":
                properties = assertion.get("properties")
                if isinstance(properties, dict):
                    assertion["properties"] = {
                        key: value
                        for key, value in properties.items()
                        if value is not None
                    }
            if assertion.get("op") == "json_equals" and isinstance(
                assertion.get("value"),
                str,
            ):
                assertion["value"] = json.loads(assertion["value"])
    return normalized


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


def resolve_tool_names(
    manifest: dict[str, Any],
    tool_names: list[str],
) -> list[str]:
    records_by_name: dict[str, list[dict[str, Any]]] = {}
    for record in manifest["tools"]:
        records_by_name.setdefault(record["tool_name"], []).append(record)

    stable_ids = []
    for tool_name in tool_names:
        matches = records_by_name.get(tool_name, [])
        if not matches:
            raise ValueError(f"tool name is absent from manifest: {tool_name!r}")
        if len(matches) != 1:
            raise ValueError(f"tool name is not unique in manifest: {tool_name!r}")
        stable_id = matches[0]["stable_id"]
        if stable_id not in stable_ids:
            stable_ids.append(stable_id)
    return stable_ids


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
    must_require_tool_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported split: {split!r}")
    model = config["models"]["generator"]
    compact_manifest = compact_manifest_for_generator(manifest)
    terminal_ids = [
        record["stable_id"]
        for record in manifest["tools"]
        if record["tool_name"] == "local-claim_done"
    ]
    if len(terminal_ids) != 1:
        raise ValueError("manifest must contain exactly one local-claim_done tool")
    terminal_tool_id = terminal_ids[0]
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
                "required_terminal_tool_id": terminal_tool_id,
            }
            if must_require_tool_ids:
                user_payload["must_require_tool_ids"] = must_require_tool_ids
            rendered_user_prompt = canonical_json(user_payload)
            requests.append(
                client.request_json(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=rendered_user_prompt,
                    max_completion_tokens=int(session["max_completion_tokens"]),
                    response_schema=TASK_SPEC_RESPONSE_SCHEMA,
                )
            )
            provenance.append(
                {
                    "generator_model": model,
                    "generator_provider": type(client).__name__,
                    "generator_prompt_version": prompt_path.stem,
                    "session_id": session["session_id"],
                    "seed": seed,
                    "tool_manifest_hash": manifest["manifest_hash"],
                    "must_require_tool_ids": list(must_require_tool_ids or ()),
                    "system_prompt_sha256": hashlib.sha256(
                        system_prompt.encode("utf-8")
                    ).hexdigest(),
                    "user_prompt_sha256": hashlib.sha256(
                        rendered_user_prompt.encode("utf-8")
                    ).hexdigest(),
                }
            )

    results = await asyncio.gather(*requests)
    candidates = []
    for result, generation_provenance in zip(results, provenance):
        candidate = normalize_structured_candidate(result)
        candidate["task_family"] = task_family
        candidate["split"] = split
        candidate["asset_seed"] = generation_provenance["seed"]
        candidate["generation_provenance"] = generation_provenance
        validate_task_candidate(candidate, manifest)
        if terminal_tool_id not in candidate["available_tools"]:
            raise ValueError(
                f"generated task {candidate['task_id']!r} omitted local-claim_done"
            )
        missing_required = sorted(
            set(must_require_tool_ids or ())
            - set(candidate["intended_required_tools"])
        )
        if missing_required:
            raise ValueError(
                f"generated task {candidate['task_id']!r} omitted required tool IDs: "
                f"{missing_required}"
            )
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
    parser.add_argument(
        "--must-require-tool-name",
        action="append",
        default=[],
        help=(
            "require the generated task to use this manifest tool_name; "
            "repeat for multiple tools"
        ),
    )
    parser.add_argument("--generation-config", default=str(DEFAULT_GENERATION_CONFIG))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.count_per_session <= 0:
        raise ValueError("--count-per-session must be positive")
    manifest = load_manifest(args.manifest)
    must_require_tool_ids = resolve_tool_names(
        manifest,
        args.must_require_tool_name,
    )
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
            must_require_tool_ids=must_require_tool_ids,
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
