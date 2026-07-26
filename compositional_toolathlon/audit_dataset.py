from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .episode_to_steps import (
    ALLOWED_SPLITS,
    flatten_episode,
    read_records,
    record_content_hash,
    validate_episode,
    validate_group_splits,
)
from .generate_tasks import validate_task_candidate
from .manifest import load_manifest, validate_tool_arguments
from .target_policy import (
    TARGET_POLICY_HASH,
    TARGET_POLICY_NAME,
    TARGET_POLICY_VERSION,
    TARGET_TOOL_NAMES,
    validate_target_tools_report,
)
from .verify_tasks import read_jsonl


def _shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _shape(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        return ["list", sorted({json.dumps(_shape(item), sort_keys=True) for item in value})]
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return type(value).__name__


def argument_shape(arguments: dict[str, Any]) -> str:
    return json.dumps(_shape(arguments), sort_keys=True, separators=(",", ":"))


def episode_id_hash(values: set[str] | list[str]) -> str:
    material = json.dumps(sorted(values), separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def semantic_task_content_hash(tasks: list[dict[str, Any]]) -> str:
    material = json.dumps(
        sorted(
            (
                {
                    "task_id": task["task_id"],
                    "instruction": task["instruction"],
                }
                for task in tasks
            ),
            key=lambda record: record["task_id"],
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _instruction_ngrams(text: str, width: int = 5) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {
        tuple(tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    }


def _asset_layout_signature(task: dict[str, Any]) -> str:
    files = []
    for record in task["initial_workspace"].get("files", []):
        path = Path(record["path"])
        files.append(
            {
                "depth": len(path.parts),
                "suffix": path.suffix.casefold(),
                "format": record["format"],
            }
        )
    material = {
        "files": sorted(files, key=lambda value: json.dumps(value, sort_keys=True)),
        "directory_depths": sorted(
            len(Path(path).parts)
            for path in task["initial_workspace"].get("directories", [])
        ),
        "evaluator_ops": sorted(
            assertion["op"] for assertion in task["evaluator"]["assertions"]
        ),
        "intended_tools": sorted(task["intended_required_tools"]),
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def audit_synthetic_task_splits(
    task_specs: list[dict[str, Any]],
    *,
    instruction_ngram_threshold: float = 0.50,
) -> dict[str, Any]:
    template_splits: dict[str, set[str]] = defaultdict(set)
    layout_splits: dict[str, set[str]] = defaultdict(set)
    plan_signature_splits: dict[str, set[str]] = defaultdict(set)
    asset_seeds = set()
    for task in task_specs:
        template_splits[task["template_id"]].add(task["split"])
        layout_splits[_asset_layout_signature(task)].add(task["split"])
        generation_provenance = task["generation_provenance"]
        if "plan_signature" in generation_provenance:
            plan_signature = generation_provenance["plan_signature"]
            if not isinstance(plan_signature, str) or not plan_signature.strip():
                raise ValueError(
                    "generation_provenance.plan_signature must be a non-empty string"
                )
            plan_signature_splits[plan_signature].add(task["split"])
        provenance_key = (
            generation_provenance.get("session_id"),
            task["asset_seed"],
        )
        if provenance_key in asset_seeds:
            raise ValueError(f"duplicate generator session/asset seed: {provenance_key}")
        asset_seeds.add(provenance_key)
    failures = []
    for template_id, splits in template_splits.items():
        if len(splits) > 1:
            failures.append(
                {
                    "kind": "template_cross_split",
                    "template_id": template_id,
                    "splits": sorted(splits),
                }
            )
    for signature, splits in layout_splits.items():
        if len(splits) > 1:
            failures.append(
                {
                    "kind": "asset_layout_cross_split",
                    "signature": signature,
                    "splits": sorted(splits),
                }
            )
    for signature, splits in plan_signature_splits.items():
        if len(splits) > 1:
            failures.append(
                {
                    "kind": "plan_signature_cross_split",
                    "signature": signature,
                    "splits": sorted(splits),
                }
            )
    for left_index, left in enumerate(task_specs):
        left_ngrams = _instruction_ngrams(left["instruction"])
        for right in task_specs[left_index + 1 :]:
            if left["split"] == right["split"]:
                continue
            right_ngrams = _instruction_ngrams(right["instruction"])
            union = left_ngrams | right_ngrams
            similarity = (
                len(left_ngrams & right_ngrams) / len(union) if union else 0.0
            )
            if similarity >= instruction_ngram_threshold:
                failures.append(
                    {
                        "kind": "instruction_near_duplicate_cross_split",
                        "left_task_id": left["task_id"],
                        "right_task_id": right["task_id"],
                        "similarity": similarity,
                        "threshold": instruction_ngram_threshold,
                    }
                )
    return {
        "passed": not failures,
        "instruction_metric": "word_5gram_jaccard",
        "instruction_threshold": instruction_ngram_threshold,
        "failures": failures,
    }


def audit_dataset(
    *,
    episodes: list[dict[str, Any]],
    task_specs: list[dict[str, Any]],
    manifest: dict[str, Any],
    target_tool_ids: set[str],
    min_successful_episodes: int,
    min_argument_shapes: int,
    min_templates: int,
    require_distractor_role: bool,
    require_real_execution: bool = False,
    include_rejected: bool = False,
    include_failed_calls: bool = False,
) -> dict[str, Any]:
    validate_group_splits(episodes)
    manifest_records = {
        record["stable_id"]: record for record in manifest["tools"]
    }
    for episode in episodes:
        if require_real_execution:
            teacher = episode.get("teacher")
            if (
                not isinstance(teacher, dict)
                or teacher.get("real_execution") is not True
            ):
                raise ValueError(
                    "formal dataset audit requires "
                    "teacher.real_execution=true for every episode"
                )
        validate_episode(
            episode,
            require_clean=not include_failed_calls,
            include_rejected=include_rejected,
        )
        if episode["tool_manifest_hash"] != manifest["manifest_hash"]:
            raise ValueError("episode uses a different manifest hash")
        for index in range(0, len(episode["messages"]), 2):
            call = episode["messages"][index]
            record = manifest_records.get(call["tool_id"])
            if record is None:
                raise ValueError(
                    f"episode calls tool absent from manifest: {call['tool_id']}"
                )
            validate_tool_arguments(record, call["arguments"])

    task_by_id = {}
    for task in task_specs:
        validate_task_candidate(task, manifest)
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or task_id in task_by_id:
            raise ValueError(f"duplicate or invalid verified task ID: {task_id!r}")
        if task.get("verified") is not True:
            raise ValueError(f"task {task_id} is not verified")
        task_by_id[task_id] = task
    split_audit = audit_synthetic_task_splits(task_specs)

    calls = Counter()
    successful_episodes: dict[str, set[str]] = defaultdict(set)
    calls_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    successful_episodes_by_split: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    shapes: dict[str, set[str]] = defaultdict(set)
    templates: dict[str, set[str]] = defaultdict(set)
    shapes_by_split: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    templates_by_split: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    step_bins = Counter()
    splits = Counter()
    split_episode_ids: dict[str, set[str]] = defaultdict(set)
    split_steps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        task = task_by_id.get(episode["task_id"])
        if task is None:
            raise ValueError(f"episode task has no verified task spec: {episode['task_id']}")
        episode_task_pairs = (
            ("template_id", "template_id"),
            ("asset_seed", "asset_seed"),
            ("split", "split"),
            ("instruction", "instruction"),
            ("available_tool_ids", "available_tools"),
        )
        for episode_field, task_field in episode_task_pairs:
            if episode.get(episode_field) != task.get(task_field):
                raise ValueError(
                    f"episode {episode['episode_id']} field {episode_field!r} "
                    f"does not match verified task {task_field!r}"
                )
        call_count = len(episode["messages"]) // 2
        if 2 <= call_count <= 4:
            step_bins["2-4"] += 1
        elif 5 <= call_count <= 8:
            step_bins["5-8"] += 1
        elif 9 <= call_count <= 15:
            step_bins["9-15"] += 1
        else:
            step_bins["outside_preregistered_bins"] += 1
        splits[episode["split"]] += 1
        split_episode_ids[episode["split"]].add(episode["episode_id"])
        split_steps[episode["split"]].extend(
            flatten_episode(
                episode,
                require_clean=not include_failed_calls,
                include_rejected=include_rejected,
            )
        )
        for index in range(0, len(episode["messages"]), 2):
            call = episode["messages"][index]
            observation = episode["messages"][index + 1]
            if observation.get("success") is not True:
                continue
            tool_id = call["tool_id"]
            calls[tool_id] += 1
            successful_episodes[tool_id].add(episode["episode_id"])
            calls_by_split[episode["split"]][tool_id] += 1
            successful_episodes_by_split[episode["split"]][tool_id].add(
                episode["episode_id"]
            )
            shapes[tool_id].add(argument_shape(call["arguments"]))
            templates[tool_id].add(episode["template_id"])
            shapes_by_split[episode["split"]][tool_id].add(
                argument_shape(call["arguments"])
            )
            templates_by_split[episode["split"]][tool_id].add(
                episode["template_id"]
            )

    distractor_roles: dict[str, set[str]] = defaultdict(set)
    intended_roles: dict[str, set[str]] = defaultdict(set)
    train_distractor_roles: dict[str, set[str]] = defaultdict(set)
    train_intended_roles: dict[str, set[str]] = defaultdict(set)
    for task in task_specs:
        for tool_id in task["distractor_tools"]:
            distractor_roles[tool_id].add(task["template_id"])
            if task["split"] == "train":
                train_distractor_roles[tool_id].add(task["template_id"])
        for tool_id in task["intended_required_tools"]:
            intended_roles[tool_id].add(task["template_id"])
            if task["split"] == "train":
                train_intended_roles[tool_id].add(task["template_id"])

    known_tools = {record["stable_id"] for record in manifest["tools"]}
    unknown_targets = sorted(target_tool_ids - known_tools)
    if unknown_targets:
        raise ValueError(f"coverage target list contains unknown tools: {unknown_targets}")
    per_tool = {}
    failures = []
    for tool_id in sorted(target_tool_ids):
        record = next(
            record for record in manifest["tools"] if record["stable_id"] == tool_id
        )
        metrics = {
            "tool_id": tool_id,
            "dispatch_kind": record["dispatch_kind"],
            "successful_call_count": calls[tool_id],
            "successful_episode_count": len(successful_episodes[tool_id]),
            "successful_call_count_by_split": {
                split: calls_by_split[split][tool_id]
                for split in ALLOWED_SPLITS
            },
            "successful_episode_count_by_split": {
                split: len(successful_episodes_by_split[split][tool_id])
                for split in ALLOWED_SPLITS
            },
            "train_successful_call_count": calls_by_split["train"][tool_id],
            "train_successful_episode_count": len(
                successful_episodes_by_split["train"][tool_id]
            ),
            "argument_shape_count": len(shapes[tool_id]),
            "actual_template_count": len(templates[tool_id]),
            "intended_template_count": len(intended_roles[tool_id]),
            "distractor_template_count": len(distractor_roles[tool_id]),
            "train_argument_shape_count": len(
                shapes_by_split["train"][tool_id]
            ),
            "train_actual_template_count": len(
                templates_by_split["train"][tool_id]
            ),
            "train_intended_template_count": len(
                train_intended_roles[tool_id]
            ),
            "train_distractor_template_count": len(
                train_distractor_roles[tool_id]
            ),
        }
        reasons = []
        if metrics["train_successful_episode_count"] < min_successful_episodes:
            reasons.append("train_successful_episode_count")
        if metrics["train_argument_shape_count"] < min_argument_shapes:
            reasons.append("train_argument_shape_count")
        if metrics["train_actual_template_count"] < min_templates:
            reasons.append("train_actual_template_count")
        if (
            require_distractor_role
            and metrics["train_distractor_template_count"] == 0
        ):
            reasons.append("train_distractor_template_count")
        metrics["passed"] = not reasons
        metrics["failed_thresholds"] = reasons
        per_tool[tool_id] = metrics
        if reasons:
            failures.append({"tool_id": tool_id, "failed_thresholds": reasons})

    return {
        "schema_version": 2,
        "passed": not failures and split_audit["passed"],
        "tool_manifest_hash": manifest["manifest_hash"],
        "target_tool_count": len(target_tool_ids),
        "thresholds": {
            "min_successful_episodes": min_successful_episodes,
            "successful_episode_scope": "train",
            "min_argument_shapes": min_argument_shapes,
            "min_templates": min_templates,
            "require_distractor_role": require_distractor_role,
        },
        "execution_requirements": {
            "real_execution": require_real_execution,
        },
        "training_episode_policy": {
            "include_rejected": include_rejected,
            "include_failed_calls": include_failed_calls,
        },
        "episode_acceptance_counts": dict(
            sorted(
                Counter(
                    "accepted" if episode["accepted"] else "rejected"
                    for episode in episodes
                ).items()
            )
        ),
        "episode_count": len(episodes),
        "split_episode_counts": {
            split: splits[split]
            for split in ALLOWED_SPLITS
        },
        "split_episode_id_hashes": {
            split: episode_id_hash(split_episode_ids[split])
            for split in ALLOWED_SPLITS
        },
        "split_step_counts": {
            split: len(split_steps[split])
            for split in ALLOWED_SPLITS
        },
        "split_step_content_hashes": {
            split: record_content_hash(split_steps[split])
            for split in ALLOWED_SPLITS
        },
        "step_length_bins": dict(sorted(step_bins.items())),
        "per_tool": per_tool,
        "synthetic_split_audit": split_audit,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit actual successful-call coverage before Toolathlon training"
    )
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--verified-tasks", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-tools", required=True)
    parser.add_argument("--semantic-leakage-audit", required=True)
    parser.add_argument(
        "--min-successful-episodes",
        type=int,
        default=3,
        help="minimum distinct successful train episodes required per target tool",
    )
    parser.add_argument("--min-argument-shapes", type=int, default=1)
    parser.add_argument("--min-templates", type=int, default=1)
    parser.add_argument("--require-distractor-role", action="store_true")
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="include complete collector-rejected episodes in the training audit",
    )
    parser.add_argument(
        "--include-failed-calls",
        action="store_true",
        help="include steps whose real tool observation reports failure",
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    threshold_values = {
        "--min-successful-episodes": args.min_successful_episodes,
        "--min-argument-shapes": args.min_argument_shapes,
        "--min-templates": args.min_templates,
    }
    non_positive = {
        name: value
        for name, value in threshold_values.items()
        if value <= 0
    }
    if non_positive:
        raise ValueError(
            "coverage thresholds must be positive integers: "
            + ", ".join(
                f"{name}={value}" for name, value in sorted(non_positive.items())
            )
        )
    if args.min_successful_episodes < 3:
        raise ValueError(
            "--min-successful-episodes must be at least 3 for the frozen "
            "Toolathlon smoke contract"
        )
    if not args.require_distractor_role:
        raise ValueError(
            "--require-distractor-role is mandatory for the frozen "
            "Toolathlon smoke contract"
        )
    manifest = load_manifest(args.manifest)
    target_tools_payload = json.loads(
        Path(args.target_tools).read_text(encoding="utf-8")
    )
    target_tools = validate_target_tools_report(target_tools_payload, manifest)
    report = audit_dataset(
        episodes=read_records(args.episodes),
        task_specs=read_jsonl(args.verified_tasks),
        manifest=manifest,
        target_tool_ids=target_tools,
        min_successful_episodes=args.min_successful_episodes,
        min_argument_shapes=args.min_argument_shapes,
        min_templates=args.min_templates,
        require_distractor_role=args.require_distractor_role,
        require_real_execution=True,
        include_rejected=args.include_rejected,
        include_failed_calls=args.include_failed_calls,
    )
    report.update(
        {
            "target_policy_name": TARGET_POLICY_NAME,
            "target_policy_version": TARGET_POLICY_VERSION,
            "target_policy_hash": TARGET_POLICY_HASH,
            "target_tool_count": len(TARGET_TOOL_NAMES),
        }
    )
    semantic_audit = json.loads(
        Path(args.semantic_leakage_audit).read_text(encoding="utf-8")
    )
    verified_task_ids = [task["task_id"] for task in read_jsonl(args.verified_tasks)]
    expected_task_hash = hashlib.sha256(
        json.dumps(sorted(verified_task_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if semantic_audit.get("task_id_hash") != expected_task_hash:
        raise ValueError("semantic leakage audit covers a different verified task set")
    verified_tasks = read_jsonl(args.verified_tasks)
    if semantic_audit.get("task_content_hash") != semantic_task_content_hash(
        verified_tasks
    ):
        raise ValueError(
            "semantic leakage audit covers different task instructions"
        )
    report["semantic_leakage_audit"] = semantic_audit
    report["passed"] = report["passed"] and semantic_audit.get("passed") is True
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"passed={report['passed']} episodes={report['episode_count']} "
        f"failed_tools={len(report['failures'])}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
