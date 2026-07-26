"""Source grouping, InterCode provenance retrieval, and leak-free split."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .data_sources import EXPECTED_INTERCODE_COUNTS
from .io_utils import (
    ensure_output_path,
    sha256_file,
    stable_id,
    write_json,
    write_jsonl,
)
from .templates import (
    bashlex_utility_operator_sequence,
    command_template_keys,
    levenshtein_distance,
    load_bashlint_template_function,
    load_official_basic_tokenizer,
    normalize_instruction,
    token_jaccard,
)

MAX_TEMPLATE_SIGNATURE_DF = 64
MIN_STRUCTURE_INSTRUCTION_JACCARD = 0.25
MIN_STRUCTURE_UTILITY_COUNT = 2
MAX_CANDIDATE_ROWS_PER_TASK = 512
MAX_INTERCODE_DENY_ROW_FRACTION = 0.50
MIN_DERIVED_ACTIVE_ROW_FRACTION = 0.50
OFFICIAL_INTERCODE_TASK_IDS = frozenset(
    f"{fs_id}:{index:03d}"
    for fs_id, count in EXPECTED_INTERCODE_COUNTS.items()
    for index in range(count)
)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _union_by_keys(
    union_find: UnionFind,
    keyed_records: Sequence[tuple[int, Sequence[str]]],
) -> dict[str, int]:
    first_seen: dict[str, int] = {}
    edge_count = 0
    for index, keys in keyed_records:
        for key in keys:
            previous = first_seen.setdefault(key, index)
            if previous != index:
                union_find.union(previous, index)
                edge_count += 1
    return {"keys": len(first_seen), "edges": edge_count}


def _component_ids(
    union_find: UnionFind,
    records: Sequence[Mapping[str, Any]],
    namespace: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    members: dict[int, list[str]] = {}
    for index, record in enumerate(records):
        root = union_find.find(index)
        members.setdefault(root, []).append(record["sample_id"])
    id_by_root = {
        root: stable_id(namespace, "\n".join(sorted(sample_ids)))
        for root, sample_ids in members.items()
    }
    row_ids = [id_by_root[union_find.find(index)] for index in range(len(records))]
    components = [
        {
            "component_id": id_by_root[root],
            "member_sample_ids": sorted(members[root]),
            "size": len(members[root]),
        }
        for root in sorted(members, key=lambda value: id_by_root[value])
    ]
    return row_ids, components


def _template_signature(keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(str(key) for key in keys)))


def _template_signature_key(signature: Sequence[str]) -> str:
    return "TEMPLATE_SIGNATURE:" + json.dumps(
        list(signature),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _exact_pair_key(normalized_instruction: str, command: str) -> str:
    return "EXACT_PAIR:" + json.dumps(
        [normalized_instruction, command],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_source_groups(
    raw_records: Sequence[Mapping[str, Any]],
    nl2bash_root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    basic_tokenizer = load_official_basic_tokenizer(nl2bash_root)
    bashlint_template = load_bashlint_template_function(nl2bash_root)
    enriched: list[dict[str, Any]] = []
    for record in raw_records:
        normalized = normalize_instruction(record["instruction_raw"], basic_tokenizer)
        template_keys = command_template_keys(
            record["command_raw"],
            bashlint_template=bashlint_template,
        )
        signature = _template_signature(template_keys)
        sequence = bashlex_utility_operator_sequence(record["command_raw"])
        enriched.append(
            {
                **record,
                "normalized_instruction": normalized,
                "template_keys": list(template_keys),
                "template_signature": list(signature),
                "utility_operator_sequence": list(sequence) if sequence is not None else None,
            }
        )

    signature_counts = collections.Counter(
        tuple(record["template_signature"])
        for record in enriched
        if record["template_signature"]
    )
    suppressed_signatures = {
        signature
        for signature, count in signature_counts.items()
        if count > MAX_TEMPLATE_SIGNATURE_DF
    }
    for record in enriched:
        signature = tuple(record["template_signature"])
        record["template_signature_df"] = int(signature_counts.get(signature, 0))
        record["template_signature_suppressed"] = (
            not signature or signature in suppressed_signatures
        )

    template_union = UnionFind(len(enriched))
    template_edges = _union_by_keys(
        template_union,
        [
            (
                index,
                [_template_signature_key(record["template_signature"])]
                if not record["template_signature_suppressed"]
                else [],
            )
            for index, record in enumerate(enriched)
        ],
    )
    template_ids, template_components = _component_ids(
        template_union,
        enriched,
        "TEMPLATE_GRAPH_V2",
    )

    source_union = UnionFind(len(enriched))
    template_source_edges = _union_by_keys(
        source_union,
        [
            (
                index,
                [_template_signature_key(record["template_signature"])]
                if not record["template_signature_suppressed"]
                else [],
            )
            for index, record in enumerate(enriched)
        ],
    )
    exact_pair_edges = _union_by_keys(
        source_union,
        [
            (
                index,
                [
                    _exact_pair_key(
                        record["normalized_instruction"],
                        record["command_raw"],
                    )
                ],
            )
            for index, record in enumerate(enriched)
        ],
    )
    source_ids, source_components = _component_ids(
        source_union,
        enriched,
        "SOURCE_GRAPH_V2",
    )
    largest_template_component = max(
        (component["size"] for component in template_components),
        default=0,
    )
    largest_source_component = max(
        (component["size"] for component in source_components),
        default=0,
    )
    if largest_template_component > MAX_TEMPLATE_SIGNATURE_DF:
        raise RuntimeError(
            "A non-hub template component exceeded the declared cap: "
            f"{largest_template_component} > {MAX_TEMPLATE_SIGNATURE_DF}"
        )
    if largest_source_component > MAX_TEMPLATE_SIGNATURE_DF:
        raise RuntimeError(
            "A source component exceeded the declared cap: "
            f"{largest_source_component} > {MAX_TEMPLATE_SIGNATURE_DF}"
        )
    for index, record in enumerate(enriched):
        record["template_group_id"] = template_ids[index]
        record["source_group_id"] = source_ids[index]

    report = {
        "schema": "nl2bash_groups_v2",
        "rows": len(enriched),
        "max_template_signature_df": MAX_TEMPLATE_SIGNATURE_DF,
        "suppressed_signature_count": len(suppressed_signatures),
        "suppressed_row_count": sum(
            record["template_signature_suppressed"] for record in enriched
        ),
        "template_graph": {
            **template_edges,
            "components": len(template_components),
            "largest_component": largest_template_component,
        },
        "source_graph": {
            "template_edges": template_source_edges["edges"],
            "exact_pair_edges": exact_pair_edges["edges"],
            "components": len(source_components),
            "largest_component": largest_source_component,
        },
        "template_components": template_components,
        "source_components": source_components,
    }
    return enriched, report


def _candidate_groups(
    task: Mapping[str, Any],
    raw_records: Sequence[Mapping[str, Any]],
    basic_tokenizer,
    bashlint_template,
) -> tuple[dict[str, set[int]], tuple[str, ...] | None, str]:
    query = task["query"]
    gold = task["gold"]
    normalized_query = normalize_instruction(query, basic_tokenizer)
    task_signature = _template_signature(
        command_template_keys(gold, bashlint_template=bashlint_template)
    )
    task_sequence = bashlex_utility_operator_sequence(gold)
    channels: dict[str, set[int]] = {
        "exact_pair": set(),
        "exact_instruction": set(),
        "exact_command": set(),
        "template": set(),
        "sequence_exact": set(),
        "sequence_edit_one": set(),
    }
    for index, record in enumerate(raw_records):
        instruction_similarity = token_jaccard(
            normalized_query,
            record["normalized_instruction"],
        )
        if record["instruction_raw"] == query and record["command_raw"] == gold:
            channels["exact_pair"].add(index)
        if record["instruction_raw"] == query:
            channels["exact_instruction"].add(index)
        if record["command_raw"] == gold:
            channels["exact_command"].add(index)
        if (
            task_signature
            and task_signature == tuple(record["template_signature"])
            and not record["template_signature_suppressed"]
            and instruction_similarity >= MIN_STRUCTURE_INSTRUCTION_JACCARD
        ):
            channels["template"].add(index)
        record_sequence_value = record.get("utility_operator_sequence")
        if task_sequence is not None and record_sequence_value is not None:
            record_sequence = tuple(record_sequence_value)
            if _is_sequence_exact_candidate(
                task_sequence,
                record_sequence,
                instruction_similarity,
            ):
                channels["sequence_exact"].add(index)
            elif _is_sequence_edit_one_candidate(
                task_sequence,
                record_sequence,
                instruction_similarity,
            ):
                channels["sequence_edit_one"].add(index)
    return channels, task_sequence, normalized_query


def _utility_count(sequence: Sequence[str]) -> int:
    return sum(str(value).startswith("UTILITY:") for value in sequence)


def _is_sequence_exact_candidate(
    task_sequence: Sequence[str],
    record_sequence: Sequence[str],
    instruction_similarity: float,
) -> bool:
    return (
        tuple(task_sequence) == tuple(record_sequence)
        and _utility_count(task_sequence) >= MIN_STRUCTURE_UTILITY_COUNT
        and _utility_count(record_sequence) >= MIN_STRUCTURE_UTILITY_COUNT
        and instruction_similarity >= MIN_STRUCTURE_INSTRUCTION_JACCARD
    )


def _is_sequence_edit_one_candidate(
    task_sequence: Sequence[str],
    record_sequence: Sequence[str],
    instruction_similarity: float,
) -> bool:
    return (
        _utility_count(task_sequence) >= MIN_STRUCTURE_UTILITY_COUNT
        and _utility_count(record_sequence) >= MIN_STRUCTURE_UTILITY_COUNT
        and instruction_similarity >= MIN_STRUCTURE_INSTRUCTION_JACCARD
        and levenshtein_distance(
            task_sequence,
            record_sequence,
            maximum=1,
        )
        == 1
    )


def _load_adjudications(
    path: str | Path | None,
) -> dict[str, dict[str, set[str]]]:
    if path is None:
        return {}
    values: dict[str, dict[str, set[str]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = str(record["task_id"])
            annotator = str(record["annotator"])
            groups = {str(value) for value in record["accepted_source_group_ids"]}
            if annotator in values.setdefault(task_id, {}):
                raise ValueError(
                    f"Duplicate adjudication for {task_id}/{annotator} at line {line_number}"
                )
            values[task_id][annotator] = groups
    return values


def retrieve_intercode_provenance(
    tasks: Sequence[Mapping[str, Any]],
    raw_records: Sequence[Mapping[str, Any]],
    nl2bash_root: str | Path,
    *,
    adjudications_path: str | Path | None = None,
    primary: bool = False,
    required_task_ids: frozenset[str] | None = None,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    required_task_ids = (
        OFFICIAL_INTERCODE_TASK_IDS
        if required_task_ids is None
        else required_task_ids
    )
    actual_task_ids = [str(task.get("task_id")) for task in tasks]
    complete_task_set = (
        len(actual_task_ids) == len(required_task_ids)
        and set(actual_task_ids) == set(required_task_ids)
    )
    if primary and not complete_task_set:
        missing = sorted(set(required_task_ids) - set(actual_task_ids))
        extra = sorted(set(actual_task_ids) - set(required_task_ids))
        raise RuntimeError(
            "Primary provenance requires the exact official 200-task set; "
            f"got={len(actual_task_ids)}, missing={missing[:3]}, extra={extra[:3]}"
        )
    basic_tokenizer = load_official_basic_tokenizer(nl2bash_root)
    bashlint_template = load_bashlint_template_function(nl2bash_root)
    adjudications = _load_adjudications(adjudications_path)
    output: list[dict[str, Any]] = []
    candidate_union_groups: set[str] = set()
    actual_deny_groups: set[str] = set()
    channel_union_rows: dict[str, set[str]] = collections.defaultdict(set)
    channel_union_groups: dict[str, set[str]] = collections.defaultdict(set)
    unresolved: list[str] = []

    for task in tasks:
        channels, task_sequence, normalized_query = _candidate_groups(
            task,
            raw_records,
            basic_tokenizer,
            bashlint_template,
        )
        all_candidate_indices = set().union(*channels.values())
        candidate_records = []
        for index in all_candidate_indices:
            record = raw_records[index]
            reasons = sorted(
                channel for channel, members in channels.items() if index in members
            )
            candidate_records.append(
                {
                    "sample_id": record["sample_id"],
                    "source_group_id": record["source_group_id"],
                    "source_line": record["source_line"],
                    "official_split": record["official_split"],
                    "reasons": reasons,
                    "instruction_jaccard": token_jaccard(
                        normalized_query,
                        record["normalized_instruction"],
                    ),
                    "instruction_raw": record["instruction_raw"],
                    "command_raw": record["command_raw"],
                }
            )
            candidate_union_groups.add(record["source_group_id"])
            for reason in reasons:
                channel_union_rows[reason].add(record["sample_id"])
                channel_union_groups[reason].add(record["source_group_id"])
        candidate_records.sort(
            key=lambda value: (
                -value["instruction_jaccard"],
                value["sample_id"],
            )
        )
        if len(candidate_records) > MAX_CANDIDATE_ROWS_PER_TASK:
            raise RuntimeError(
                f"Provenance retrieval for {task['task_id']} produced "
                f"{len(candidate_records)} rows, exceeding the explicit "
                f"{MAX_CANDIDATE_ROWS_PER_TASK}-row audit cap"
            )

        exact_pair_groups = {
            raw_records[index]["source_group_id"] for index in channels["exact_pair"]
        }
        template_groups = {
            raw_records[index]["source_group_id"] for index in channels["template"]
        }
        resolution_mode: str | None = None
        resolved_groups: set[str] = set()
        if len(exact_pair_groups) == 1:
            resolution_mode = "automatic_exact_pair"
            resolved_groups = exact_pair_groups
        elif (
            not exact_pair_groups
            and len(template_groups) == 1
            and all(
                raw_records[index]["source_group_id"] in template_groups
                for channel in ("exact_instruction", "exact_command")
                for index in channels[channel]
            )
        ):
            resolution_mode = "automatic_unique_template"
            resolved_groups = template_groups
        else:
            task_labels = adjudications.get(task["task_id"], {})
            independent = [
                groups
                for annotator, groups in sorted(task_labels.items())
                if annotator != "ADJUDICATOR"
            ]
            adjudicator = task_labels.get("ADJUDICATOR")
            if len(independent) > 2:
                raise ValueError(
                    f"{task['task_id']} has more than two independent annotators"
                )
            if adjudicator is not None and len(independent) != 2:
                raise ValueError(
                    f"{task['task_id']} may use ADJUDICATOR only after two "
                    "independent annotations"
                )
            if len(independent) == 2 and independent[0] == independent[1]:
                if adjudicator is not None:
                    raise ValueError(
                        f"{task['task_id']} has an unnecessary ADJUDICATOR "
                        "despite independent agreement"
                    )
                resolution_mode = "independent_annotator_agreement"
                resolved_groups = set(independent[0])
            elif len(independent) == 2 and adjudicator is not None:
                resolution_mode = "explicit_adjudication"
                resolved_groups = set(adjudicator)
        candidate_group_set = {
            candidate["source_group_id"] for candidate in candidate_records
        }
        if (
            resolution_mode is None
            or not resolved_groups <= candidate_group_set
        ):
            unresolved.append(task["task_id"])
            resolution_mode = None
            resolved_groups = set()
        # The formal policy is deliberately conservative: every retrieved
        # candidate source group is removed, including all candidates for
        # tasks whose exact source cannot be resolved automatically.  Human
        # annotations remain optional audit metadata and never decide which
        # rows are allowed back into training.
        actual_deny_groups.update(candidate_group_set)
        output.append(
            {
                "task_id": task["task_id"],
                "query": task["query"],
                "gold": task["gold"],
                "utility_operator_sequence": list(task_sequence)
                if task_sequence is not None
                else None,
                "candidate_count": len(candidate_records),
                "candidate_group_count": len(candidate_group_set),
                "candidates": candidate_records,
                "resolution_mode": resolution_mode,
                "resolved_source_group_ids": sorted(resolved_groups),
                "denied_source_group_ids": sorted(candidate_group_set),
            }
        )

    unresolved = sorted(set(unresolved))
    all_candidates_denied = actual_deny_groups == candidate_union_groups
    report = {
        "schema": "intercode_provenance_v3",
        "tasks": len(tasks),
        "resolved": len(tasks) - len(unresolved),
        "unresolved": len(unresolved),
        "unresolved_task_ids": unresolved,
        "unresolved_candidates_removed": all_candidates_denied,
        "adjudication_required_for_formal": False,
        "minimum_structure_instruction_jaccard": (
            MIN_STRUCTURE_INSTRUCTION_JACCARD
        ),
        "minimum_structure_utility_count": MIN_STRUCTURE_UTILITY_COUNT,
        "maximum_candidate_rows_per_task": MAX_CANDIDATE_ROWS_PER_TASK,
        "candidate_union_group_count": len(candidate_union_groups),
        "actual_deny_group_count": len(actual_deny_groups),
        "deny_policy": "all_candidates",
        "channel_union_counts": {
            channel: {
                "rows": len(channel_union_rows[channel]),
                "source_groups": len(channel_union_groups[channel]),
            }
            for channel in sorted(channel_union_rows)
        },
        "primary_ready": bool(
            primary
            and complete_task_set
            and all_candidates_denied
        ),
    }
    return output, actual_deny_groups, report


def _derived_bucket(source_group_id: str) -> int:
    digest = hashlib.sha256(source_group_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10


def build_derived_split(
    raw_records: Sequence[Mapping[str, Any]],
    intercode_deny_groups: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    official_test_groups = {
        record["source_group_id"]
        for record in raw_records
        if record["official_split"] == "TEST"
    }
    allowed_official = {"TRAIN", "DEV"}
    output: list[dict[str, Any]] = []
    exclude_counts: collections.Counter[str] = collections.Counter()
    for record in raw_records:
        reasons: list[str] = []
        if record["official_split"] not in allowed_official:
            reasons.append(f"official_{record['official_split'].lower()}")
        if record["source_group_id"] in official_test_groups:
            reasons.append("official_test_source_component")
        if record["source_group_id"] in intercode_deny_groups:
            reasons.append("intercode_denied_source_component")
        if reasons:
            exclude_counts.update(reasons)
            derived_split = None
        else:
            derived_split = (
                "DEV" if _derived_bucket(record["source_group_id"]) == 0 else "TRAIN"
            )
        output.append(
            {
                **record,
                "derived_split": derived_split,
                "exclusion_reasons": reasons,
            }
        )

    active = [record for record in output if record["derived_split"] is not None]
    train_groups = {
        record["source_group_id"] for record in active if record["derived_split"] == "TRAIN"
    }
    dev_groups = {
        record["source_group_id"] for record in active if record["derived_split"] == "DEV"
    }
    if train_groups & dev_groups:
        raise AssertionError("A source component was split across derived train/dev")
    if (train_groups | dev_groups) & official_test_groups:
        raise AssertionError("Official test source component leaked into derived data")
    if (train_groups | dev_groups) & intercode_deny_groups:
        raise AssertionError("InterCode denied source component leaked into derived data")

    train_exact_pairs = {
        _exact_pair_key(record["normalized_instruction"], record["command_raw"])
        for record in active
        if record["derived_split"] == "TRAIN"
    }
    dev_exact_pairs = {
        _exact_pair_key(record["normalized_instruction"], record["command_raw"])
        for record in active
        if record["derived_split"] == "DEV"
    }
    if train_exact_pairs & dev_exact_pairs:
        raise AssertionError("Derived train/dev exact-pair overlap remains")
    train_templates = {
        _template_signature_key(record["template_signature"])
        for record in active
        if record["derived_split"] == "TRAIN"
        and not record["template_signature_suppressed"]
    }
    dev_templates = {
        _template_signature_key(record["template_signature"])
        for record in active
        if record["derived_split"] == "DEV"
        and not record["template_signature_suppressed"]
    }
    if train_templates & dev_templates:
        raise AssertionError("Derived train/dev non-hub template overlap remains")

    allowed_rows = [
        record
        for record in raw_records
        if record["official_split"] in allowed_official
    ]
    intercode_denied_allowed_rows = sum(
        record["source_group_id"] in intercode_deny_groups
        for record in allowed_rows
    )
    deny_fraction = (
        intercode_denied_allowed_rows / len(allowed_rows) if allowed_rows else 0.0
    )
    active_fraction = len(active) / len(allowed_rows) if allowed_rows else 0.0
    if deny_fraction > MAX_INTERCODE_DENY_ROW_FRACTION:
        raise RuntimeError(
            "InterCode provenance retrieval would remove too much released "
            f"train/dev data: {deny_fraction:.4f} > "
            f"{MAX_INTERCODE_DENY_ROW_FRACTION:.2f}"
        )
    if active_fraction < MIN_DERIVED_ACTIVE_ROW_FRACTION:
        raise RuntimeError(
            "Too little released train/dev data remains after all leakage gates: "
            f"{active_fraction:.4f} < {MIN_DERIVED_ACTIVE_ROW_FRACTION:.2f}"
        )

    report = {
        "schema": "derived_split_v2",
        "counts": dict(collections.Counter(record["derived_split"] for record in active)),
        "source_group_counts": {
            "TRAIN": len(train_groups),
            "DEV": len(dev_groups),
        },
        "excluded": dict(exclude_counts),
        "official_test_deny_groups": len(official_test_groups),
        "intercode_deny_groups": len(intercode_deny_groups),
        "released_train_dev_rows": len(allowed_rows),
        "intercode_denied_released_train_dev_rows": intercode_denied_allowed_rows,
        "intercode_deny_row_fraction": deny_fraction,
        "maximum_intercode_deny_row_fraction": MAX_INTERCODE_DENY_ROW_FRACTION,
        "active_row_fraction": active_fraction,
        "minimum_active_row_fraction": MIN_DERIVED_ACTIVE_ROW_FRACTION,
    }
    return output, report


def materialize_provenance(
    raw_records: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    nl2bash_root: str | Path,
    output_dir: str | Path,
    *,
    adjudications_path: str | Path | None = None,
    primary: bool = False,
    source_input_sha256: Mapping[str, str],
) -> list[dict[str, Any]]:
    output_dir = ensure_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_ids = {str(task["task_id"]) for task in tasks}
    adjudications_snapshot = output_dir / "adjudications.used.jsonl"
    adjudication_rows: list[dict[str, Any]] = []
    if adjudications_path is not None:
        with Path(adjudications_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid adjudication JSON at line {line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise TypeError(
                        f"Adjudication line {line_number} must be an object"
                    )
                if set(value) != {
                    "task_id",
                    "annotator",
                    "accepted_source_group_ids",
                }:
                    raise ValueError(
                        f"Unexpected adjudication schema at line {line_number}"
                    )
                groups = value["accepted_source_group_ids"]
                if not isinstance(groups, list):
                    raise TypeError(
                        f"accepted_source_group_ids must be a list at line {line_number}"
                    )
                if str(value["task_id"]) not in task_ids:
                    raise ValueError(
                        f"Adjudication line {line_number} names an unknown task"
                    )
                adjudication_rows.append(
                    {
                        "task_id": str(value["task_id"]),
                        "annotator": str(value["annotator"]),
                        "accepted_source_group_ids": sorted(
                            {str(group_id) for group_id in groups}
                        ),
                    }
                )
    write_jsonl(adjudications_snapshot, adjudication_rows)
    grouped, group_report = build_source_groups(raw_records, nl2bash_root)
    provenance, deny_groups, provenance_report = retrieve_intercode_provenance(
        tasks,
        grouped,
        nl2bash_root,
        adjudications_path=adjudications_snapshot,
        primary=primary,
    )
    # Persist the retrieval audit even when a downstream deletion-rate gate
    # refuses to create a derived split.
    provenance_path = output_dir / "provenance.jsonl"
    write_jsonl(provenance_path, provenance)
    groups_path = output_dir / "groups.json"
    write_json(groups_path, group_report)
    derived, split_report = build_derived_split(grouped, deny_groups)
    derived_path = output_dir / "raw_pairs.grouped_and_split.jsonl"
    write_jsonl(derived_path, derived)
    derived_report_path = output_dir / "derived_split_report.json"
    write_json(derived_report_path, split_report)
    # Bind the provenance decision to the exact candidate audit and derived
    # split.  ``primary_run`` remains explicit so an exploratory artifact
    # cannot be promoted later even though it uses the same conservative
    # all-candidates deletion rule.
    provenance_report = {
        **provenance_report,
        "primary_run": bool(primary),
        "input_sha256": dict(source_input_sha256),
        "adjudications_sha256": sha256_file(adjudications_snapshot),
        "groups_sha256": sha256_file(groups_path),
        "provenance_sha256": sha256_file(provenance_path),
        "derived_split_sha256": sha256_file(derived_path),
        "derived_split_report_sha256": sha256_file(derived_report_path),
    }
    write_json(output_dir / "provenance_report.json", provenance_report)
    return derived
