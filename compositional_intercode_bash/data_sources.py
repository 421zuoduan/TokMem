"""Load the released NL2Bash split and the 200 InterCode-Bash tasks.

The current NL2Bash source tree can no longer reproduce the published split:
its legacy parser/filter now returns different rows under modern Python
dependencies.  The default path therefore consumes the released 9,305-row
split at a pinned revision and verifies every Parquet file by SHA-256.  The
legacy replay remains available as an explicit audit mode; it is never allowed
to silently replace the released split.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io_utils import (
    PACKAGE_ROOT,
    ensure_output_path,
    read_json,
    read_jsonl,
    sha256_file,
    stable_id,
    write_json,
    write_jsonl,
)
from .templates import _stage_legacy_modules


EXPECTED_NL2BASH = {
    "raw": 12607,
    "filtered": 9305,
    "TRAIN": 8090,
    "DEV": 609,
    "TEST": 606,
}
EXPECTED_INTERCODE_COUNTS = {"fs1": 60, "fs2": 53, "fs3": 60, "fs4": 27}
CANONICAL_NL2BASH_REPOSITORY = "jiacheng-ye/nl2bash"
CANONICAL_NL2BASH_REVISION = "f92a7f891996d1d1230f25294a2a1c1c8b0cb1a4"
CANONICAL_SPLIT_FILES = {
    "TRAIN": {
        "filename": "train.parquet",
        "remote_path": "default/train/0000.parquet",
        "rows": EXPECTED_NL2BASH["TRAIN"],
        "sha256": "f5b077fb193fa517eaa1b2826e2bfef5b7f94e7c42aa6c83cdb8085481929054",
    },
    "DEV": {
        "filename": "dev.parquet",
        "remote_path": "default/validation/0000.parquet",
        "rows": EXPECTED_NL2BASH["DEV"],
        "sha256": "f596028521252e55289970ca1fcc14702f7e9faf2fd3e143d6e2430c5c4c3185",
    },
    "TEST": {
        "filename": "test.parquet",
        "remote_path": "default/test/0000.parquet",
        "rows": EXPECTED_NL2BASH["TEST"],
        "sha256": "bb88287af45b1eb154db5ee25c927d40d16c0110466e0b282e6ee68bd8501558",
    },
}
DEFAULT_CANONICAL_SPLIT_DIR = PACKAGE_ROOT / "artifacts" / "canonical_nl2bash"
SOURCE_ARTIFACT_FILENAMES = {
    "raw_pairs_official": "raw_pairs.official.jsonl",
    "intercode_tasks": "intercode_tasks.jsonl",
    "source_manifest": "source_manifest.json",
}


def _read_source_lines(path: str | Path) -> list[str]:
    """Remove one storage line ending, preserving every other character."""

    values: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.endswith("\r\n"):
                values.append(line[:-2])
            elif line.endswith("\n"):
                values.append(line[:-1])
            else:
                values.append(line)
    return values


def load_raw_nl2bash(data_dir: str | Path) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    instructions = _read_source_lines(data_dir / "all.nl")
    commands = _read_source_lines(data_dir / "all.cm")
    if len(instructions) != len(commands):
        raise ValueError("NL2Bash all.nl and all.cm have unequal line counts")
    if len(instructions) != EXPECTED_NL2BASH["raw"]:
        raise ValueError(
            f"Expected {EXPECTED_NL2BASH['raw']} NL2Bash pairs, got {len(instructions)}"
        )
    records = []
    for index, (instruction, command) in enumerate(
        zip(instructions, commands),
        start=1,
    ):
        records.append(
            {
                "sample_id": stable_id(
                    "NL2BASH_RAW_V1",
                    "data/bash",
                    index,
                    instruction,
                    command,
                ),
                "source_line": index,
                "instruction_raw": instruction,
                "command_raw": command,
            }
        )
    return records


def load_canonical_nl2bash_splits(
    split_dir: str | Path,
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, Any]]:
    """Read and verify the pinned, released 8,090/609/606 split."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "Reading the released NL2Bash split requires pyarrow. "
            "Install the requirements in the tokmem environment."
        ) from exc

    split_dir = Path(split_dir)
    split_pairs: dict[str, list[tuple[str, str]]] = {}
    files: dict[str, Any] = {}
    for split, specification in CANONICAL_SPLIT_FILES.items():
        path = split_dir / str(specification["filename"])
        if not path.exists():
            raise FileNotFoundError(
                f"Missing released NL2Bash file {path}. Run "
                "`python -m compositional_intercode_bash.download_data` first, "
                "or pass --canonical-split-dir."
            )
        actual_hash = sha256_file(path)
        if actual_hash != specification["sha256"]:
            raise ValueError(
                f"Released NL2Bash hash mismatch for {path}: "
                f"expected={specification['sha256']}, actual={actual_hash}"
            )
        table = parquet.read_table(path)
        if table.schema.names != ["nl", "bash"]:
            raise ValueError(
                f"Unexpected released NL2Bash schema in {path}: {table.schema.names}"
            )
        values = table.to_pydict()
        instructions = values["nl"]
        commands = values["bash"]
        if len(instructions) != int(specification["rows"]):
            raise ValueError(
                f"{path} should contain {specification['rows']} rows, "
                f"got {len(instructions)}"
            )
        pairs: list[tuple[str, str]] = []
        for row_index, (instruction, command) in enumerate(
            zip(instructions, commands)
        ):
            if not isinstance(instruction, str) or not isinstance(command, str):
                raise TypeError(
                    f"NL2Bash nl/bash must be strings at {split}:{row_index}"
                )
            pairs.append((instruction, command))
        split_pairs[split] = pairs
        files[split] = {
            "path": str(path.resolve()),
            "sha256": actual_hash,
            "rows": len(pairs),
        }

    total = sum(len(values) for values in split_pairs.values())
    if total != EXPECTED_NL2BASH["filtered"]:
        raise AssertionError(
            f"Released NL2Bash total should be {EXPECTED_NL2BASH['filtered']}, "
            f"got {total}"
        )
    return split_pairs, {
        "kind": "pinned_released_parquet",
        "repository": CANONICAL_NL2BASH_REPOSITORY,
        "revision": CANONICAL_NL2BASH_REVISION,
        "rows": total,
        "files": files,
    }


def load_intercode_tasks(intercode_root: str | Path) -> list[dict[str, Any]]:
    intercode_root = Path(intercode_root)
    tasks: list[dict[str, Any]] = []
    for fs_id, expected_count in EXPECTED_INTERCODE_COUNTS.items():
        fs_number = fs_id.removeprefix("fs")
        source = (
            intercode_root
            / "data"
            / "bash"
            / "nl2bash"
            / f"nl2bash_fs_{fs_number}.json"
        )
        with source.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if len(records) != expected_count:
            raise ValueError(
                f"{source} should contain {expected_count} tasks, got {len(records)}"
            )
        for local_index, record in enumerate(records):
            if set(record) != {"query", "gold"}:
                raise ValueError(
                    f"Unexpected InterCode schema at {fs_id}:{local_index}: "
                    f"{sorted(record)}"
                )
            if not isinstance(record["query"], str) or not isinstance(record["gold"], str):
                raise TypeError(f"InterCode query/gold must be strings at {fs_id}:{local_index}")
            tasks.append(
                {
                    "task_id": f"{fs_id}:{local_index:03d}",
                    "fs_id": fs_id,
                    "local_index": local_index,
                    "query": record["query"],
                    "gold": record["gold"],
                    "source_path": str(source.resolve()),
                    "source_sha256": sha256_file(source),
                }
            )
    if len(tasks) != 200:
        raise AssertionError(f"Expected 200 InterCode tasks, got {len(tasks)}")
    if len({task["task_id"] for task in tasks}) != 200:
        raise AssertionError("InterCode task IDs are not unique")
    return tasks


def _intercode_source_files(intercode_root: str | Path) -> dict[str, dict[str, Any]]:
    intercode_root = Path(intercode_root)
    output: dict[str, dict[str, Any]] = {}
    for fs_id, expected_count in EXPECTED_INTERCODE_COUNTS.items():
        fs_number = fs_id.removeprefix("fs")
        path = (
            intercode_root
            / "data"
            / "bash"
            / "nl2bash"
            / f"nl2bash_fs_{fs_number}.json"
        )
        output[fs_id] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": expected_count,
        }
    return output


def validate_materialized_sources(
    artifact_dir: str | Path,
    *,
    require_canonical: bool,
) -> dict[str, Any]:
    """Validate the complete, materialized source dataset and its manifest."""

    artifact_dir = Path(artifact_dir)
    raw_path = artifact_dir / SOURCE_ARTIFACT_FILENAMES["raw_pairs_official"]
    tasks_path = artifact_dir / SOURCE_ARTIFACT_FILENAMES["intercode_tasks"]
    manifest_path = artifact_dir / SOURCE_ARTIFACT_FILENAMES["source_manifest"]
    for path in (raw_path, tasks_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required source artifact is missing: {path}")

    raw_records = list(read_jsonl(raw_path))
    if len(raw_records) != EXPECTED_NL2BASH["raw"]:
        raise ValueError(
            "raw_pairs.official.jsonl must contain exactly "
            f"{EXPECTED_NL2BASH['raw']} rows; got {len(raw_records)}"
        )
    expected_raw_fields = {
        "sample_id",
        "source_line",
        "instruction_raw",
        "command_raw",
        "official_split",
    }
    sample_ids: set[str] = set()
    split_counts: collections.Counter[str] = collections.Counter()
    for index, record in enumerate(raw_records):
        if set(record) != expected_raw_fields:
            raise ValueError(
                f"Unexpected materialized NL2Bash schema at row {index}: "
                f"{sorted(record)}"
            )
        sample_id = record["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise TypeError(f"NL2Bash row {index} has an invalid sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate NL2Bash sample_id: {sample_id}")
        sample_ids.add(sample_id)
        if record["source_line"] != index + 1:
            raise ValueError(
                f"NL2Bash source_line must be consecutive at row {index}"
            )
        if not isinstance(record["instruction_raw"], str) or not isinstance(
            record["command_raw"],
            str,
        ):
            raise TypeError(f"NL2Bash row {index} contains non-string text")
        split_counts[str(record["official_split"])] += 1
    expected_split_counts = {
        "FILTERED_OUT": EXPECTED_NL2BASH["raw"] - EXPECTED_NL2BASH["filtered"],
        "TRAIN": EXPECTED_NL2BASH["TRAIN"],
        "DEV": EXPECTED_NL2BASH["DEV"],
        "TEST": EXPECTED_NL2BASH["TEST"],
    }
    if dict(split_counts) != expected_split_counts:
        raise ValueError(
            "Materialized NL2Bash split counts differ: "
            f"expected={expected_split_counts}, actual={dict(split_counts)}"
        )

    tasks = list(read_jsonl(tasks_path))
    if len(tasks) != sum(EXPECTED_INTERCODE_COUNTS.values()):
        raise ValueError(
            "intercode_tasks.jsonl must contain exactly 200 tasks; "
            f"got {len(tasks)}"
        )
    expected_task_fields = {
        "task_id",
        "fs_id",
        "local_index",
        "query",
        "gold",
        "source_path",
        "source_sha256",
    }
    expected_task_ids = {
        f"{fs_id}:{index:03d}"
        for fs_id, count in EXPECTED_INTERCODE_COUNTS.items()
        for index in range(count)
    }
    actual_task_ids: set[str] = set()
    task_counts: collections.Counter[str] = collections.Counter()
    task_source_hashes: dict[str, set[str]] = collections.defaultdict(set)
    for index, task in enumerate(tasks):
        if set(task) != expected_task_fields:
            raise ValueError(
                f"Unexpected materialized InterCode schema at row {index}: "
                f"{sorted(task)}"
            )
        task_id = task["task_id"]
        fs_id = task["fs_id"]
        local_index = task["local_index"]
        if (
            not isinstance(task_id, str)
            or fs_id not in EXPECTED_INTERCODE_COUNTS
            or not isinstance(local_index, int)
            or task_id != f"{fs_id}:{local_index:03d}"
        ):
            raise ValueError(f"Invalid InterCode task identity at row {index}")
        if not isinstance(task["query"], str) or not isinstance(task["gold"], str):
            raise TypeError(f"InterCode task {task_id} has non-string query/gold")
        if not isinstance(task["source_sha256"], str):
            raise TypeError(f"InterCode task {task_id} lacks source_sha256")
        actual_task_ids.add(task_id)
        task_counts[fs_id] += 1
        task_source_hashes[fs_id].add(task["source_sha256"])
    if actual_task_ids != expected_task_ids:
        missing = sorted(expected_task_ids - actual_task_ids)
        extra = sorted(actual_task_ids - expected_task_ids)
        raise ValueError(
            "InterCode task IDs are not the exact official 200-task set: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    if dict(task_counts) != EXPECTED_INTERCODE_COUNTS:
        raise ValueError(
            "InterCode filesystem counts differ: "
            f"expected={EXPECTED_INTERCODE_COUNTS}, actual={dict(task_counts)}"
        )

    manifest = read_json(manifest_path)
    if manifest.get("schema") != "intercode_bash_sources_v2":
        raise ValueError("source_manifest.json has an unsupported schema")
    actual_artifacts = {
        "raw_pairs_official": {
            "sha256": sha256_file(raw_path),
            "rows": len(raw_records),
        },
        "intercode_tasks": {
            "sha256": sha256_file(tasks_path),
            "rows": len(tasks),
        },
    }
    if manifest.get("materialized_artifacts") != actual_artifacts:
        raise ValueError(
            "source_manifest.json does not bind the current materialized JSONL files"
        )
    if manifest.get("counts") != {
        "nl2bash_raw": EXPECTED_NL2BASH["raw"],
        "nl2bash_splits": expected_split_counts,
        "intercode": len(tasks),
        "intercode_filesystems": EXPECTED_INTERCODE_COUNTS,
    }:
        raise ValueError("source_manifest.json contains incorrect source counts")

    intercode_files = manifest.get("intercode_files")
    if not isinstance(intercode_files, dict) or set(intercode_files) != set(
        EXPECTED_INTERCODE_COUNTS
    ):
        raise ValueError("source_manifest.json lacks the four InterCode source files")
    for fs_id, expected_count in EXPECTED_INTERCODE_COUNTS.items():
        entry = intercode_files[fs_id]
        if (
            not isinstance(entry, dict)
            or entry.get("rows") != expected_count
            or not isinstance(entry.get("sha256"), str)
            or task_source_hashes[fs_id] != {entry["sha256"]}
        ):
            raise ValueError(
                f"InterCode source metadata differs for {fs_id}"
            )

    split_source = manifest.get("split_source")
    canonical_ready = split_source == "canonical"
    if canonical_ready:
        split_manifest = manifest.get("split_manifest")
        if not isinstance(split_manifest, dict):
            raise ValueError("Canonical source manifest lacks split_manifest")
        expected_identity = {
            "kind": "pinned_released_parquet",
            "repository": CANONICAL_NL2BASH_REPOSITORY,
            "revision": CANONICAL_NL2BASH_REVISION,
            "rows": EXPECTED_NL2BASH["filtered"],
        }
        for field, expected in expected_identity.items():
            if split_manifest.get(field) != expected:
                raise ValueError(
                    f"Canonical split manifest field {field!r} differs"
                )
        split_files = split_manifest.get("files")
        if not isinstance(split_files, dict):
            raise ValueError("Canonical split manifest lacks files")
        for split, expected in CANONICAL_SPLIT_FILES.items():
            entry = split_files.get(split)
            if (
                not isinstance(entry, dict)
                or entry.get("rows") != expected["rows"]
                or entry.get("sha256") != expected["sha256"]
            ):
                raise ValueError(
                    f"Canonical split file metadata differs for {split}"
                )
    if require_canonical and not canonical_ready:
        raise RuntimeError(
            "Formal provenance requires the pinned canonical NL2Bash split"
        )

    return {
        "canonical_ready": canonical_ready,
        "raw_pairs_official": {
            "path": str(raw_path.resolve()),
            **actual_artifacts["raw_pairs_official"],
        },
        "intercode_tasks": {
            "path": str(tasks_path.resolve()),
            **actual_artifacts["intercode_tasks"],
        },
        "source_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "source_counts": manifest["counts"],
    }


def _count_file_lines(path: Path) -> int:
    return len(_read_source_lines(path))


def verify_official_replay(stage_dir: str | Path) -> dict[str, Any]:
    stage_dir = Path(stage_dir)
    files = {
        "filtered_nl": stage_dir / "all.nl.filtered",
        "filtered_cm": stage_dir / "all.cm.filtered",
        "train_nl": stage_dir / "train.nl.filtered",
        "train_cm": stage_dir / "train.cm.filtered",
        "dev_nl": stage_dir / "dev.nl.filtered",
        "dev_cm": stage_dir / "dev.cm.filtered",
        "test_nl": stage_dir / "test.nl.filtered",
        "test_cm": stage_dir / "test.cm.filtered",
        "random_tokens": stage_dir / "random_tokens.txt",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Official replay outputs are missing: {missing}")
    counts = {
        "filtered": _count_file_lines(files["filtered_nl"]),
        "TRAIN": _count_file_lines(files["train_nl"]),
        "DEV": _count_file_lines(files["dev_nl"]),
        "TEST": _count_file_lines(files["test_nl"]),
        "normalized_instruction_groups": _count_file_lines(files["random_tokens"]),
    }
    expected = {
        "filtered": EXPECTED_NL2BASH["filtered"],
        "TRAIN": EXPECTED_NL2BASH["TRAIN"],
        "DEV": EXPECTED_NL2BASH["DEV"],
        "TEST": EXPECTED_NL2BASH["TEST"],
        "normalized_instruction_groups": 8412,
    }
    if counts != expected:
        raise ValueError(f"Official replay counts differ: expected={expected}, actual={counts}")
    for stem in ("filtered", "train", "dev", "test"):
        nl_key = f"{stem}_nl"
        cm_key = f"{stem}_cm"
        if _count_file_lines(files[nl_key]) != _count_file_lines(files[cm_key]):
            raise ValueError(f"Official replay pair files differ for {stem}")
    return {
        "counts": counts,
        "files": {
            key: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "lines": _count_file_lines(path),
            }
            for key, path in files.items()
        },
    }


def run_official_replay(
    nl2bash_root: str | Path,
    output_dir: str | Path,
    *,
    python39: str | Path = "/home/shilong/anaconda3/envs/tokmem/bin/python",
) -> dict[str, Any]:
    """Run the original filter/split functions on a private staging copy."""

    nl2bash_root = Path(nl2bash_root).resolve()
    source_data = nl2bash_root / "data" / "bash"
    output_dir = ensure_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("all.nl", "all.cm"):
        shutil.copyfile(source_data / filename, output_dir / filename)

    entry = PACKAGE_ROOT / "official_replay_entry.py"
    replay_code_root = _stage_legacy_modules(nl2bash_root)
    command = [
        str(Path(python39)),
        str(entry),
        "--nl2bash-root",
        str(replay_code_root),
        "--stage-dir",
        str(output_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=PACKAGE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    (output_dir / "official_replay.stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (output_dir / "official_replay.stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "NL2Bash official replay failed; inspect "
            f"{output_dir / 'official_replay.stderr.log'}"
        )
    report = verify_official_replay(output_dir)
    write_json(output_dir / "official_replay_manifest.json", report)
    return report


def attach_official_splits(
    raw_records: list[dict[str, Any]],
    replay_dir: str | Path,
) -> list[dict[str, Any]]:
    """Map legacy replay pair files back to stable raw IDs."""

    replay_dir = Path(replay_dir)
    split_pairs: dict[str, list[tuple[str, str]]] = {}
    for split, prefix in (("TRAIN", "train"), ("DEV", "dev"), ("TEST", "test")):
        instructions = _read_source_lines(replay_dir / f"{prefix}.nl.filtered")
        commands = _read_source_lines(replay_dir / f"{prefix}.cm.filtered")
        if len(instructions) != len(commands):
            raise ValueError(f"Official {split} pair files have unequal lengths")
        split_pairs[split] = list(zip(instructions, commands))
    return attach_split_pairs(raw_records, split_pairs)


def attach_split_pairs(
    raw_records: Sequence[Mapping[str, Any]],
    split_pairs: Mapping[str, Sequence[tuple[str, str]]],
) -> list[dict[str, Any]]:
    """Map a released split back to stable raw line IDs using multiset queues."""

    expected_keys = {"TRAIN", "DEV", "TEST"}
    if set(split_pairs) != expected_keys:
        raise ValueError(
            f"Expected split keys {sorted(expected_keys)}, got {sorted(split_pairs)}"
        )
    queues: dict[tuple[str, str], collections.deque[int]] = {}
    for index, record in enumerate(raw_records):
        key = (record["instruction_raw"].strip(), record["command_raw"].strip())
        queues.setdefault(key, collections.deque()).append(index)

    split_by_index = ["FILTERED_OUT"] * len(raw_records)
    assigned: set[int] = set()
    for split in ("TRAIN", "DEV", "TEST"):
        for instruction, command in split_pairs[split]:
            key = (instruction.strip(), command.strip())
            queue = queues.get(key)
            if not queue:
                raise ValueError(
                    "Could not map released split output back to a raw pair: "
                    f"split={split}, pair={key!r}"
                )
            raw_index = queue.popleft()
            if raw_index in assigned:
                raise AssertionError("A raw record was assigned to two released splits")
            assigned.add(raw_index)
            split_by_index[raw_index] = split

    output = []
    for index, record in enumerate(raw_records):
        output.append({**record, "official_split": split_by_index[index]})
    counts = collections.Counter(record["official_split"] for record in output)
    expected = {
        "FILTERED_OUT": EXPECTED_NL2BASH["raw"] - EXPECTED_NL2BASH["filtered"],
        "TRAIN": EXPECTED_NL2BASH["TRAIN"],
        "DEV": EXPECTED_NL2BASH["DEV"],
        "TEST": EXPECTED_NL2BASH["TEST"],
    }
    if dict(counts) != expected:
        raise ValueError(f"Mapped split counts differ: expected={expected}, actual={dict(counts)}")
    return output


def materialize_sources(
    nl2bash_root: str | Path,
    intercode_root: str | Path,
    artifact_dir: str | Path,
    *,
    split_source: str = "canonical",
    canonical_split_dir: str | Path = DEFAULT_CANONICAL_SPLIT_DIR,
    replay_python: str | Path = "/home/shilong/anaconda3/envs/tokmem/bin/python",
    reuse_replay: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    artifact_dir = ensure_output_path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_records = load_raw_nl2bash(Path(nl2bash_root) / "data" / "bash")
    if split_source == "canonical":
        split_pairs, split_manifest = load_canonical_nl2bash_splits(
            canonical_split_dir
        )
        raw_records = attach_split_pairs(raw_records, split_pairs)
    elif split_source == "legacy-replay":
        replay_dir = artifact_dir / "official_replay"
        replay_manifest = replay_dir / "official_replay_manifest.json"
        if not reuse_replay or not replay_manifest.exists():
            split_manifest = run_official_replay(
                nl2bash_root,
                replay_dir,
                python39=replay_python,
            )
        else:
            split_manifest = verify_official_replay(replay_dir)
        raw_records = attach_official_splits(raw_records, replay_dir)
    else:
        raise ValueError(
            f"Unknown split_source={split_source!r}; expected canonical or legacy-replay"
        )

    intercode_tasks = load_intercode_tasks(intercode_root)
    raw_path = artifact_dir / "raw_pairs.official.jsonl"
    tasks_path = artifact_dir / "intercode_tasks.jsonl"
    write_jsonl(raw_path, raw_records)
    write_jsonl(tasks_path, intercode_tasks)
    write_json(
        artifact_dir / "source_manifest.json",
        {
            "schema": "intercode_bash_sources_v2",
            "split_source": split_source,
            "split_manifest": split_manifest,
            "nl2bash_root": str(Path(nl2bash_root).resolve()),
            "intercode_root": str(Path(intercode_root).resolve()),
            "nl2bash_all_nl_sha256": sha256_file(
                Path(nl2bash_root) / "data" / "bash" / "all.nl"
            ),
            "nl2bash_all_cm_sha256": sha256_file(
                Path(nl2bash_root) / "data" / "bash" / "all.cm"
            ),
            "intercode_files": _intercode_source_files(intercode_root),
            "materialized_artifacts": {
                "raw_pairs_official": {
                    "sha256": sha256_file(raw_path),
                    "rows": len(raw_records),
                },
                "intercode_tasks": {
                    "sha256": sha256_file(tasks_path),
                    "rows": len(intercode_tasks),
                },
            },
            "counts": {
                "nl2bash_raw": len(raw_records),
                "nl2bash_splits": dict(
                    collections.Counter(
                        record["official_split"] for record in raw_records
                    )
                ),
                "intercode": len(intercode_tasks),
                "intercode_filesystems": dict(EXPECTED_INTERCODE_COUNTS),
            },
        },
    )
    validate_materialized_sources(
        artifact_dir,
        require_canonical=split_source == "canonical",
    )
    return raw_records, intercode_tasks


def _entry_main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nl2bash-root", required=True)
    parser.add_argument("--intercode-root", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument(
        "--split-source",
        choices=("canonical", "legacy-replay"),
        default="canonical",
    )
    parser.add_argument(
        "--canonical-split-dir",
        default=str(DEFAULT_CANONICAL_SPLIT_DIR),
    )
    parser.add_argument(
        "--replay-python",
        default="/home/shilong/anaconda3/envs/tokmem/bin/python",
        help="Python executable used by the isolated legacy replay entrypoint",
    )
    args = parser.parse_args(argv)
    materialize_sources(
        args.nl2bash_root,
        args.intercode_root,
        args.artifact_dir,
        split_source=args.split_source,
        canonical_split_dir=args.canonical_split_dir,
        replay_python=args.replay_python,
    )


if __name__ == "__main__":
    _entry_main()
