#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compositional.run_layout import write_json


SEQUENTIAL_LOG_RE = re.compile(r"sequential_training_(\d{8}_\d{6})(?:_eval_results)?\.log$")


def move_or_print(src: Path, dst: Path, dry_run: bool):
    print(f"{'DRY-RUN' if dry_run else 'MOVE'} {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def write_legacy_metadata(run_dir: Path, run_name: str, source_paths, notes, dry_run: bool):
    run_config = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "source": "legacy_migration",
        "legacy_sources": source_paths,
    }
    run_summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "source": "legacy_migration",
        "notes": notes,
        "legacy_sources": source_paths,
    }
    if dry_run:
        print(f"DRY-RUN write {run_dir / 'run_config.json'}")
        print(f"DRY-RUN write {run_dir / 'run_summary.json'}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(str(run_dir / "run_config.json"), run_config)
    write_json(str(run_dir / "run_summary.json"), run_summary)


def migrate_sequential_logs(compositional_dir: Path, runs_dir: Path, dry_run: bool):
    log_dir = compositional_dir / "log"
    if not log_dir.exists():
        return

    grouped = {}
    for path in sorted(log_dir.glob("sequential_training_*.log")):
        match = SEQUENTIAL_LOG_RE.match(path.name)
        if not match:
            continue
        timestamp = match.group(1)
        bucket = grouped.setdefault(timestamp, {})
        if path.name.endswith("_eval_results.log"):
            bucket["evaluation.log"] = path
        else:
            bucket["training.log"] = path

    for timestamp, artifacts in grouped.items():
        run_name = f"legacy_sequential_training_{timestamp}"
        run_dir = runs_dir / run_name
        source_paths = []
        for target_name, src in artifacts.items():
            source_paths.append(str(src))
            move_or_print(src, run_dir / target_name, dry_run)
        write_legacy_metadata(
            run_dir,
            run_name,
            source_paths,
            ["Legacy sequential log migration"],
            dry_run,
        )


def migrate_checkpoint_dirs(compositional_dir: Path, runs_dir: Path, dry_run: bool):
    for checkpoint_dir in sorted(compositional_dir.glob("checkpoints*")):
        if not checkpoint_dir.is_dir():
            continue
        contents = list(checkpoint_dir.iterdir())
        if not contents:
            continue

        run_name = f"legacy_{checkpoint_dir.name}"
        run_dir = runs_dir / run_name
        source_paths = []

        for src in sorted(contents):
            source_paths.append(str(src))
            move_or_print(src, run_dir / src.name, dry_run)

        write_legacy_metadata(
            run_dir,
            run_name,
            source_paths,
            [f"Migrated from legacy checkpoint directory {checkpoint_dir.name}"],
            dry_run,
        )


def migrate_icl_results(compositional_dir: Path, runs_dir: Path, dry_run: bool):
    for path in sorted(compositional_dir.glob("icl_results*.json")):
        run_name = f"legacy_{path.stem}"
        run_dir = runs_dir / run_name
        move_or_print(path, run_dir / "evaluation_results.json", dry_run)
        write_legacy_metadata(
            run_dir,
            run_name,
            [str(path)],
            ["Migrated legacy ICL evaluation JSON"],
            dry_run,
        )


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy compositional outputs into compositional/runs/")
    parser.add_argument("--compositional-dir", type=str, default=None, help="Path to compositional directory")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without moving files")
    args = parser.parse_args()

    if args.compositional_dir is not None:
        compositional_dir = Path(args.compositional_dir).resolve()
    else:
        compositional_dir = Path(__file__).resolve().parents[1]

    runs_dir = compositional_dir / "runs"
    if not args.dry_run:
        runs_dir.mkdir(parents=True, exist_ok=True)

    migrate_sequential_logs(compositional_dir, runs_dir, args.dry_run)
    migrate_checkpoint_dirs(compositional_dir, runs_dir, args.dry_run)
    migrate_icl_results(compositional_dir, runs_dir, args.dry_run)


if __name__ == "__main__":
    main()
