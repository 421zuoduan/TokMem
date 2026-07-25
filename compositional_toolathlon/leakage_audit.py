from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_experiment_config
from .verify_tasks import read_jsonl


def _task_id_hash(task_ids: list[str]) -> str:
    material = json.dumps(sorted(task_ids), separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def semantic_leakage_audit(
    *,
    tasks: list[dict[str, Any]],
    protected_task_root: str | Path,
    embedding_model: str | Path,
    cosine_threshold: float,
) -> dict[str, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "semantic leakage audit requires sentence-transformers in tokmem"
        ) from exc
    if not 0.0 <= cosine_threshold <= 1.0:
        raise ValueError("cosine threshold must be in [0, 1]")
    task_ids = [task.get("task_id") for task in tasks]
    if any(not isinstance(task_id, str) for task_id in task_ids):
        raise ValueError("every synthetic task requires a task_id")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("synthetic leakage input contains duplicate task IDs")

    config = load_experiment_config()
    protected_texts = []
    protected_root = Path(protected_task_root)
    for task_id in config.task_ids:
        for filename in ("docs/task.md", "docs/task_cn.md"):
            path = protected_root / task_id / filename
            if path.is_file():
                protected_texts.append(path.read_text(encoding="utf-8"))
    if not protected_texts:
        raise ValueError("semantic leakage audit found no protected task prompts")

    model_path = Path(embedding_model).resolve()
    if not model_path.exists():
        raise ValueError(f"embedding model must already exist locally: {model_path}")
    model = SentenceTransformer(str(model_path), local_files_only=True)
    synthetic_embeddings = model.encode(
        [task["instruction"] for task in tasks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    protected_embeddings = model.encode(
        protected_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    similarity = np.asarray(synthetic_embeddings) @ np.asarray(
        protected_embeddings
    ).T
    per_task = []
    for index, task_id in enumerate(task_ids):
        maximum = float(similarity[index].max())
        per_task.append(
            {
                "task_id": task_id,
                "max_cosine_similarity": maximum,
                "passed": maximum < cosine_threshold,
            }
        )
    return {
        "schema_version": 1,
        "passed": all(record["passed"] for record in per_task),
        "task_id_hash": _task_id_hash(task_ids),
        "metric": "normalized_embedding_cosine",
        "embedding_model_path": str(model_path),
        "embedding_model_name": model_path.name,
        "cosine_threshold": cosine_threshold,
        "protected_file_count": len(protected_texts),
        "protected_text_not_recorded": True,
        "per_task": per_task,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline semantic rejection audit against protected Toolathlon prompts"
    )
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--protected-task-root", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--cosine-threshold", type=float, default=0.85)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = semantic_leakage_audit(
        tasks=read_jsonl(args.tasks),
        protected_task_root=args.protected_task_root,
        embedding_model=args.embedding_model,
        cosine_threshold=args.cosine_threshold,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"passed={report['passed']} tasks={len(report['per_task'])} "
        f"protected_files={report['protected_file_count']}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
