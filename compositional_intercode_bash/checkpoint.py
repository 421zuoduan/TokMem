"""Trainable-only checkpoints for expandable procedural memory."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from .io_utils import (
    canonical_json_bytes,
    ensure_output_path,
    read_json,
    sha256_bytes,
    sha256_directory,
    write_json,
)
from .memory_model import (
    MemoryTokenRegistry,
    ProceduralMemoryModel,
    resize_token_embeddings,
)

HF_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
MEMORY_CHECKPOINT_SCHEMA = "intercode_bash_memory_checkpoint_v4"


def build_base_model_identity(
    model_name: str,
    revision: str | None,
    *,
    require_reproducible: bool,
) -> dict[str, Any]:
    """Resolve a local snapshot hash or an immutable Hugging Face commit."""

    local_path = Path(model_name).expanduser()
    if local_path.is_dir():
        identity: dict[str, Any] = {
            "schema": "intercode_bash_base_model_v1",
            "kind": "local_directory",
            "resolved_path": str(local_path.resolve()),
            "directory_sha256": sha256_directory(local_path),
            "revision": None,
            "reproducible": True,
        }
    else:
        pinned = isinstance(revision, str) and HF_COMMIT_RE.fullmatch(revision) is not None
        identity = {
            "schema": "intercode_bash_base_model_v1",
            "kind": "huggingface",
            "repository": str(model_name),
            "requested_revision": revision,
            "resolved_commit": revision if pinned else None,
            "reproducible": pinned,
        }
    if require_reproducible and identity["reproducible"] is not True:
        raise RuntimeError(
            "Formal training requires either a local model directory, which is "
            "hashed in full, or a Hugging Face --model-revision containing the "
            "exact 40-character commit SHA."
        )
    identity["identity_sha256"] = sha256_bytes(
        canonical_json_bytes(identity)
    )
    return identity


def validate_base_model_identity(identity: Mapping[str, Any]) -> None:
    """Refuse to load a base model whose recorded identity no longer matches."""

    value = dict(identity)
    recorded_identity_sha = value.pop("identity_sha256", None)
    if recorded_identity_sha != sha256_bytes(canonical_json_bytes(value)):
        raise ValueError("Base-model identity record has been altered")
    kind = value.get("kind")
    if kind == "local_directory":
        path = Path(str(value.get("resolved_path", "")))
        if not path.is_dir():
            raise FileNotFoundError(f"Recorded local base model is missing: {path}")
        actual = sha256_directory(path)
        if actual != value.get("directory_sha256"):
            raise ValueError(
                "Local base-model directory differs from the training checkpoint"
            )
        if value.get("reproducible") is not True:
            raise ValueError("A hashed local base model must be reproducible")
    elif kind == "huggingface":
        commit = value.get("resolved_commit")
        reproducible = value.get("reproducible") is True
        if reproducible and (
            not isinstance(commit, str) or HF_COMMIT_RE.fullmatch(commit) is None
        ):
            raise ValueError(
                "Reproducible Hugging Face identity lacks an exact commit SHA"
            )
    else:
        raise ValueError(f"Unknown base-model identity kind: {kind!r}")


def base_model_load_location(
    identity: Mapping[str, Any],
) -> tuple[str, str | None]:
    validate_base_model_identity(identity)
    if identity["kind"] == "local_directory":
        return str(identity["resolved_path"]), None
    revision = (
        identity.get("resolved_commit")
        if identity.get("reproducible") is True
        else identity.get("requested_revision")
    )
    return str(identity["repository"]), revision


def _trainable_tensors(model: ProceduralMemoryModel) -> dict[str, torch.Tensor]:
    tensors = {
        "procedure_embeddings": model.procedure_embeddings.detach().cpu().contiguous(),
    }
    if model.eoc_embedding is not None:
        tensors["eoc_embedding"] = model.eoc_embedding.detach().cpu().contiguous()
    if model.routing_head is not None:
        for name, value in model.routing_head.state_dict().items():
            tensors[f"routing_head.{name}"] = value.detach().cpu().contiguous()
    return tensors


def save_checkpoint(
    output_dir: str | Path,
    model: ProceduralMemoryModel,
    tokenizer,
    *,
    base_model_name: str,
    base_model_revision: str | None,
    base_model_identity: Mapping[str, Any],
    lexicon_hash: str,
    training_config: Mapping[str, Any],
    training_summary: Mapping[str, Any],
) -> None:
    validate_base_model_identity(base_model_identity)
    output_dir = ensure_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(_trainable_tensors(model), str(output_dir / "trainable.safetensors"))
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    write_json(
        output_dir / "checkpoint.json",
        {
            "schema": MEMORY_CHECKPOINT_SCHEMA,
            "base_model_name": base_model_name,
            "base_model_revision": base_model_revision,
            "base_model_identity": dict(base_model_identity),
            "method": model.method,
            "logit_bias_scale": model.logit_bias_scale,
            "memory_bank_probability_threshold": (
                model.memory_bank_probability_threshold
            ),
            "lexicon_hash": lexicon_hash,
            "view_metadata": dict(training_summary.get("view_metadata", {})),
            "registry": model.registry.to_dict(),
            "hidden_size": model.hidden_size,
            "orthogonality_at_save": model.orthogonality_report(),
            "trainable_parameter_count": model.trainable_parameter_count(),
            "artifact_integrity": dict(
                training_summary.get("artifact_integrity", {})
            ),
            "artifact_sha256": dict(
                training_summary.get("artifact_sha256", {})
            ),
            "primary_ready": training_summary.get("primary_ready"),
            "primary_run": training_summary.get("primary_run"),
            "deny_policy": training_summary.get("deny_policy"),
            "formal_ready": training_summary.get("formal_ready"),
            "training_modes": dict(
                training_summary.get("training_modes", {})
            ),
            "loss_normalization": training_summary.get("loss_normalization"),
            "training_config": dict(training_config),
            "training_summary": dict(training_summary),
        },
    )


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    local_files_only: bool = True,
) -> tuple[ProceduralMemoryModel, Any, dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_dir = Path(checkpoint_dir)
    metadata = read_json(checkpoint_dir / "checkpoint.json")
    if metadata.get("schema") != MEMORY_CHECKPOINT_SCHEMA:
        raise ValueError("Unknown procedural-memory checkpoint schema")
    model_name, model_revision = base_model_load_location(
        metadata["base_model_identity"]
    )
    if metadata.get("base_model_name") != model_name:
        raise ValueError("Checkpoint base_model_name differs from its identity")
    if metadata.get("base_model_revision") != model_revision:
        raise ValueError("Checkpoint base_model_revision differs from its identity")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir / "tokenizer",
        local_files_only=True,
    )
    registry = MemoryTokenRegistry.restore(tokenizer, metadata["registry"])
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    resize_token_embeddings(base_model, len(tokenizer))
    base_model.to(device)
    model = ProceduralMemoryModel(
        base_model,
        registry,
        method=metadata["method"],
        initialization_seed=int(
            metadata.get("training_config", {}).get("model_seed", 0)
        ),
        logit_bias_scale=float(metadata["logit_bias_scale"]),
        memory_bank_probability_threshold=float(
            metadata["memory_bank_probability_threshold"]
        ),
    )
    tensors = load_file(str(checkpoint_dir / "trainable.safetensors"), device=str(device))
    expected = {"procedure_embeddings"}
    if model.eoc_embedding is not None:
        expected.add("eoc_embedding")
    if model.routing_head is not None:
        expected.update(
            f"routing_head.{name}" for name in model.routing_head.state_dict()
        )
    if set(tensors) != expected:
        raise ValueError(
            f"Checkpoint tensor keys differ: expected={sorted(expected)}, "
            f"actual={sorted(tensors)}"
        )
    if tensors["procedure_embeddings"].shape != model.procedure_embeddings.shape:
        raise ValueError("Procedure embedding checkpoint shape differs")
    with torch.no_grad():
        model.procedure_embeddings.copy_(tensors["procedure_embeddings"])
        if model.eoc_embedding is not None:
            model.eoc_embedding.copy_(tensors["eoc_embedding"])
        if model.routing_head is not None:
            model.routing_head.load_state_dict(
                {
                    name: tensors[f"routing_head.{name}"]
                    for name in model.routing_head.state_dict()
                },
                strict=True,
            )
    model.to(device)
    return model, tokenizer, metadata
