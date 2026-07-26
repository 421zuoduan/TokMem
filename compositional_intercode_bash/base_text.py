"""Frozen base-model baseline with a bounded, TRAIN-only Bash tool catalog.

This module deliberately stays separate from the procedural-memory training
path.  A base-text artifact contains no learned parameters: it binds an
immutable base model identity to a deterministic text catalog, a tokenizer,
and the formal provenance/procedure evidence used to construct that catalog.

The public entry points intended for the evaluator are:

``prepare_base_text_artifact``
    Load the unmodified base tokenizer and materialize a catalog artifact.
``load_base_text_artifact``
    Validate that artifact and return a frozen greedy-decoding adapter.
``compose_base_text_system_prompt``
    Add the catalog to the ordinary InterCode-Bash system prompt.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from .checkpoint import (
    base_model_load_location,
    build_base_model_identity,
    validate_base_model_identity,
)
from .io_utils import (
    PACKAGE_ROOT,
    canonical_json_bytes,
    ensure_output_path,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_directory,
    sha256_file,
    sha256_text,
    write_json,
)
from .prepare_data import (
    VIEW_INPUT_ARTIFACTS,
    load_procedure_integrity,
    load_provenance_integrity,
)
from .training_data import SYSTEM_PROMPT
from .unigram import Piece, ProcedureUnigramModel


BASE_TEXT_METHOD = "base_tool_desc"
BASE_TEXT_ARTIFACT_SCHEMA = "intercode_bash_base_text_artifact_v1"
TOOL_CATALOG_REPORT_SCHEMA = "intercode_bash_tool_catalog_v1"
DEFAULT_MAX_CATALOG_TOKENS = 3072
TOOL_CATALOG_FILENAME = "tool_catalog.txt"
TOOL_CATALOG_REPORT_FILENAME = "tool_catalog.report.json"
ARTIFACT_METADATA_FILENAME = "checkpoint.json"
TOKENIZER_DIRECTORY_NAME = "tokenizer"

_CONNECTOR_TEXT = {
    "START": "",
    "PIPE": "| ",
    "PIPE_STDERR": "|& ",
}
_UNKNOWN_UTILITY = "<UNK_UTILITY>"


@dataclass(frozen=True)
class ToolCatalog:
    """Rendered catalog text and the evidence needed to audit it."""

    text: str
    report: dict[str, Any]


def tokenizer_vocabulary_hash(tokenizer) -> str:
    """Hash the exact token-to-ID mapping used to count the catalog."""

    vocabulary = tokenizer.get_vocab()
    normalized = sorted(
        (str(token), int(token_id)) for token, token_id in vocabulary.items()
    )
    return sha256_bytes(canonical_json_bytes(normalized))


def _encode(tokenizer, text: str) -> list[int]:
    values = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(values, torch.Tensor):
        values = values.flatten().tolist()
    return [int(value) for value in values]


def _piece_text(piece: Piece) -> str:
    fields: list[str] = []
    for connector, utility in piece:
        if connector not in _CONNECTOR_TEXT:
            raise ValueError(f"Unknown Bash atom connector: {connector!r}")
        fields.append(f"{_CONNECTOR_TEXT[connector]}{utility}")
    return " ".join(fields)


def _validate_training_atoms(
    atom_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return deterministic, structurally checked TRAIN atom records."""

    training_records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for source_record in atom_records:
        if source_record.get("derived_split") != "TRAIN":
            continue
        record = dict(source_record)
        sample_id = record.get("sample_id")
        source_group_id = record.get("source_group_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("A TRAIN atom record lacks a non-empty sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate TRAIN atom sample_id: {sample_id}")
        sample_ids.add(sample_id)
        if not isinstance(source_group_id, str) or not source_group_id:
            raise ValueError(
                f"TRAIN atom record {sample_id} lacks source_group_id"
            )
        atoms = record.get("atoms")
        signatures = record.get("canonical_signatures")
        if not isinstance(atoms, list) or not isinstance(signatures, list):
            raise TypeError(
                f"TRAIN atom record {sample_id} lacks atoms/signatures lists"
            )
        if not atoms or len(atoms) != len(signatures):
            raise ValueError(
                f"TRAIN atom record {sample_id} has inconsistent atom counts"
            )
        normalized_signatures: list[tuple[str, str]] = []
        for atom_index, (atom, signature_value) in enumerate(
            zip(atoms, signatures)
        ):
            if not isinstance(atom, Mapping):
                raise TypeError(
                    f"TRAIN atom {sample_id}:{atom_index} is not an object"
                )
            if (
                not isinstance(signature_value, (list, tuple))
                or len(signature_value) != 2
            ):
                raise ValueError(
                    f"TRAIN atom {sample_id}:{atom_index} has invalid signature"
                )
            signature = (
                str(signature_value[0]),
                str(signature_value[1]),
            )
            atom_signature_value = atom.get("canonical_signature")
            if (
                not isinstance(atom_signature_value, (list, tuple))
                or len(atom_signature_value) != 2
                or tuple(str(value) for value in atom_signature_value)
                != signature
            ):
                raise ValueError(
                    f"TRAIN atom {sample_id}:{atom_index} signature differs "
                    "from canonical_signatures"
                )
            raw_core = atom.get("raw_core")
            if not isinstance(raw_core, str) or not raw_core.strip():
                raise ValueError(
                    f"TRAIN atom {sample_id}:{atom_index} lacks raw_core"
                )
            normalized_signatures.append(signature)
        record["_normalized_signatures"] = tuple(normalized_signatures)
        training_records.append(record)
    if not training_records:
        raise ValueError("No derived TRAIN atom records are available")
    training_records.sort(key=lambda item: str(item["sample_id"]).encode("utf-8"))
    return training_records


def _singleton_example(
    piece_id: int,
    piece: Piece,
    training_records: Sequence[Mapping[str, Any]],
    tokenizer,
) -> dict[str, Any]:
    if len(piece) != 1:
        raise ValueError(f"Procedure {piece_id} is not a singleton")
    connector, utility = piece[0]
    occurrences: list[dict[str, Any]] = []
    for record in training_records:
        signatures = record["_normalized_signatures"]
        for atom_index, signature in enumerate(signatures):
            if signature != piece[0]:
                continue
            atom = record["atoms"][atom_index]
            raw_core = str(atom["raw_core"])
            has_usage_detail = (
                utility != _UNKNOWN_UTILITY
                and raw_core.strip() != utility
            )
            occurrences.append(
                {
                    "piece_id": piece_id,
                    "signature": [connector, utility],
                    "sample_id": str(record["sample_id"]),
                    "source_group_id": str(record["source_group_id"]),
                    "atom_index": atom_index,
                    "raw_core": raw_core,
                    "has_usage_detail": has_usage_detail,
                    "raw_token_count": len(_encode(tokenizer, raw_core)),
                }
            )
    if not occurrences:
        raise ValueError(
            f"Mandatory singleton procedure {piece_id} has no TRAIN occurrence"
        )
    occurrences.sort(
        key=lambda item: (
            not bool(item["has_usage_detail"]),
            int(item["raw_token_count"]),
            len(str(item["raw_core"]).encode("utf-8")),
            str(item["raw_core"]).encode("utf-8"),
            str(item["sample_id"]).encode("utf-8"),
            int(item["atom_index"]),
        )
    )
    return occurrences[0]


def render_tool_catalog(
    model: ProcedureUnigramModel,
    atom_records: Iterable[Mapping[str, Any]],
    tokenizer,
    *,
    max_catalog_tokens: int = DEFAULT_MAX_CATALOG_TOKENS,
) -> ToolCatalog:
    """Render every procedure plus one example for each mandatory singleton.

    Only rows whose derived split is exactly ``TRAIN`` are considered for
    examples.  Natural-language task instructions are intentionally excluded:
    they describe whole commands and cannot be reliably assigned to a latent
    procedure span.
    """

    if max_catalog_tokens <= 0:
        raise ValueError("max_catalog_tokens must be positive")
    training_records = _validate_training_atoms(atom_records)
    structure_lines: list[str] = []
    example_records: list[dict[str, Any]] = []
    for piece_id, piece in enumerate(model.pieces):
        if not piece:
            raise ValueError(f"Procedure {piece_id} is empty")
        structure_lines.append(f"- {_piece_text(piece)}")
        if model.mandatory[piece_id]:
            if len(piece) != 1:
                raise ValueError(
                    f"Mandatory procedure {piece_id} is not a singleton"
                )
            example_records.append(
                _singleton_example(
                    piece_id,
                    piece,
                    training_records,
                    tokenizer,
                )
            )

    example_lines = [
        (
            f"- {_piece_text(model.pieces[int(record['piece_id'])])}: "
            f"{json.dumps(record['raw_core'], ensure_ascii=False)}"
        )
        for record in example_records
    ]
    text = (
        "Training-only Bash procedure catalog.\n"
        "`|` passes stdout; `|&` passes stdout and stderr. "
        "Treat each line as an ordinary Bash form.\n"
        "Available procedure structures:\n"
        + "\n".join(structure_lines)
        + "\nAtomic utility usage examples:\n"
        + "\n".join(example_lines)
        + "\n"
    )
    token_count = len(_encode(tokenizer, text))
    if token_count > max_catalog_tokens:
        raise ValueError(
            f"Tool catalog uses {token_count} tokens, exceeding the fixed "
            f"limit {max_catalog_tokens}; query-dependent truncation or "
            "retrieval is forbidden for the full-catalog baseline"
        )
    report = {
        "schema": TOOL_CATALOG_REPORT_SCHEMA,
        "procedure_count": model.size,
        "structure_count": len(structure_lines),
        "mandatory_singleton_count": sum(model.mandatory),
        "example_count": len(example_records),
        "procedure_model_hash": model.model_hash(),
        "procedure_inventory_hash": model.inventory_hash(),
        "catalog_token_count": token_count,
        "max_catalog_tokens": int(max_catalog_tokens),
        "tokenizer_vocabulary_sha256": tokenizer_vocabulary_hash(tokenizer),
        "catalog_sha256": sha256_text(text),
        "examples": example_records,
    }
    return ToolCatalog(text=text, report=report)


def _integrity_hashes(
    integrity: Mapping[str, Any],
) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in VIEW_INPUT_ARTIFACTS:
        record = integrity.get(name)
        if not isinstance(record, Mapping) or not isinstance(
            record.get("sha256"),
            str,
        ):
            raise ValueError(
                f"Procedure integrity lacks SHA-256 for {name!r}"
            )
        values[name] = str(record["sha256"])
    return values


def build_tool_catalog_from_artifacts(
    source_artifact_dir: str | Path,
    tokenizer,
    *,
    max_catalog_tokens: int = DEFAULT_MAX_CATALOG_TOKENS,
    require_primary: bool = True,
) -> ToolCatalog:
    """Build a catalog after validating the full provenance/procedure chain."""

    source_artifact_dir = Path(source_artifact_dir)
    provenance_integrity = load_provenance_integrity(
        source_artifact_dir,
        require_primary=require_primary,
    )
    procedure_integrity = load_procedure_integrity(
        source_artifact_dir,
        provenance_integrity,
        require_primary=require_primary,
    )
    formal_ready = bool(
        procedure_integrity.get("primary_ready") is True
        and procedure_integrity.get("primary_run") is True
        and procedure_integrity.get("deny_policy") == "all_candidates"
    )
    if require_primary and not formal_ready:
        raise RuntimeError(
            "A formal base-text catalog requires primary, all-candidates "
            "provenance/procedure artifacts"
        )
    model = ProcedureUnigramModel.from_dict(
        read_json(source_artifact_dir / "procedure_lexicon.json")
    )
    if model.size != int(procedure_integrity.get("procedure_count", -1)):
        raise ValueError("Procedure integrity count differs from the lexicon")
    if model.model_hash() != procedure_integrity.get("procedure_model_hash"):
        raise ValueError("Procedure integrity model hash differs from the lexicon")
    if (
        model.inventory_hash()
        != procedure_integrity.get("procedure_inventory_hash")
    ):
        raise ValueError(
            "Procedure integrity inventory hash differs from the lexicon"
        )
    catalog = render_tool_catalog(
        model,
        read_jsonl(source_artifact_dir / "atoms.jsonl"),
        tokenizer,
        max_catalog_tokens=max_catalog_tokens,
    )
    report = {
        **catalog.report,
        "primary_ready": procedure_integrity.get("primary_ready") is True,
        "primary_run": procedure_integrity.get("primary_run") is True,
        "deny_policy": procedure_integrity.get("deny_policy"),
        "formal_ready": formal_ready,
        "input_sha256": _integrity_hashes(procedure_integrity),
    }
    return ToolCatalog(text=catalog.text, report=report)


def compose_base_text_system_prompt(
    catalog_text: str,
    *,
    base_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Append one fixed catalog to the ordinary Bash system instruction."""

    if not catalog_text.strip():
        raise ValueError("catalog_text must not be empty")
    return f"{base_prompt.rstrip()}\n\n{catalog_text.rstrip()}"


def _validate_unaugmented_tokenizer(tokenizer) -> None:
    unexpected = sorted(
        token
        for token in tokenizer.get_vocab()
        if str(token).startswith("<|procedure_memory_")
        or str(token) == "<|procedure_eoc|>"
    )
    if unexpected:
        raise ValueError(
            "The base-text artifact must use the unaugmented base tokenizer; "
            f"found procedural control tokens: {unexpected[:3]}"
        )


def save_base_text_artifact(
    source_artifact_dir: str | Path,
    output_dir: str | Path,
    tokenizer,
    *,
    model_name: str,
    model_revision: str | None,
    max_catalog_tokens: int = DEFAULT_MAX_CATALOG_TOKENS,
    require_primary: bool = True,
    base_model_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a no-training base artifact using an already loaded tokenizer."""

    _validate_unaugmented_tokenizer(tokenizer)
    output_dir = ensure_output_path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty base artifact directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = build_tool_catalog_from_artifacts(
        source_artifact_dir,
        tokenizer,
        max_catalog_tokens=max_catalog_tokens,
        require_primary=require_primary,
    )
    if base_model_identity is None:
        identity = build_base_model_identity(
            model_name,
            model_revision,
            require_reproducible=require_primary,
        )
    else:
        identity = dict(base_model_identity)
        validate_base_model_identity(identity)
        if require_primary and identity.get("reproducible") is not True:
            raise RuntimeError(
                "A formal base-text artifact requires an immutable base model"
            )
    load_name, load_revision = base_model_load_location(identity)

    catalog_path = output_dir / TOOL_CATALOG_FILENAME
    report_path = output_dir / TOOL_CATALOG_REPORT_FILENAME
    tokenizer_dir = output_dir / TOKENIZER_DIRECTORY_NAME
    catalog_path.write_text(catalog.text, encoding="utf-8")
    write_json(report_path, catalog.report)
    tokenizer.save_pretrained(tokenizer_dir)
    if not tokenizer_dir.is_dir() or not any(tokenizer_dir.iterdir()):
        raise RuntimeError("Tokenizer save_pretrained produced no files")

    formal_ready = bool(
        catalog.report.get("formal_ready") is True
        and identity.get("reproducible") is True
    )
    metadata = {
        "schema": BASE_TEXT_ARTIFACT_SCHEMA,
        "method": BASE_TEXT_METHOD,
        "base_model_name": load_name,
        "base_model_revision": load_revision,
        "base_model_identity": identity,
        "formal_ready": formal_ready,
        "trainable_parameter_count": 0,
        "catalog_token_count": catalog.report["catalog_token_count"],
        "max_catalog_tokens": catalog.report["max_catalog_tokens"],
        "catalog_sha256": sha256_file(catalog_path),
        "catalog_report_sha256": sha256_file(report_path),
        "tokenizer_sha256": sha256_directory(tokenizer_dir),
        "tokenizer_vocabulary_sha256": tokenizer_vocabulary_hash(tokenizer),
        "system_prompt_sha256": sha256_text(
            compose_base_text_system_prompt(catalog.text)
        ),
        "artifact_integrity": {
            "primary_ready": catalog.report.get("primary_ready") is True,
            "primary_run": catalog.report.get("primary_run") is True,
            "deny_policy": catalog.report.get("deny_policy"),
            "formal_ready": catalog.report.get("formal_ready") is True,
            "input_sha256": dict(catalog.report.get("input_sha256", {})),
        },
    }
    write_json(output_dir / ARTIFACT_METADATA_FILENAME, metadata)
    validate_base_text_artifact(output_dir, tokenizer=tokenizer)
    return metadata


def prepare_base_text_artifact(
    source_artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str,
    model_revision: str | None = None,
    max_catalog_tokens: int = DEFAULT_MAX_CATALOG_TOKENS,
    require_primary: bool = True,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Load the base tokenizer and materialize an artifact without training."""

    from transformers import AutoTokenizer

    identity = build_base_model_identity(
        model_name,
        model_revision,
        require_reproducible=require_primary,
    )
    load_name, load_revision = base_model_load_location(identity)
    tokenizer = AutoTokenizer.from_pretrained(
        load_name,
        revision=load_revision,
        local_files_only=local_files_only,
    )
    return save_base_text_artifact(
        source_artifact_dir,
        output_dir,
        tokenizer,
        model_name=load_name,
        model_revision=load_revision,
        max_catalog_tokens=max_catalog_tokens,
        require_primary=require_primary,
        base_model_identity=identity,
    )


def validate_base_text_artifact(
    artifact_dir: str | Path,
    *,
    tokenizer=None,
) -> dict[str, Any]:
    """Validate every saved base-text artifact component and its hashes."""

    artifact_dir = Path(artifact_dir)
    metadata = read_json(artifact_dir / ARTIFACT_METADATA_FILENAME)
    if metadata.get("schema") != BASE_TEXT_ARTIFACT_SCHEMA:
        raise ValueError("Unknown base-text artifact schema")
    if metadata.get("method") != BASE_TEXT_METHOD:
        raise ValueError("Base-text artifact has the wrong method")
    if int(metadata.get("trainable_parameter_count", -1)) != 0:
        raise ValueError("A base-text artifact must contain no trainable parameters")
    identity = metadata.get("base_model_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Base-text artifact lacks base_model_identity")
    validate_base_model_identity(identity)
    load_name, load_revision = base_model_load_location(identity)
    if metadata.get("base_model_name") != load_name:
        raise ValueError("Base-text model name differs from its identity")
    if metadata.get("base_model_revision") != load_revision:
        raise ValueError("Base-text model revision differs from its identity")

    catalog_path = artifact_dir / TOOL_CATALOG_FILENAME
    report_path = artifact_dir / TOOL_CATALOG_REPORT_FILENAME
    tokenizer_dir = artifact_dir / TOKENIZER_DIRECTORY_NAME
    if sha256_file(catalog_path) != metadata.get("catalog_sha256"):
        raise ValueError("Base-text catalog differs from checkpoint.json")
    if sha256_file(report_path) != metadata.get("catalog_report_sha256"):
        raise ValueError("Base-text catalog report differs from checkpoint.json")
    if sha256_directory(tokenizer_dir) != metadata.get("tokenizer_sha256"):
        raise ValueError("Base-text tokenizer differs from checkpoint.json")

    catalog_text = catalog_path.read_text(encoding="utf-8")
    report = read_json(report_path)
    if report.get("schema") != TOOL_CATALOG_REPORT_SCHEMA:
        raise ValueError("Unknown tool-catalog report schema")
    if report.get("catalog_sha256") != sha256_text(catalog_text):
        raise ValueError("Tool catalog differs from its report")
    for field in (
        "catalog_token_count",
        "max_catalog_tokens",
        "tokenizer_vocabulary_sha256",
    ):
        if report.get(field) != metadata.get(field):
            raise ValueError(
                f"Catalog report field {field!r} differs from checkpoint.json"
            )
    if int(report["catalog_token_count"]) > int(report["max_catalog_tokens"]):
        raise ValueError("Saved tool catalog exceeds its fixed token limit")
    expected_formal = bool(
        report.get("formal_ready") is True
        and identity.get("reproducible") is True
    )
    artifact_integrity = metadata.get("artifact_integrity")
    if not isinstance(artifact_integrity, Mapping):
        raise ValueError("Base-text artifact lacks source integrity metadata")
    expected_integrity = {
        "primary_ready": report.get("primary_ready") is True,
        "primary_run": report.get("primary_run") is True,
        "deny_policy": report.get("deny_policy"),
        "formal_ready": report.get("formal_ready") is True,
        "input_sha256": dict(report.get("input_sha256", {})),
    }
    if dict(artifact_integrity) != expected_integrity:
        raise ValueError(
            "Base-text source integrity differs from the catalog report"
        )
    if metadata.get("formal_ready") is not expected_formal:
        raise ValueError("Base-text formal_ready status is inconsistent")
    if metadata.get("system_prompt_sha256") != sha256_text(
        compose_base_text_system_prompt(catalog_text)
    ):
        raise ValueError("Base-text system prompt differs from checkpoint.json")

    if tokenizer is not None:
        _validate_unaugmented_tokenizer(tokenizer)
        vocabulary_hash = tokenizer_vocabulary_hash(tokenizer)
        if vocabulary_hash != report.get("tokenizer_vocabulary_sha256"):
            raise ValueError("Loaded tokenizer vocabulary differs from the catalog")
        actual_tokens = len(_encode(tokenizer, catalog_text))
        if actual_tokens != int(report["catalog_token_count"]):
            raise ValueError("Loaded tokenizer gives a different catalog token count")
    return metadata


class FrozenBaseTextAdapter(nn.Module):
    """Greedy inference adapter for an unmodified, permanently frozen LM."""

    procedure_token_ids_to_strip: tuple[int, ...] = ()
    eoc_token_id_to_strip: None = None

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        self.train(False)

    def train(self, mode: bool = True) -> "FrozenBaseTextAdapter":
        # This artifact is an inference-only baseline.  Even an accidental
        # model.train() call must not enable dropout in the frozen backbone.
        super().train(False)
        return self

    @torch.inference_mode()
    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        response_end_sequences: Sequence[Sequence[int]],
        max_new_tokens: int = 512,
    ) -> dict[str, Any]:
        """Decode by exact greedy argmax with the same stop contract as TapMem."""

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("InterCode generation requires batch size one")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape differs from input_ids")
        stop_sequences = [
            [int(value) for value in sequence]
            for sequence in response_end_sequences
            if sequence
        ]
        generated: list[int] = []
        running_input_ids = input_ids
        output = self.base_model(
            input_ids=running_input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past = getattr(output, "past_key_values", None)
        next_logits = output.logits[:, -1, :]
        terminated = False
        for _step in range(max_new_tokens):
            next_token = int(torch.argmax(next_logits, dim=-1).item())
            generated.append(next_token)
            if any(
                len(sequence) <= len(generated)
                and generated[-len(sequence) :] == sequence
                for sequence in stop_sequences
            ):
                terminated = True
                break

            token_tensor = torch.tensor(
                [[next_token]],
                dtype=torch.long,
                device=input_ids.device,
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
            running_input_ids = torch.cat(
                [running_input_ids, token_tensor],
                dim=1,
            )
            if past is None:
                output = self.base_model(
                    input_ids=running_input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                output = self.base_model(
                    input_ids=token_tensor,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            past = getattr(output, "past_key_values", None)
            next_logits = output.logits[:, -1, :]
        return {
            "generated_ids": generated,
            "terminated": terminated,
            "missing_terminator": not terminated,
        }


def load_base_text_artifact(
    artifact_dir: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    local_files_only: bool = True,
) -> tuple[FrozenBaseTextAdapter, Any, dict[str, Any], str]:
    """Load a validated artifact for later evaluator schema dispatch."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    artifact_dir = Path(artifact_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        artifact_dir / TOKENIZER_DIRECTORY_NAME,
        local_files_only=True,
    )
    metadata = validate_base_text_artifact(
        artifact_dir,
        tokenizer=tokenizer,
    )
    model_name, model_revision = base_model_load_location(
        metadata["base_model_identity"]
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    base_model.to(device)
    adapter = FrozenBaseTextAdapter(base_model)
    catalog_text = (
        artifact_dir / TOOL_CATALOG_FILENAME
    ).read_text(encoding="utf-8")
    return adapter, tokenizer, metadata, catalog_text


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a no-training Base+ToolDesc artifact from the fixed "
            "TRAIN-only procedure catalog."
        )
    )
    parser.add_argument("--source-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--max-catalog-tokens",
        type=int,
        default=DEFAULT_MAX_CATALOG_TOKENS,
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--allow-exploratory",
        action="store_true",
        help="Build a debugging-only artifact that cannot be used as a paper result",
    )
    args = parser.parse_args(argv)
    if args.max_catalog_tokens <= 0:
        parser.error("--max-catalog-tokens must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = prepare_base_text_artifact(
        args.source_artifact_dir,
        args.output_dir,
        model_name=args.model_name,
        model_revision=args.model_revision,
        max_catalog_tokens=args.max_catalog_tokens,
        require_primary=not args.allow_exploratory,
        local_files_only=not args.allow_download,
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


__all__ = [
    "ARTIFACT_METADATA_FILENAME",
    "BASE_TEXT_ARTIFACT_SCHEMA",
    "BASE_TEXT_METHOD",
    "DEFAULT_MAX_CATALOG_TOKENS",
    "FrozenBaseTextAdapter",
    "TOKENIZER_DIRECTORY_NAME",
    "TOOL_CATALOG_FILENAME",
    "TOOL_CATALOG_REPORT_FILENAME",
    "ToolCatalog",
    "build_tool_catalog_from_artifacts",
    "compose_base_text_system_prompt",
    "load_base_text_artifact",
    "main",
    "parse_args",
    "prepare_base_text_artifact",
    "render_tool_catalog",
    "save_base_text_artifact",
    "tokenizer_vocabulary_hash",
    "validate_base_text_artifact",
]


if __name__ == "__main__":
    main()
