from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
COMPOSITIONAL_DIR = REPO_ROOT / "compositional"
if str(COMPOSITIONAL_DIR) not in sys.path:
    sys.path.insert(0, str(COMPOSITIONAL_DIR))

from backbone_registry import resolve_function_calling_model_class  # noqa: E402
from checkpoint_io import load_checkpoint_into_model  # noqa: E402

from .context import ObservationPolicy, render_step_context
from .dataset import (
    tokenize_context_with_memory_slots,
    truncate_preserving_supervised_target,
)
from .decode_one_call import decode_one_call
from .manifest import load_manifest
from .masked_routing import build_available_tool_mask
from .rollout import Action


METHOD_FLAGS = {
    "tokmem": {"use_eoc": False, "use_logit_bias": False},
    "eoc_only": {"use_eoc": True, "use_logit_bias": False},
    "tapmem": {"use_eoc": True, "use_logit_bias": True},
}


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


@dataclass
class LoadedToolModel:
    model: Any
    tokenizer: Any
    manifest: dict[str, Any]
    method: str
    device: str
    max_length: int


def load_tool_model(
    *,
    run_dir: str | Path,
    manifest_path: str | Path,
    device: str | None = None,
    dtype: str | None = None,
    max_length: int | None = None,
) -> LoadedToolModel:
    run_dir = Path(run_dir)
    run_config = json.loads(
        (run_dir / "run_config.json").read_text(encoding="utf-8")
    )
    manifest = load_manifest(manifest_path)
    if run_config.get("manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("run config and runtime manifest hashes differ")
    method = run_config.get("method")
    if method not in METHOD_FLAGS:
        raise ValueError(f"checkpoint has unsupported method: {method!r}")
    flags = METHOD_FLAGS[method]
    runtime_device = device or run_config["device"]
    runtime_dtype = dtype or run_config["dtype"]
    if runtime_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime was requested but is not visible")

    checkpoint_path = run_dir / "checkpoint_trainable.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    toolathlon_metadata = checkpoint.get("toolathlon")
    if not isinstance(toolathlon_metadata, dict):
        raise ValueError("checkpoint lacks Toolathlon metadata")
    if toolathlon_metadata.get("manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("checkpoint and runtime manifest hashes differ")
    if toolathlon_metadata.get("method") != method:
        raise ValueError("checkpoint and run config method differ")

    model_name = checkpoint["model_metadata"]["base_model"]
    if os.path.realpath(run_config["model_name"]) != os.path.realpath(model_name):
        raise ValueError("checkpoint and run config base model differ")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    tool_names = [record["stable_id"] for record in manifest["tools"]]
    model_class = resolve_function_calling_model_class(model_name)
    model = model_class(
        model_name=model_name,
        num_tools=len(tool_names),
        tool_names=tool_names,
        tokenizer=tokenizer,
        device=runtime_device,
        dtype=_dtype(runtime_dtype),
        decouple_embeddings=bool(run_config["decouple_embeddings"]),
        lora_config=None,
        use_eoc=flags["use_eoc"],
        use_logit_bias=flags["use_logit_bias"],
        use_tool_head_replacement=False,
        logit_bias_network=run_config["logit_bias_network"],
        logit_bias_scale=float(run_config["logit_bias_scale"]),
    )
    load_checkpoint_into_model(model, checkpoint)
    model.eval()
    return LoadedToolModel(
        model=model,
        tokenizer=tokenizer,
        manifest=manifest,
        method=method,
        device=runtime_device,
        max_length=int(max_length or run_config["max_length"]),
    )


class TokMemClosedLoopPolicy:
    def __init__(
        self,
        loaded: LoadedToolModel,
        *,
        max_new_tokens: int = 768,
        observation_policy: ObservationPolicy = ObservationPolicy(),
    ) -> None:
        self.loaded = loaded
        self.max_new_tokens = int(max_new_tokens)
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.context_budget = int(loaded.max_length) - self.max_new_tokens
        if self.context_budget <= 0:
            raise ValueError(
                "checkpoint max_length must be larger than max_new_tokens"
            )
        self.observation_policy = observation_policy
        self.tool_names = [
            record["stable_id"] for record in loaded.manifest["tools"]
        ]
        self.tool_slots = {
            record["stable_id"]: int(record["memory_slot"])
            for record in loaded.manifest["tools"]
        }
        self.slot_to_tool = {
            slot: tool_id for tool_id, slot in self.tool_slots.items()
        }

    async def next_action(
        self,
        *,
        instruction: str,
        available_tool_ids: list[str],
        history: list[dict[str, Any]],
    ) -> Action:
        step = {
            "instruction": instruction,
            "history": history,
        }
        context_text = render_step_context(
            step,
            self.tool_slots,
            observation_policy=self.observation_policy,
        )
        context_ids = tokenize_context_with_memory_slots(
            context_text,
            tokenizer=self.loaded.tokenizer,
            model=self.loaded.model,
            slot_to_tool=self.slot_to_tool,
        )
        context_ids, _ = truncate_preserving_supervised_target(
            context_ids,
            [-100] * len(context_ids),
            self.context_budget,
        )
        input_ids = torch.tensor(
            [context_ids],
            dtype=torch.long,
            device=self.loaded.device,
        )
        attention_mask = torch.ones_like(input_ids)
        available_mask = build_available_tool_mask(
            self.tool_names,
            available_tool_ids,
        ).to(self.loaded.device)
        flags = METHOD_FLAGS[self.loaded.method]
        decoded = decode_one_call(
            model=self.loaded.model,
            tokenizer=self.loaded.tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            available_tool_mask=available_mask,
            use_eoc=flags["use_eoc"],
            use_logit_bias=flags["use_logit_bias"],
            max_new_tokens=self.max_new_tokens,
        )
        return Action(
            tool_id=decoded.tool_id,
            arguments=decoded.arguments,
            stop_reason=decoded.stop_reason,
        )
