#!/usr/bin/env python3
"""CPU-only checks for full and trainable-only compositional checkpoints."""

import io
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)
from transformers import LlamaConfig, LlamaForCausalLM

from checkpoint_io import (
    TRAINABLE_CHECKPOINT_FORMAT,
    build_checkpoint_payload,
    checkpoint_tool_names,
    load_checkpoint_into_model,
    load_lora_adapter,
)
from main_sequential import build_parser


class TinyWrapper(nn.Module):
    def __init__(
        self,
        *,
        decoupled=False,
        use_eoc=False,
        with_head=False,
        tool_names=None,
    ):
        super().__init__()
        self.model_name = "tiny-base"
        self.config = SimpleNamespace(model_type="tiny")
        self.tool_names = list(tool_names or ["tool_b", "tool_a", "tool_c"])
        self.num_tools = len(self.tool_names)
        self.decouple_embeddings = decoupled
        self.use_eoc = use_eoc
        self.use_logit_bias = with_head
        self.use_tool_head_replacement = False
        self.use_memory_bank_constraint = False
        self.memory_bank_probability_threshold = 0.5
        self.logit_bias_network = "mlp" if with_head else "linear"
        self.logit_bias_scale = 1.0
        self.lora_config = None

        slot_count = self.num_tools + int(use_eoc)
        self.trainable_reserved_token_names = [
            f"<|reserved_special_token_{index}|>"
            for index in range(slot_count)
        ]
        self.trainable_reserved_token_ids = list(range(100, 100 + slot_count))
        self.eoc_token_name = (
            self.trainable_reserved_token_names[-1] if use_eoc else None
        )
        self.eoc_token_id = (
            self.trainable_reserved_token_ids[-1] if use_eoc else None
        )

        if decoupled:
            self.trainable_tool_input_embeddings = nn.Parameter(
                torch.zeros(slot_count, 5)
            )
            self.trainable_tool_output_embeddings = nn.Parameter(
                torch.zeros(slot_count, 5)
            )
        else:
            self.trainable_tool_embeddings = nn.Parameter(
                torch.zeros(slot_count, 5)
            )
            # Match the real coupled wrapper's backward-compatible aliases.
            self.trainable_tool_input_embeddings = self.trainable_tool_embeddings
            self.trainable_tool_output_embeddings = self.trainable_tool_embeddings

        self.logit_bias_head = (
            nn.Sequential(
                nn.Linear(5, 5),
                nn.GELU(),
                nn.Linear(5, self.num_tools),
            )
            if with_head
            else None
        )
        self.model = nn.Linear(5, 5, bias=False)
        for parameter in self.model.parameters():
            parameter.requires_grad = False


def serialized(payload):
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def tiny_base_model():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    config._name_or_path = "tiny-base"
    return LlamaForCausalLM(config)


def tiny_peft_model():
    return get_peft_model(
        tiny_base_model(),
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj"],
            bias="none",
        ),
    )


class CheckpointIOTest(unittest.TestCase):
    def test_default_format_remains_full_and_legacy_payload_loads_strictly(self):
        args = build_parser().parse_args(
            ["--model_name", "unused", "--training_rounds", "1-2:1"]
        )
        self.assertEqual(args.checkpoint_format, "full")

        source = TinyWrapper()
        source.trainable_tool_embeddings.data.fill_(2.0)
        payload = build_checkpoint_payload(
            source,
            1,
            ["tool_c"],
            {"loss": 0.1},
        )
        self.assertEqual(
            set(payload),
            {"round", "tools", "model_state_dict", "results"},
        )

        target = TinyWrapper()
        loaded_format = load_checkpoint_into_model(target, payload)
        self.assertEqual(loaded_format, "full")
        torch.testing.assert_close(
            target.trainable_tool_embeddings,
            source.trainable_tool_embeddings,
        )

    def test_trainable_coupled_memory_round_trip_excludes_backbone(self):
        source = TinyWrapper()
        source.trainable_tool_embeddings.data.copy_(
            torch.arange(15, dtype=torch.float32).reshape(3, 5)
        )
        payload = serialized(
            build_checkpoint_payload(
                source,
                1,
                ["tool_c"],
                {},
                checkpoint_format="trainable_only",
                base_model_name="tiny-base",
            )
        )

        self.assertEqual(
            payload["checkpoint_format"],
            TRAINABLE_CHECKPOINT_FORMAT,
        )
        self.assertEqual(
            set(payload["trainable_state"]["memory"]),
            {"mode", "embeddings"},
        )
        self.assertNotIn("model_state_dict", payload)
        self.assertEqual(payload["tools"], ["tool_c"])
        self.assertEqual(checkpoint_tool_names(payload), source.tool_names)

        target = TinyWrapper()
        target.trainable_tool_embeddings.data.fill_(-1.0)
        base_before = target.model.weight.detach().clone()
        load_checkpoint_into_model(target, payload)

        torch.testing.assert_close(
            target.trainable_tool_embeddings,
            source.trainable_tool_embeddings,
        )
        torch.testing.assert_close(target.model.weight, base_before)
        self.assertIs(
            target.trainable_tool_input_embeddings,
            target.trainable_tool_embeddings,
        )
        self.assertIs(
            target.trainable_tool_output_embeddings,
            target.trainable_tool_embeddings,
        )

    def test_decoupled_memory_eoc_and_head_round_trip(self):
        source = TinyWrapper(decoupled=True, use_eoc=True, with_head=True)
        source.trainable_tool_input_embeddings.data.copy_(
            torch.arange(20, dtype=torch.float32).reshape(4, 5)
        )
        source.trainable_tool_output_embeddings.data.copy_(
            torch.arange(20, 40, dtype=torch.float32).reshape(4, 5)
        )
        for index, parameter in enumerate(source.logit_bias_head.parameters()):
            parameter.data.fill_(index + 0.25)

        payload = serialized(
            build_checkpoint_payload(
                source,
                2,
                ["tool_c"],
                {},
                checkpoint_format="trainable_only",
                base_model_name="tiny-base",
            )
        )
        target = TinyWrapper(decoupled=True, use_eoc=True, with_head=True)
        load_checkpoint_into_model(target, payload)

        torch.testing.assert_close(
            target.trainable_tool_input_embeddings,
            source.trainable_tool_input_embeddings,
        )
        torch.testing.assert_close(
            target.trainable_tool_output_embeddings,
            source.trainable_tool_output_embeddings,
        )
        for target_parameter, source_parameter in zip(
            target.logit_bias_head.parameters(),
            source.logit_bias_head.parameters(),
        ):
            torch.testing.assert_close(target_parameter, source_parameter)

    def test_frozen_lora_adapter_is_saved_and_restored(self):
        lora_config = {
            "r": 2,
            "alpha": 4,
            "dropout": 0.0,
            "target_modules": ["q_proj"],
        }
        source = TinyWrapper()
        source.model = tiny_peft_model()
        source.lora_config = dict(lora_config)
        for name, parameter in source.model.named_parameters():
            if "lora_A" in name:
                parameter.data.fill_(0.25)
                parameter.requires_grad = False
            elif "lora_B" in name:
                parameter.data.fill_(0.5)
                parameter.requires_grad = False

        payload = serialized(
            build_checkpoint_payload(
                source,
                1,
                source.tool_names,
                {},
                checkpoint_format="trainable_only",
                base_model_name="tiny-base",
            )
        )
        self.assertTrue(payload["trainable_state"]["lora"]["state_dict"])

        target = TinyWrapper()
        target.model = tiny_peft_model()
        target.lora_config = dict(lora_config)
        base_before = {
            name: parameter.detach().clone()
            for name, parameter in target.model.named_parameters()
            if "lora_" not in name
        }
        load_checkpoint_into_model(target, payload)

        source_adapter = get_peft_model_state_dict(
            source.model,
            adapter_name="default",
            save_embedding_layers=False,
        )
        target_adapter = get_peft_model_state_dict(
            target.model,
            adapter_name="default",
            save_embedding_layers=False,
        )
        self.assertEqual(set(source_adapter), set(target_adapter))
        for name in source_adapter:
            torch.testing.assert_close(target_adapter[name], source_adapter[name])
        for name, parameter in target.model.named_parameters():
            if name in base_before:
                torch.testing.assert_close(parameter, base_before[name])

    def test_tool_order_mismatch_fails_before_loading_state(self):
        source = TinyWrapper(tool_names=["tool_b", "tool_a", "tool_c"])
        source.trainable_tool_embeddings.data.fill_(4.0)
        payload = build_checkpoint_payload(
            source,
            1,
            ["tool_c"],
            {},
            checkpoint_format="trainable_only",
            base_model_name="tiny-base",
        )
        target = TinyWrapper(tool_names=["tool_a", "tool_b", "tool_c"])
        before = target.trainable_tool_embeddings.detach().clone()

        with self.assertRaisesRegex(ValueError, "tool_names mismatch"):
            load_checkpoint_into_model(target, payload)
        torch.testing.assert_close(target.trainable_tool_embeddings, before)

    def test_base_model_and_logit_bias_scale_mismatches_are_rejected(self):
        source = TinyWrapper(use_eoc=True, with_head=True)
        payload = build_checkpoint_payload(
            source,
            1,
            source.tool_names,
            {},
            checkpoint_format="trainable_only",
            base_model_name="tiny-base",
        )

        wrong_base = TinyWrapper(use_eoc=True, with_head=True)
        wrong_base.model_name = "other-base"
        with self.assertRaisesRegex(ValueError, "base_model mismatch"):
            load_checkpoint_into_model(wrong_base, payload)

        wrong_scale = TinyWrapper(use_eoc=True, with_head=True)
        wrong_scale.logit_bias_scale = 2.0
        with self.assertRaisesRegex(ValueError, "logit_bias_scale mismatch"):
            load_checkpoint_into_model(wrong_scale, payload)

    def test_standalone_lora_adapter_loads_on_its_base_model(self):
        source = tiny_peft_model()
        for name, parameter in source.named_parameters():
            if "lora_A" in name:
                parameter.data.fill_(0.125)
            elif "lora_B" in name:
                parameter.data.fill_(0.375)

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            source.save_pretrained(
                checkpoint_dir,
                save_embedding_layers=False,
            )
            loaded = load_lora_adapter(
                tiny_base_model(),
                checkpoint_dir,
                expected_base_model_name="tiny-base",
            )

        source_state = get_peft_model_state_dict(
            source,
            adapter_name="default",
            save_embedding_layers=False,
        )
        loaded_state = get_peft_model_state_dict(
            loaded,
            adapter_name="default",
            save_embedding_layers=False,
        )
        self.assertEqual(set(source_state), set(loaded_state))
        for name in source_state:
            torch.testing.assert_close(loaded_state[name], source_state[name])

    def test_unknown_or_incompatible_delta_fails_strictly(self):
        model = TinyWrapper()
        with self.assertRaisesRegex(ValueError, "Unknown checkpoint format"):
            load_checkpoint_into_model(model, {"checkpoint_format": "other"})

        payload = build_checkpoint_payload(
            model,
            1,
            model.tool_names,
            {},
            checkpoint_format="trainable_only",
            base_model_name="tiny-base",
        )
        payload["checkpoint_version"] = 999
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            load_checkpoint_into_model(model, payload)


if __name__ == "__main__":
    unittest.main()
