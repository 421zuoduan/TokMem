#!/usr/bin/env python3
"""CPU regression checks for backbone wrapper selection."""

import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn


COMPOSITIONAL_DIR = Path(__file__).resolve().parent
if str(COMPOSITIONAL_DIR) not in sys.path:
    sys.path.insert(0, str(COMPOSITIONAL_DIR))

from backbone_registry import resolve_function_calling_model_class  # noqa: E402
from checkpoint_io import (  # noqa: E402
    build_checkpoint_payload,
    load_checkpoint_into_model,
)
from model import FunctionCallingModel  # noqa: E402
from qwen35_model import (  # noqa: E402
    Qwen35FunctionCallingModel,
    resolve_qwen35_lora_targets,
)


class FakeQwen35TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])

        self.model.layers[0].linear_attn = nn.Module()
        self.model.layers[0].linear_attn.out_proj = nn.Linear(4, 4)

        self.model.layers[1].self_attn = nn.Module()
        self.model.layers[1].self_attn.o_proj = nn.Linear(4, 4)


class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 4
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.vocab = {
            "ordinary": 0,
            "<|reserved_special_token_0|>": 5,
            "<|reserved_special_token_1|>": 6,
            "<|reserved_special_token_2|>": 7,
        }

    def get_vocab(self):
        return dict(self.vocab)

    def __len__(self):
        return 10


class FakeCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3_5_text",
            hidden_size=4,
            num_hidden_layers=2,
        )
        self.embed_tokens = nn.Embedding(10, 4)
        self.lm_head = nn.Linear(4, 10, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, size):
        raise AssertionError(f"Unexpected embedding resize to {size}")

    def forward(
        self,
        input_ids,
        attention_mask,
        output_hidden_states=False,
        return_dict=True,
        **_kwargs,
    ):
        hidden_states = self.embed_tokens(input_ids)
        logits = self.lm_head(hidden_states)
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden_states,) if output_hidden_states else None,
            past_key_values=None,
        )


class BackboneRegistryTest(unittest.TestCase):
    def _resolve(self, model_type):
        config = SimpleNamespace(model_type=model_type)
        with patch(
            "backbone_registry.AutoConfig.from_pretrained",
            return_value=config,
        ):
            return resolve_function_calling_model_class("/unused/model")

    def test_existing_backbones_keep_the_original_wrapper(self):
        self.assertIs(self._resolve("llama"), FunctionCallingModel)
        self.assertIs(self._resolve("qwen2"), FunctionCallingModel)

    def test_qwen35_uses_the_dedicated_wrapper(self):
        self.assertIs(
            self._resolve("qwen3_5"),
            Qwen35FunctionCallingModel,
        )

    def test_qwen35_wrapper_preserves_the_existing_public_constructor(self):
        existing_parameters = list(
            inspect.signature(FunctionCallingModel).parameters
        )
        qwen35_parameters = list(
            inspect.signature(Qwen35FunctionCallingModel).parameters
        )
        self.assertEqual(qwen35_parameters, existing_parameters)

    def test_qwen35_reuses_existing_generation_and_parsing(self):
        self.assertIs(
            Qwen35FunctionCallingModel.generate_with_tool_prediction,
            FunctionCallingModel.generate_with_tool_prediction,
        )
        self.assertIs(
            Qwen35FunctionCallingModel._parse_generated_sequences,
            FunctionCallingModel._parse_generated_sequences,
        )

    def test_qwen35_constructor_uses_text_config_without_changing_interface(self):
        outer_config = SimpleNamespace(model_type="qwen3_5")
        fake_causal_lm = FakeCausalLM()
        with (
            patch(
                "qwen35_model.AutoConfig.from_pretrained",
                return_value=outer_config,
            ),
            patch(
                "qwen35_model.AutoModelForCausalLM.from_pretrained",
                return_value=fake_causal_lm,
            ),
        ):
            model = Qwen35FunctionCallingModel(
                model_name="/unused/qwen35",
                num_tools=2,
                tool_names=["tool_a", "tool_b"],
                tokenizer=FakeTokenizer(),
                device="cpu",
                dtype=torch.float32,
                use_eoc=True,
                use_logit_bias=True,
            )

        self.assertIs(model.config, fake_causal_lm.config)
        self.assertEqual(model.config.hidden_size, 4)
        self.assertEqual(model.tool_reserved_token_ids, [5, 6])
        self.assertEqual(model.eoc_token_id, 7)
        self.assertEqual(model._native_generation_pad_token_id, 0)
        self.assertEqual(model._native_generation_eos_token_id, 4)
        self.assertEqual(model.logit_bias_head.in_features, 4)

        logits = model(
            input_ids=torch.tensor([[0, 5, 6]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
        )
        self.assertEqual(tuple(logits.shape), (1, 3, 10))
        logits.sum().backward()
        self.assertIsNotNone(model.trainable_tool_embeddings.grad)

    def test_qwen35_trainable_checkpoint_round_trip(self):
        outer_config = SimpleNamespace(model_type="qwen3_5")

        def build_model():
            with (
                patch(
                    "qwen35_model.AutoConfig.from_pretrained",
                    return_value=outer_config,
                ),
                patch(
                    "qwen35_model.AutoModelForCausalLM.from_pretrained",
                    return_value=FakeCausalLM(),
                ),
            ):
                return Qwen35FunctionCallingModel(
                    model_name="/unused/qwen35",
                    num_tools=2,
                    tool_names=["tool_a", "tool_b"],
                    tokenizer=FakeTokenizer(),
                    device="cpu",
                    dtype=torch.float32,
                    use_eoc=True,
                    use_logit_bias=True,
                )

        source = build_model()
        source.trainable_tool_embeddings.data.copy_(
            torch.arange(12, dtype=torch.float32).reshape(3, 4)
        )
        source.logit_bias_head.weight.data.fill_(0.25)
        source.logit_bias_head.bias.data.fill_(0.5)
        checkpoint = build_checkpoint_payload(
            source,
            1,
            source.tool_names,
            {},
            checkpoint_format="trainable_only",
            base_model_name="/unused/qwen35",
        )

        target = build_model()
        base_before = target.model.embed_tokens.weight.detach().clone()
        load_checkpoint_into_model(target, checkpoint)

        torch.testing.assert_close(
            target.trainable_tool_embeddings,
            source.trainable_tool_embeddings,
        )
        torch.testing.assert_close(
            target.logit_bias_head.weight,
            source.logit_bias_head.weight,
        )
        torch.testing.assert_close(
            target.logit_bias_head.bias,
            source.logit_bias_head.bias,
        )
        torch.testing.assert_close(target.model.embed_tokens.weight, base_before)

    def test_qwen35_cached_generation_step_uses_hybrid_cache(self):
        try:
            from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        except ImportError as exc:
            self.skipTest(f"Installed Transformers does not support Qwen3.5: {exc}")

        text_config = Qwen3_5TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            layer_types=["linear_attention", "full_attention"],
            linear_conv_kernel_dim=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            max_position_embeddings=128,
            eos_token_id=4,
            pad_token_id=0,
        )
        tiny_causal_lm = Qwen3_5ForCausalLM(text_config)
        outer_config = SimpleNamespace(model_type="qwen3_5")
        with (
            patch(
                "qwen35_model.AutoConfig.from_pretrained",
                return_value=outer_config,
            ),
            patch(
                "qwen35_model.AutoModelForCausalLM.from_pretrained",
                return_value=tiny_causal_lm,
            ),
        ):
            model = Qwen35FunctionCallingModel(
                model_name="/unused/qwen35",
                num_tools=2,
                tool_names=["tool_a", "tool_b"],
                tokenizer=FakeTokenizer(),
                device="cpu",
                dtype=torch.float32,
                use_eoc=True,
                use_logit_bias=True,
            )

        first_ids = torch.tensor([[0, 5, 6]])
        first_mask = torch.ones_like(first_ids)
        first_logits, first_hidden, cache = model._generation_forward_step(
            input_ids=first_ids,
            attention_mask=first_mask,
            return_last_hidden_state=True,
        )
        self.assertEqual(tuple(first_logits.shape), (1, 32))
        self.assertEqual(tuple(first_hidden.shape), (1, 16))
        self.assertEqual(cache.get_seq_length(), 3)

        second_logits, second_hidden, cache = model._generation_forward_step(
            input_ids=torch.tensor([[1]]),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            past_key_values=cache,
            return_last_hidden_state=True,
        )
        self.assertEqual(tuple(second_logits.shape), (1, 32))
        self.assertEqual(tuple(second_hidden.shape), (1, 16))
        self.assertEqual(cache.get_seq_length(), 4)


class Qwen35LoraTargetTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeQwen35TextModel()
        self.config = SimpleNamespace(num_hidden_layers=2)

    def test_hybrid_layer_targets_are_resolved_by_actual_module_name(self):
        targets, layers, unmatched = resolve_qwen35_lora_targets(
            self.model,
            self.config,
            layer_indices=[0, 1],
            base_modules=["o_proj", "out_proj"],
        )

        self.assertEqual(layers, [0, 1])
        self.assertEqual(unmatched, [])
        self.assertEqual(
            targets,
            [
                "model.layers.0.linear_attn.out_proj",
                "model.layers.1.self_attn.o_proj",
            ],
        )

    def test_negative_layer_index_is_supported(self):
        targets, layers, unmatched = resolve_qwen35_lora_targets(
            self.model,
            self.config,
            layer_indices=[-1],
            base_modules=["o_proj"],
        )

        self.assertEqual(layers, [1])
        self.assertEqual(unmatched, [])
        self.assertEqual(
            targets,
            ["model.layers.1.self_attn.o_proj"],
        )

    def test_missing_selected_modules_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "No Qwen3.5 LoRA modules matched"):
            resolve_qwen35_lora_targets(
                self.model,
                self.config,
                layer_indices=[0],
                base_modules=["q_proj"],
            )


if __name__ == "__main__":
    unittest.main()
