#!/usr/bin/env python3
"""Regression checks for backbone-specific conversation formatting."""

import unittest
from types import SimpleNamespace

import torch

from backbone_prompting import (
    format_system_user_assistant_prompt,
    format_user_assistant_prompt,
    qwen35_generation_token_ids,
    response_end_text,
    response_end_token_ids,
    uses_qwen35_prompting,
)
from model import FunctionCallingModel


class FakeTokenizer:
    eos_token_id = 46
    eos_token = "<|im_end|>"
    pad_token_id = 44

    def __init__(self):
        self.chat_template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_calls.append((messages, kwargs))
        roles = ",".join(message["role"] for message in messages)
        return f"qwen:{roles}:no-thinking"

    def __call__(self, text, add_special_tokens=False):
        if text == "<|eot_id|>":
            return {"input_ids": [9]}
        raise AssertionError(f"Unexpected tokenization request: {text}")


class CapturingCausalLM:
    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return kwargs["input_ids"]


class NativeGenerationHarness:
    generate_with_tool_prediction = FunctionCallingModel.generate_with_tool_prediction

    def __init__(self, qwen=False):
        self.use_logit_bias = False
        self.use_tool_head_replacement = False
        self.use_memory_bank_constraint = False
        self.memory_bank_probability_threshold = 0.5
        self.use_eoc = False
        self.model = CapturingCausalLM()
        if qwen:
            self._native_generation_pad_token_id = 44
            self._native_generation_eos_token_id = 46

    def _parse_generated_sequences(self, *_args):
        return []

    def eval(self):
        return self


class BackbonePromptingTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_qwen35_outer_and_text_configs_use_qwen_protocol(self):
        self.assertTrue(uses_qwen35_prompting(model_type="qwen3_5"))
        self.assertTrue(uses_qwen35_prompting(model_type="qwen3_5_text"))
        self.assertFalse(uses_qwen35_prompting(model_type="llama"))

    def test_legacy_user_prompt_is_byte_for_byte_unchanged(self):
        prompt = format_user_assistant_prompt(
            self.tokenizer,
            "hello",
            model_type="llama",
        )
        self.assertEqual(
            prompt,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            "hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        )

    def test_legacy_lora_prompt_keeps_trailing_newline(self):
        prompt = format_user_assistant_prompt(
            self.tokenizer,
            "hello",
            model_type="llama",
            legacy_assistant_newline=True,
        )
        self.assertTrue(prompt.endswith("<|end_header_id|>\n"))

    def test_qwen_user_prompt_uses_chat_template_without_thinking(self):
        prompt = format_user_assistant_prompt(
            self.tokenizer,
            "hello",
            model_type="qwen3_5_text",
        )
        self.assertEqual(prompt, "qwen:user:no-thinking")
        messages, kwargs = self.tokenizer.chat_template_calls[-1]
        self.assertEqual(messages, [{"role": "user", "content": "hello"}])
        self.assertFalse(kwargs["enable_thinking"])
        self.assertTrue(kwargs["add_generation_prompt"])
        self.assertFalse(kwargs["tokenize"])

    def test_qwen_icl_prompt_uses_system_and_user_messages(self):
        prompt = format_system_user_assistant_prompt(
            self.tokenizer,
            "system",
            "hello",
            model_type="qwen3_5",
            legacy_prompt="legacy",
        )
        self.assertEqual(prompt, "qwen:system,user:no-thinking")

    def test_response_end_tokens_are_backbone_specific(self):
        self.assertEqual(
            response_end_token_ids(self.tokenizer, model_type="llama"),
            [9],
        )
        self.assertEqual(
            response_end_token_ids(self.tokenizer, model_type="qwen3_5"),
            [46],
        )
        self.assertEqual(
            response_end_text(self.tokenizer, model_type="llama"),
            "<|eot_id|>",
        )
        self.assertEqual(
            response_end_text(self.tokenizer, model_type="qwen3_5"),
            "<|im_end|>",
        )

    def test_qwen_generation_uses_distinct_pad_and_eos_ids(self):
        self.assertIsNone(
            qwen35_generation_token_ids(
                self.tokenizer,
                model_type="llama",
            )
        )
        self.assertEqual(
            qwen35_generation_token_ids(
                self.tokenizer,
                model_type="qwen3_5_text",
            ),
            (44, 46),
        )

    def test_model_outer_config_takes_precedence(self):
        model = SimpleNamespace(
            full_config=SimpleNamespace(model_type="qwen3_5"),
            config=SimpleNamespace(model_type="qwen3_5_text"),
        )
        self.assertTrue(uses_qwen35_prompting(model=model))

    def test_native_generation_only_overrides_qwen_pad_and_eos(self):
        input_ids = torch.tensor([[1, 2]])
        attention_mask = torch.ones_like(input_ids)

        legacy = NativeGenerationHarness()
        legacy.generate_with_tool_prediction(
            input_ids,
            attention_mask,
            self.tokenizer,
        )
        self.assertEqual(legacy.model.generate_kwargs["pad_token_id"], 46)
        self.assertNotIn("eos_token_id", legacy.model.generate_kwargs)

        qwen = NativeGenerationHarness(qwen=True)
        qwen.generate_with_tool_prediction(
            input_ids,
            attention_mask,
            self.tokenizer,
        )
        self.assertEqual(qwen.model.generate_kwargs["pad_token_id"], 44)
        self.assertEqual(qwen.model.generate_kwargs["eos_token_id"], 46)


if __name__ == "__main__":
    unittest.main()
