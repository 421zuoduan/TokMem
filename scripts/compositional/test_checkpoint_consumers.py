#!/usr/bin/env python3
"""Path-resolution checks for checkpoint evaluation entrypoints."""

import unittest

from evaluate_lora_checkpoint import (
    final_round_spec,
    resolve_data_path as resolve_lora_data_path,
)
from generate_checkpoint_predictions import (
    resolve_data_path as resolve_tokmem_data_path,
    resolve_max_new_tokens,
)


class CheckpointConsumerTest(unittest.TestCase):
    def test_adaptation_defaults_to_the_final_round_test_split(self):
        run_config = {
            "args": {
                "data_dir": "/data/splits",
                "training_rounds": "1-50:1,51-100:3",
                "test_max_function_calls": 4,
                "test_max_function_calls_per_round": "4,10",
            },
            "rounds": [
                {"tools": "1-50", "epochs": 1},
                {"tools": "51-100", "epochs": 3},
            ],
        }

        self.assertEqual(final_round_spec(run_config), ("51-100", 10))
        expected = (
            "/data/splits/test/"
            "function_calling_test_tools51-100_10calls.json"
        )
        self.assertEqual(str(resolve_lora_data_path(run_config, None)), expected)
        self.assertEqual(str(resolve_tokmem_data_path(run_config, None)), expected)

    def test_single_round_legacy_config_keeps_the_existing_split(self):
        run_config = {
            "args": {
                "data_dir": "/data/splits",
                "training_rounds": "51-100:3",
                "test_max_function_calls": 4,
            }
        }

        self.assertEqual(final_round_spec(run_config), ("51-100", 4))
        expected = (
            "/data/splits/test/"
            "function_calling_test_tools51-100_4calls.json"
        )
        self.assertEqual(str(resolve_lora_data_path(run_config, None)), expected)
        self.assertEqual(str(resolve_tokmem_data_path(run_config, None)), expected)

    def test_prediction_length_reuses_run_config_with_legacy_fallback(self):
        self.assertEqual(
            resolve_max_new_tokens(
                {"args": {"max_new_tokens": 512}},
                None,
            ),
            512,
        )
        self.assertEqual(resolve_max_new_tokens({"args": {}}, None), 256)
        self.assertEqual(
            resolve_max_new_tokens(
                {"args": {"max_new_tokens": 512}},
                128,
            ),
            128,
        )


if __name__ == "__main__":
    unittest.main()
