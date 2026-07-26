from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from compositional_intercode_bash.base_text import (
    ARTIFACT_METADATA_FILENAME,
    BASE_TEXT_ARTIFACT_SCHEMA,
    BASE_TEXT_METHOD,
    TOOL_CATALOG_FILENAME,
    FrozenBaseTextAdapter,
    render_tool_catalog,
    save_base_text_artifact,
    validate_base_text_artifact,
)
from compositional_intercode_bash import evaluate as evaluate_module
from compositional_intercode_bash.intercode_runner import (
    INTERCODE_BASH_EPISODE_SCHEMA,
)
from compositional_intercode_bash.io_utils import (
    PACKAGE_ROOT,
    read_json,
    sha256_text,
    write_json,
    write_jsonl,
)
from compositional_intercode_bash.prepare_data import (
    PROVENANCE_ARTIFACTS,
    VIEW_INPUT_ARTIFACTS,
)
from compositional_intercode_bash.tests.helpers import ByteTokenizer
from compositional_intercode_bash.training_data import SYSTEM_PROMPT
from compositional_intercode_bash.unigram import ProcedureUnigramModel


class SavingByteTokenizer(ByteTokenizer):
    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text(
            json.dumps(self.get_vocab(), sort_keys=True),
            encoding="utf-8",
        )
        return (str(directory / "tokenizer.json"),)


class ScriptedCausalLM(nn.Module):
    def __init__(self, vocabulary_size: int, scripted_tokens: list[int]):
        super().__init__()
        self.marker = nn.Parameter(torch.ones(()))
        self.vocabulary_size = vocabulary_size
        self.scripted_tokens = list(scripted_tokens)
        self.calls = []

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        past_key_values=None,
        use_cache=False,
        return_dict=True,
    ):
        call_index = len(self.calls)
        self.calls.append(
            {
                "input_length": int(input_ids.shape[1]),
                "attention_length": int(attention_mask.shape[1]),
                "past": past_key_values,
            }
        )
        logits = torch.full(
            (1, input_ids.shape[1], self.vocabulary_size),
            -1000.0,
            device=input_ids.device,
        )
        logits[:, -1, self.scripted_tokens[call_index]] = 1000.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=("cache", call_index) if use_cache else None,
        )


class BaseTextTest(unittest.TestCase):
    def _model(self) -> ProcedureUnigramModel:
        return ProcedureUnigramModel(
            [
                (("START", "find"),),
                (("PIPE", "grep"),),
                (("START", "find"), ("PIPE", "grep")),
            ],
            [0.4, 0.35, 0.25],
            0.2,
            mandatory=[True, True, False],
        )

    @staticmethod
    def _atom_record(
        sample_id: str,
        split: str,
        signatures,
        raw_cores,
    ):
        return {
            "sample_id": sample_id,
            "source_group_id": f"group-{sample_id}",
            "derived_split": split,
            "canonical_signatures": [list(value) for value in signatures],
            "atoms": [
                {
                    "canonical_signature": list(signature),
                    "raw_core": raw_core,
                }
                for signature, raw_core in zip(signatures, raw_cores)
            ],
        }

    def _records(self):
        return [
            self._atom_record(
                "train-bare",
                "TRAIN",
                [("START", "find")],
                ["find"],
            ),
            self._atom_record(
                "train-find",
                "TRAIN",
                [("START", "find")],
                ["find ."],
            ),
            self._atom_record(
                "train-pipe",
                "TRAIN",
                [("START", "find"), ("PIPE", "grep")],
                ["find /tmp", "grep -i needle"],
            ),
            self._atom_record(
                "dev-secret",
                "DEV",
                [("START", "find")],
                ["find /INTERCODE_DEV_SECRET"],
            ),
            self._atom_record(
                "test-secret",
                "TEST",
                [("PIPE", "grep")],
                ["grep INTERCODE_TEST_SECRET"],
            ),
        ]

    def test_catalog_covers_all_k_and_examples_are_train_only(self):
        tokenizer = ByteTokenizer()
        first = render_tool_catalog(
            self._model(),
            self._records(),
            tokenizer,
            max_catalog_tokens=100000,
        )
        second = render_tool_catalog(
            self._model(),
            reversed(self._records()),
            tokenizer,
            max_catalog_tokens=100000,
        )
        self.assertEqual(first.text, second.text)
        self.assertEqual(first.report, second.report)
        self.assertEqual(first.report["procedure_count"], 3)
        self.assertEqual(first.report["structure_count"], 3)
        self.assertEqual(first.report["mandatory_singleton_count"], 2)
        self.assertEqual(first.report["example_count"], 2)
        self.assertIn("- find | grep", first.text)
        self.assertIn('"find ."', first.text)
        self.assertIn('"grep -i needle"', first.text)
        self.assertNotIn("INTERCODE_DEV_SECRET", first.text)
        self.assertNotIn("INTERCODE_TEST_SECRET", first.text)
        self.assertEqual(
            {record["sample_id"] for record in first.report["examples"]},
            {"train-find", "train-pipe"},
        )

    def test_catalog_token_limit_is_a_hard_failure(self):
        tokenizer = ByteTokenizer()
        catalog = render_tool_catalog(
            self._model(),
            self._records(),
            tokenizer,
            max_catalog_tokens=100000,
        )
        with self.assertRaisesRegex(ValueError, "retrieval is forbidden"):
            render_tool_catalog(
                self._model(),
                self._records(),
                tokenizer,
                max_catalog_tokens=catalog.report["catalog_token_count"] - 1,
            )

    def test_save_binds_formal_inputs_and_detects_catalog_tampering(self):
        tokenizer = SavingByteTokenizer()
        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            output_dir = root / "base"
            source_dir.mkdir()
            write_json(
                source_dir / "procedure_lexicon.json",
                self._model().to_dict(),
            )
            write_jsonl(source_dir / "atoms.jsonl", self._records())

            hashes = {
                name: {"sha256": f"{index + 1:064x}"}
                for index, name in enumerate(VIEW_INPUT_ARTIFACTS)
            }
            provenance_integrity = {
                **{
                    name: hashes[name]
                    for name in PROVENANCE_ARTIFACTS
                },
                "primary_ready": True,
                "primary_run": True,
                "deny_policy": "all_candidates",
            }
            procedure_integrity = {
                **hashes,
                "primary_ready": True,
                "primary_run": True,
                "deny_policy": "all_candidates",
                "procedure_count": self._model().size,
                "procedure_model_hash": self._model().model_hash(),
                "procedure_inventory_hash": self._model().inventory_hash(),
            }
            with (
                mock.patch(
                    "compositional_intercode_bash.base_text."
                    "load_provenance_integrity",
                    return_value=provenance_integrity,
                ) as provenance_loader,
                mock.patch(
                    "compositional_intercode_bash.base_text."
                    "load_procedure_integrity",
                    return_value=procedure_integrity,
                ) as procedure_loader,
            ):
                metadata = save_base_text_artifact(
                    source_dir,
                    output_dir,
                    tokenizer,
                    model_name="example/base-model",
                    model_revision="a" * 40,
                    max_catalog_tokens=100000,
                    require_primary=True,
                )

            provenance_loader.assert_called_once_with(
                source_dir,
                require_primary=True,
            )
            procedure_loader.assert_called_once_with(
                source_dir,
                provenance_integrity,
                require_primary=True,
            )
            self.assertEqual(metadata["schema"], BASE_TEXT_ARTIFACT_SCHEMA)
            self.assertEqual(metadata["method"], BASE_TEXT_METHOD)
            self.assertTrue(metadata["formal_ready"])
            self.assertEqual(metadata["trainable_parameter_count"], 0)
            self.assertEqual(
                set(metadata["artifact_integrity"]["input_sha256"]),
                set(VIEW_INPUT_ARTIFACTS),
            )
            self.assertEqual(
                read_json(output_dir / ARTIFACT_METADATA_FILENAME),
                metadata,
            )
            validate_base_text_artifact(output_dir, tokenizer=tokenizer)
            with (output_dir / TOOL_CATALOG_FILENAME).open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "catalog differs"):
                validate_base_text_artifact(output_dir, tokenizer=tokenizer)

    def test_frozen_adapter_uses_exact_greedy_multitoken_stop(self):
        base_model = ScriptedCausalLM(
            vocabulary_size=32,
            scripted_tokens=[7, 8, 9, 10],
        )
        adapter = FrozenBaseTextAdapter(base_model)
        self.assertFalse(adapter.training)
        self.assertFalse(base_model.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in adapter.parameters())
        )
        adapter.train(True)
        self.assertFalse(adapter.training)
        self.assertFalse(base_model.training)
        result = adapter.generate_tokens(
            torch.tensor([[1, 2]], dtype=torch.long),
            torch.ones((1, 2), dtype=torch.long),
            response_end_sequences=[[8, 9]],
            max_new_tokens=8,
        )
        self.assertEqual(result["generated_ids"], [7, 8, 9])
        self.assertTrue(result["terminated"])
        self.assertFalse(result["missing_terminator"])
        self.assertEqual(
            [call["input_length"] for call in base_model.calls],
            [2, 1, 1],
        )
        self.assertEqual(
            [call["attention_length"] for call in base_model.calls],
            [2, 3, 4],
        )
        self.assertIsNone(base_model.calls[0]["past"])
        self.assertIsNotNone(base_model.calls[1]["past"])
        self.assertEqual(adapter.procedure_token_ids_to_strip, ())
        self.assertIsNone(adapter.eoc_token_id_to_strip)

    def test_evaluator_dispatches_base_artifact_without_trainable_weights(self):
        metadata = {
            "schema": BASE_TEXT_ARTIFACT_SCHEMA,
            "method": BASE_TEXT_METHOD,
            "formal_ready": True,
            "base_model_name": "/fixed/base",
            "base_model_revision": None,
            "base_model_identity": {"identity": "fixed"},
            "catalog_token_count": 12,
        }
        task = {
            "task_id": "fs1:000",
            "fs_id": "fs1",
            "local_index": 0,
            "query": "task query",
            "source_sha256": "1" * 64,
        }
        episode = {
            "schema": INTERCODE_BASH_EPISODE_SCHEMA,
            "task_id": task["task_id"],
            "fs_id": task["fs_id"],
            "local_index": task["local_index"],
            "query": task["query"],
            "max_turns": 1,
            "max_reward": 0.01,
            "max_released_reward": 0.01,
            "released_success": False,
            "success": False,
            "turns_taken": 1,
            "termination_reason": "max_turns",
            "context_overflow": None,
            "reset_info": {},
            "turns": [
                {
                    "turn": 1,
                    "reward": 0.01,
                    "released_reward": 0.01,
                    "reward_info": {
                        "reward": {
                            "file_diff": 0.0,
                            "file_changes": 0.0,
                            "answer_similarity": 0.0,
                        },
                    },
                    "observation": "ok",
                    "observation_record": {
                        "raw_utf8_bytes": 2,
                        "raw_characters": 2,
                        "sha256": sha256_text("ok"),
                        "truncated": False,
                        "feedback_utf8_bytes": 2,
                        "kept_head_utf8_bytes": 2,
                        "kept_tail_utf8_bytes": 0,
                        "feedback": "ok",
                    },
                    "dropped_history_turns": 0,
                    "missing_terminator": False,
                    "memory_bank_constraint_trigger_count": 0,
                    "memory_bank_constraint_changed_token_count": 0,
                }
            ],
        }
        with tempfile.TemporaryDirectory(dir=PACKAGE_ROOT) as temporary:
            root = Path(temporary)
            artifact = root / "artifact"
            output = root / "evaluation"
            tokenizer_dir = artifact / "tokenizer"
            tokenizer_dir.mkdir(parents=True)
            write_json(artifact / "checkpoint.json", metadata)
            (artifact / TOOL_CATALOG_FILENAME).write_text(
                "fixed catalog\n",
                encoding="utf-8",
            )
            write_json(artifact / "tool_catalog.report.json", {})
            write_json(tokenizer_dir / "tokenizer.json", {})
            fake_model = nn.Linear(1, 1)
            fake_tokenizer = object()
            args = SimpleNamespace(
                checkpoint=str(artifact),
                intercode_root=str(root / "intercode"),
                output_dir=str(output),
                max_turns=1,
                max_new_tokens=16,
                context_limit=128,
                image_map=None,
                fs_id=None,
                limit=None,
                resume=False,
                device="cpu",
                dtype="float32",
                allow_download=False,
                allow_exploratory_checkpoint=False,
            )
            fake_environment = mock.MagicMock()
            fake_environment.__enter__.return_value = fake_environment
            fake_environment.__exit__.return_value = False
            with (
                mock.patch.object(
                    evaluate_module,
                    "load_base_text_artifact",
                    return_value=(
                        fake_model,
                        fake_tokenizer,
                        metadata,
                        "fixed catalog\n",
                    ),
                ),
                mock.patch.object(
                    evaluate_module,
                    "load_intercode_tasks",
                    return_value=[task],
                ),
                mock.patch.object(
                    evaluate_module,
                    "resolve_docker_images",
                    return_value={
                        "fs1": {
                            "image_id": "sha256:image-1",
                            "labels": {"fs_id": "fs1"},
                        }
                    },
                ),
                mock.patch.object(
                    evaluate_module,
                    "docker_runtime_identity",
                    return_value={"formal_isolation": True},
                ),
                mock.patch.object(
                    evaluate_module,
                    "_official_source_identity",
                    return_value={"identity": "official-source"},
                ),
                mock.patch.object(
                    evaluate_module,
                    "_runtime_identity",
                    return_value={"device_argument": "cpu"},
                ),
                mock.patch.object(
                    evaluate_module,
                    "ReusableOfficialBashEnv",
                    return_value=fake_environment,
                ) as environment_factory,
                mock.patch.object(
                    evaluate_module,
                    "run_try_again_episode",
                    return_value=dict(episode),
                ) as runner,
            ):
                summary = evaluate_module.evaluate(args)

        runner_prompt = runner.call_args.kwargs["system_prompt"]
        self.assertEqual(environment_factory.call_count, 1)
        fake_environment.__exit__.assert_called_once()
        self.assertEqual(runner_prompt.count(SYSTEM_PROMPT), 1)
        self.assertIn("fixed catalog", runner_prompt)
        self.assertEqual(summary["artifact_schema"], BASE_TEXT_ARTIFACT_SCHEMA)
        self.assertEqual(summary["catalog_token_count"], 12)
        self.assertNotIn("trainable_sha256", summary["checkpoint_hashes"])
        self.assertFalse(summary["evaluation_complete"])
        self.assertFalse(summary["paper_ready"])


if __name__ == "__main__":
    unittest.main()
