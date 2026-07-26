from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from compositional_toolathlon.model_runtime import (
    prepare_checkpoint_for_base_model,
    resolve_base_model_path,
)


class BaseModelPathTests(unittest.TestCase):
    def test_relative_checkpoint_path_is_resolved_against_repo_root(self):
        with tempfile.TemporaryDirectory() as repository_directory:
            repository_root = Path(repository_directory).resolve()
            relative_model = Path("models") / "tiny-model"
            expected_path = repository_root / relative_model
            expected_path.mkdir(parents=True)
            expected = str(expected_path)
            with tempfile.TemporaryDirectory() as unrelated_directory:
                with patch(
                    "compositional_toolathlon.model_runtime.REPO_ROOT",
                    repository_root,
                ):
                    previous_directory = Path.cwd()
                    try:
                        os.chdir(unrelated_directory)
                        actual = resolve_base_model_path(
                            run_config_model_name=expected,
                            checkpoint_model_name=str(relative_model),
                        )
                    finally:
                        os.chdir(previous_directory)
        self.assertEqual(actual, expected)

    def test_mismatched_checkpoint_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            configured = root / "models" / "configured"
            configured.mkdir(parents=True)
            other = root / "models" / "other"
            other.mkdir()
            with patch(
                "compositional_toolathlon.model_runtime.REPO_ROOT",
                root,
            ):
                with self.assertRaisesRegex(ValueError, "base model differ"):
                    resolve_base_model_path(
                        run_config_model_name=str(configured),
                        checkpoint_model_name="models/other",
                    )

    def test_run_config_requires_an_absolute_model_path(self):
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            resolve_base_model_path(
                run_config_model_name="models/Llama-3.1-8B-Instruct",
                checkpoint_model_name="models/Llama-3.1-8B-Instruct",
            )

    def test_missing_and_non_directory_model_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            missing = root / "missing-model"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                resolve_base_model_path(
                    run_config_model_name=str(missing),
                    checkpoint_model_name=str(missing),
                )

            model_file = root / "model-file"
            model_file.write_text("not a model directory", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be a directory"):
                resolve_base_model_path(
                    run_config_model_name=str(model_file),
                    checkpoint_model_name=str(model_file),
                )

    def test_prepare_checkpoint_canonicalizes_only_the_in_memory_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            model_path = root / "models" / "tiny-model"
            model_path.mkdir(parents=True)
            checkpoint = {
                "model_metadata": {
                    "base_model": "models/tiny-model",
                    "wrapper_class": "TinyWrapper",
                },
                "trainable_state": {"memory": "fixture"},
            }
            with patch(
                "compositional_toolathlon.model_runtime.REPO_ROOT",
                root,
            ):
                resolved, prepared = prepare_checkpoint_for_base_model(
                    checkpoint=checkpoint,
                    run_config_model_name=str(model_path),
                )
        self.assertEqual(resolved, str(model_path))
        self.assertEqual(
            prepared["model_metadata"]["base_model"],
            str(model_path),
        )
        self.assertEqual(
            checkpoint["model_metadata"]["base_model"],
            "models/tiny-model",
        )
        self.assertIsNot(prepared, checkpoint)
        self.assertIsNot(
            prepared["model_metadata"],
            checkpoint["model_metadata"],
        )
        self.assertIs(
            prepared["trainable_state"],
            checkpoint["trainable_state"],
        )


if __name__ == "__main__":
    unittest.main()
