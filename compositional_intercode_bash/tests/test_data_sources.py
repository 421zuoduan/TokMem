from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from compositional_intercode_bash.data_sources import (
    DEFAULT_CANONICAL_SPLIT_DIR,
    attach_split_pairs,
    load_canonical_nl2bash_splits,
    load_intercode_tasks,
    load_raw_nl2bash,
    validate_materialized_sources,
)
from compositional_intercode_bash.tests.helpers import write_valid_source_artifacts


ROOT = Path(__file__).resolve().parents[2]


class DataSourceTest(unittest.TestCase):
    def test_materialized_source_gate_rejects_truncated_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_valid_source_artifacts(root)
            integrity = validate_materialized_sources(
                root,
                require_canonical=True,
            )
            self.assertTrue(integrity["canonical_ready"])
            (root / "intercode_tasks.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly 200"):
                validate_materialized_sources(
                    root,
                    require_canonical=True,
                )

    def test_downloaded_source_counts(self):
        raw = load_raw_nl2bash(ROOT / "datasets" / "nl2bash" / "data" / "bash")
        tasks = load_intercode_tasks(ROOT / "datasets" / "intercode")
        self.assertEqual(len(raw), 12607)
        self.assertEqual(len(tasks), 200)
        counts = {}
        for task in tasks:
            counts[task["fs_id"]] = counts.get(task["fs_id"], 0) + 1
        self.assertEqual(counts, {"fs1": 60, "fs2": 53, "fs3": 60, "fs4": 27})

    @unittest.skipUnless(
        (DEFAULT_CANONICAL_SPLIT_DIR / "train.parquet").exists(),
        "Pinned NL2Bash Parquet files have not been downloaded",
    )
    def test_released_split_hashes_counts_and_raw_mapping(self):
        raw = load_raw_nl2bash(ROOT / "datasets" / "nl2bash" / "data" / "bash")
        split_pairs, manifest = load_canonical_nl2bash_splits(
            DEFAULT_CANONICAL_SPLIT_DIR
        )
        attached = attach_split_pairs(raw, split_pairs)
        counts = {}
        for record in attached:
            split = record["official_split"]
            counts[split] = counts.get(split, 0) + 1
        self.assertEqual(manifest["rows"], 9305)
        self.assertEqual(
            counts,
            {
                "TRAIN": 8090,
                "DEV": 609,
                "TEST": 606,
                "FILTERED_OUT": 3302,
            },
        )


if __name__ == "__main__":
    unittest.main()
