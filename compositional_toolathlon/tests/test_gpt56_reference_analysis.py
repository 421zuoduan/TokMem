from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from compositional_toolathlon.analyze_gpt56_reference import (
    aggregate,
    analyze_rollout,
)


class Gpt56ReferenceAnalysisTests(unittest.TestCase):
    def test_counts_reads_writes_verification_and_changed_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpt5.6_rollout.json"
            path.write_text(
                json.dumps(
                    {
                        "rollout": {
                            "task_dir": "finalpool/example",
                            "prompt_variant": "state-first",
                            "termination_reason": "claim_done",
                            "events": [
                                {
                                    "wire_name": "filesystem_list_directory",
                                    "arguments": {"path": "."},
                                    "success": True,
                                },
                                {
                                    "wire_name": "filesystem_write_file",
                                    "arguments": {"path": "out.txt"},
                                    "success": False,
                                },
                                {
                                    "wire_name": "filesystem_write_file",
                                    "arguments": {
                                        "path": "out.txt",
                                        "content": "done",
                                    },
                                    "success": True,
                                },
                                {
                                    "wire_name": "filesystem_read_text_file",
                                    "arguments": {"path": "out.txt"},
                                    "success": True,
                                },
                                {
                                    "wire_name": None,
                                    "arguments": {},
                                    "success": True,
                                },
                            ],
                        },
                        "failure": None,
                    }
                ),
                encoding="utf-8",
            )
            record = analyze_rollout(path)

        self.assertEqual(record["pre_write_read_count"], 1)
        self.assertEqual(record["post_write_read_count"], 1)
        self.assertEqual(record["changed_retry_count"], 1)
        self.assertEqual(record["unchanged_retry_count"], 0)
        self.assertTrue(record["claimed_done"])

    def test_failed_rollout_is_not_aggregated_as_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpt5.6_rollout.json"
            path.write_text(
                json.dumps(
                    {
                        "rollout": None,
                        "failure": {
                            "error_type": "RuntimeError",
                            "error": "unavailable",
                        },
                    }
                ),
                encoding="utf-8",
            )
            record = analyze_rollout(path)

        summary = aggregate([record])
        self.assertFalse(record["usable"])
        self.assertEqual(summary["usable_trajectories"], 0)
        self.assertEqual(summary["failed_trajectories"], 1)


if __name__ == "__main__":
    unittest.main()
