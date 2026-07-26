from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from compositional_toolathlon.merge_jsonl import merge_jsonl


class MergeJsonlTests(unittest.TestCase):
    def test_merges_in_input_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_text('{"task_id":"a"}\n', encoding="utf-8")
            second.write_text('{"task_id":"b"}\n', encoding="utf-8")

            records = merge_jsonl([first, second], "task_id")

        self.assertEqual([record["task_id"] for record in records], ["a", "b"])

    def test_rejects_duplicate_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tasks.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps({"task_id": "same"})
                    for _ in range(2)
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate task_id"):
                merge_jsonl([path], "task_id")

    def test_excludes_named_key_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tasks.jsonl"
            path.write_text(
                '{"task_id":"keep"}\n{"task_id":"drop"}\n',
                encoding="utf-8",
            )
            records = merge_jsonl(
                [path],
                "task_id",
                {"drop"},
            )
        self.assertEqual([record["task_id"] for record in records], ["keep"])


if __name__ == "__main__":
    unittest.main()
