from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from compositional_toolathlon.collect_llm_episode_batch import (
    episode_port,
    require_fresh_artifacts,
    safe_name,
    select_shard,
)


class CollectLlmEpisodeBatchTests(unittest.TestCase):
    def test_selects_stable_modulo_shard(self):
        tasks = [{"task_id": str(index)} for index in range(7)]
        selected = select_shard(tasks, shard_index=1, num_shards=3)
        self.assertEqual(
            [(index, task["task_id"]) for index, task in selected],
            [(1, "1"), (4, "4")],
        )

    def test_rejects_invalid_shard(self):
        with self.assertRaisesRegex(ValueError, "shard-index"):
            select_shard([], shard_index=2, num_shards=2)

    def test_sanitizes_task_id_for_artifact_paths(self):
        self.assertEqual(safe_name("task / one"), "task-one")

    def test_rejects_stale_episode_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            stale = Path(temporary) / "episode.jsonl"
            stale.write_text("old\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to reuse"):
                require_fresh_artifacts([stale])

    def test_allocates_disjoint_port_blocks(self):
        self.assertEqual(episode_port(8400, 0, 2), 8402)
        self.assertEqual(episode_port(8400, 1, 2), 8502)


if __name__ == "__main__":
    unittest.main()
