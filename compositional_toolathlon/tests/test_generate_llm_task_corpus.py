import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from compositional_toolathlon.generate_llm_task_corpus import (
    build_parser,
    generate_task_corpus,
)


def _manifest() -> dict:
    return {
        "tools": [
            {
                "stable_id": "tool::alpha",
                "tool_name": "alpha",
            },
            {
                "stable_id": "tool::beta",
                "tool_name": "beta",
            },
        ]
    }


class TaskCorpusSchedulerTests(unittest.TestCase):
    def test_selected_groups_are_generated_and_merged_in_config_order(self):
        groups = [
            {
                "group_id": "first",
                "task_family": "family-first",
                "must_require_tool_names": ["alpha"],
            },
            {
                "group_id": "skipped",
                "task_family": "family-skipped",
                "must_require_tool_names": ["alpha"],
            },
            {
                "group_id": "third",
                "task_family": "family-third",
                "must_require_tool_names": ["beta"],
            },
        ]

        async def fake_generate_candidates(**kwargs):
            return [
                {
                    "task_id": f"{kwargs['task_family']}-{session_index}",
                    "generation_provenance": {
                        "session_id": f"session-{session_index}"
                    },
                }
                for session_index in range(3)
            ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            with patch(
                "compositional_toolathlon.generate_llm_task_corpus.generate_candidates",
                side_effect=fake_generate_candidates,
            ) as generate:
                candidates = asyncio.run(
                    generate_task_corpus(
                        client=object(),
                        manifest=_manifest(),
                        config={"generator_sessions": [{}, {}, {}]},
                        coverage_groups=groups,
                        output_root=output_root,
                        group_ids=["third", "first"],
                        base_seed=100,
                        split="train",
                    )
                )

            self.assertEqual(generate.call_count, 2)
            first_call = generate.call_args_list[0].kwargs
            third_call = generate.call_args_list[1].kwargs
            self.assertEqual(first_call["count_per_session"], 1)
            self.assertEqual(first_call["base_seed"], 100)
            self.assertEqual(first_call["must_require_tool_ids"], ["tool::alpha"])
            self.assertEqual(third_call["base_seed"], 102)
            self.assertEqual(third_call["must_require_tool_ids"], ["tool::beta"])

            candidates_dir = output_root / "tasks" / "candidates"
            self.assertTrue((candidates_dir / "first.jsonl").is_file())
            self.assertFalse((candidates_dir / "skipped.jsonl").exists())
            self.assertTrue((candidates_dir / "third.jsonl").is_file())

            with (candidates_dir / "all.jsonl").open(
                "r", encoding="utf-8"
            ) as handle:
                merged = [json.loads(line) for line in handle]
            self.assertEqual(merged, candidates)
            self.assertEqual(
                [
                    candidate["generation_provenance"]["coverage_group_id"]
                    for candidate in merged
                ],
                ["first", "first", "first", "third", "third", "third"],
            )

    def test_cli_accepts_repeated_group_ids(self):
        args = build_parser().parse_args(
            ["--group-id", "first", "--group-id", "third"]
        )
        self.assertEqual(args.group_id, ["first", "third"])


if __name__ == "__main__":
    unittest.main()
