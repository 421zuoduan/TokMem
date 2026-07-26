from __future__ import annotations

import unittest

from compositional_intercode_bash.atomizer import (
    PIPE,
    PIPE_STDERR,
    RARE_UTILITY,
    START,
    IneligibleCommand,
    atomize_command,
    count_static_head_group_support,
)


class AtomizerTest(unittest.TestCase):
    def test_rarity_is_conditioned_on_connector_and_utility(self):
        support = count_static_head_group_support(
            [
                ("group-a", "gzip first"),
                ("group-b", "gzip second"),
            ]
        )
        self.assertEqual(support[(START, "gzip")], 2)
        parsed = atomize_command(
            "find . | gzip",
            head_group_support=support,
            minimum_head_group_support=2,
        )
        self.assertEqual(parsed.signatures[0], (START, RARE_UTILITY))
        self.assertEqual(parsed.signatures[1], (PIPE, RARE_UTILITY))

    def test_simple_and_pipeline_are_lossless(self):
        simple = atomize_command("  /bin/echo 'héllo'  # keep")
        self.assertEqual(simple.signatures, ((START, "echo"),))
        self.assertEqual("".join(simple.base_chunks), simple.command_raw)

        raw = "find . -type f  |&  sort | head -n 2 # tail"
        parsed = atomize_command(raw)
        self.assertEqual(
            parsed.signatures,
            ((START, "find"), (PIPE_STDERR, "sort"), (PIPE, "head")),
        )
        self.assertEqual("".join(parsed.base_chunks), raw)
        self.assertIn("|&", parsed.base_chunks[0])
        self.assertIn("# tail", parsed.base_chunks[-1])

    def test_nested_commands_are_not_promoted(self):
        parsed = atomize_command("cd $(which python | xargs dirname)")
        self.assertEqual(len(parsed.atoms), 1)
        self.assertEqual(parsed.atoms[0].static_utility_head, "cd")

        parsed = atomize_command(r"find . -exec echo {} \; | wc -l")
        self.assertEqual(
            [atom.static_utility_head for atom in parsed.atoms],
            ["find", "wc"],
        )

    def test_compound_and_multiple_roots_are_rejected(self):
        for raw in ("echo a && echo b", "for x in a; do echo $x; done", "echo a\necho b"):
            with self.subTest(raw=raw):
                with self.assertRaises(IneligibleCommand):
                    atomize_command(raw)


if __name__ == "__main__":
    unittest.main()
