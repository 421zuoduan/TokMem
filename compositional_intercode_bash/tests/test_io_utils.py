from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from compositional_intercode_bash.io_utils import sha256_directory


class DirectoryHashTest(unittest.TestCase):
    def test_hash_covers_relative_names_and_file_contents(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "nested").mkdir()
            first = root / "tokenizer.json"
            second = root / "nested" / "config.json"
            first.write_bytes(b"tokenizer")
            second.write_bytes(b"config")
            original = sha256_directory(root)
            self.assertEqual(original, sha256_directory(root))

            second.write_bytes(b"changed")
            self.assertNotEqual(original, sha256_directory(root))
            second.write_bytes(b"config")
            second.rename(root / "nested" / "renamed.json")
            self.assertNotEqual(original, sha256_directory(root))


if __name__ == "__main__":
    unittest.main()
