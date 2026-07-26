from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from compositional_toolathlon.rootless_helpers.idmap_helper import (
    write_single_id_map,
)
from compositional_toolathlon.rootless_helpers.runc_singleuid import (
    _bundle_path,
    _rewrite_devpts_gid,
)


class RootlessHelperTests(unittest.TestCase):
    def test_bundle_path_supports_both_runc_forms(self):
        self.assertEqual(
            _bundle_path(["create", "--bundle", "/tmp/a", "container"]),
            Path("/tmp/a"),
        )
        self.assertEqual(
            _bundle_path(["create", "--bundle=/tmp/b", "container"]),
            Path("/tmp/b"),
        )
        self.assertIsNone(_bundle_path(["list"]))

    def test_rewrite_changes_only_devpts_gid(self):
        config = {
            "mounts": [
                {
                    "type": "devpts",
                    "options": ["nosuid", "noexec", "gid=5", "mode=620"],
                },
                {
                    "type": "tmpfs",
                    "options": ["gid=5"],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            config_path = bundle / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            _rewrite_devpts_gid(bundle)
            rewritten = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            rewritten["mounts"][0]["options"],
            ["nosuid", "noexec", "gid=0", "mode=620"],
        )
        self.assertEqual(rewritten["mounts"][1]["options"], ["gid=5"])

    def test_idmap_helper_rejects_unsupported_kind_before_proc_write(self):
        with self.assertRaisesRegex(ValueError, "unsupported ID-map kind"):
            write_single_id_map("other", ["123", "0", "1002", "1"])

    def test_idmap_helper_rejects_malformed_arguments_before_proc_write(self):
        with self.assertRaisesRegex(ValueError, "expected PID"):
            write_single_id_map("uid", ["123", "0"])


if __name__ == "__main__":
    unittest.main()
