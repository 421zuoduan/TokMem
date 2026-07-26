from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from compositional_intercode_bash.rootless_helpers import idmap_helper
from compositional_intercode_bash.rootless_helpers.runc_singleuid import (
    _rewrite_devpts_gid,
)


class SingleUidHelperTest(unittest.TestCase):
    def test_idmap_installs_only_the_current_user_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc_dir = Path(temporary)
            (proc_dir / "uid_map").write_text("", encoding="ascii")
            real_path = Path

            def redirected_path(value):
                if str(value) == "/proc/123":
                    return proc_dir
                return real_path(value)

            with (
                mock.patch.object(idmap_helper, "Path", side_effect=redirected_path),
                mock.patch.object(idmap_helper.os, "getuid", return_value=1002),
                mock.patch.object(
                    idmap_helper,
                    "_read_process_owner",
                    return_value=1002,
                ),
                mock.patch.object(
                    idmap_helper,
                    "_configured_subid_triple",
                    return_value=[1, 231072, 65536],
                ),
            ):
                idmap_helper.write_single_id_map(
                    "uid",
                    ["123", "0", "1002", "1", "1", "231072", "65536"],
                )

            self.assertEqual(
                (proc_dir / "uid_map").read_text(encoding="ascii"),
                "0 1002 1\n",
            )

    def test_idmap_rejects_an_unexpected_extra_range(self):
        with (
            mock.patch.object(idmap_helper.os, "getuid", return_value=1002),
            mock.patch.object(
                idmap_helper,
                "_configured_subid_triple",
                return_value=[1, 231072, 65536],
            ),
            self.assertRaisesRegex(PermissionError, "unexpected subordinate-ID"),
        ):
            idmap_helper.write_single_id_map(
                "uid",
                ["123", "0", "1002", "1", "1", "999999", "65536"],
            )

    def test_runc_rewrites_only_the_expected_devpts_gid(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            config_path = bundle / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "mounts": [
                            {
                                "destination": "/dev/pts",
                                "type": "devpts",
                                "options": ["nosuid", "noexec", "gid=5"],
                            },
                            {
                                "destination": "/proc",
                                "type": "proc",
                                "options": ["nosuid"],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config_path.chmod(0o640)

            _rewrite_devpts_gid(bundle)

            rewritten = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                rewritten["mounts"][0]["options"],
                ["nosuid", "noexec", "gid=0"],
            )
            self.assertEqual(config_path.stat().st_mode & 0o777, 0o640)
            self.assertEqual(
                list(bundle.glob(".config.singleuid.*.tmp")),
                [],
            )

    def test_runc_rejects_an_unexpected_oci_spec(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            (bundle / "config.json").write_text(
                json.dumps({"mounts": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                _rewrite_devpts_gid(bundle)


if __name__ == "__main__":
    unittest.main()
