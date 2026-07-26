from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from compositional_toolathlon.rootless_helpers.normalize_oci_ownership import (
    NormalizationError,
    normalize_registry_cache,
)


def _gzip_bytes(payload: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=output,
        mtime=0,
    ) as archive:
        archive.write(payload)
    return output.getvalue()


def _layer(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> tuple[bytes, str]:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for member, payload in entries:
            archive.addfile(
                member,
                io.BytesIO(payload) if payload is not None else None,
            )
    uncompressed = output.getvalue()
    return _gzip_bytes(uncompressed), "sha256:" + hashlib.sha256(uncompressed).hexdigest()


def _regular(
    name: str,
    payload: bytes,
    *,
    uid: int = 0,
    gid: int = 0,
    pax_headers: dict[str, str] | None = None,
) -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mode = 0o640
    member.mtime = 123456789
    member.uid = uid
    member.gid = gid
    member.pax_headers = dict(pax_headers or {})
    return member, payload


class NormalizeOciOwnershipTests(unittest.TestCase):
    def _write_cache(
        self,
        root: Path,
        layers: list[tuple[bytes, str]],
    ) -> tuple[dict, dict]:
        blob_dir = root / "blobs" / "sha256"
        manifest_dir = root / "manifests"
        blob_dir.mkdir(parents=True)
        manifest_dir.mkdir()
        layer_descriptors = []
        diff_ids = []
        for payload, diff_id in layers:
            digest = hashlib.sha256(payload).hexdigest()
            (blob_dir / digest).write_bytes(payload)
            layer_descriptors.append(
                {
                    "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                    "digest": f"sha256:{digest}",
                    "size": len(payload),
                }
            )
            diff_ids.append(diff_id)

        config = {
            "architecture": "amd64",
            "os": "linux",
            "config": {"Cmd": ["/bin/sh"]},
            "rootfs": {"type": "layers", "diff_ids": diff_ids},
            "history": [],
        }
        config_payload = json.dumps(config, separators=(",", ":")).encode()
        config_digest = hashlib.sha256(config_payload).hexdigest()
        (blob_dir / config_digest).write_bytes(config_payload)
        manifest = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {
                "mediaType": "application/vnd.docker.container.image.v1+json",
                "digest": f"sha256:{config_digest}",
                "size": len(config_payload),
            },
            "layers": layer_descriptors,
        }
        (manifest_dir / "source").write_bytes(
            json.dumps(manifest, separators=(",", ":")).encode()
        )
        return manifest, config

    def test_normalizes_only_affected_layer_and_preserves_semantics(self):
        capability = "\x01\x00\x00\x02capability"
        owned_file = _regular(
            "long/" + "directory-" * 15 + "/owned.txt",
            b"payload",
            uid=1001,
            gid=117,
            pax_headers={"SCHILY.xattr.security.capability": capability},
        )
        hardlink = tarfile.TarInfo("owned-hardlink")
        hardlink.type = tarfile.LNKTYPE
        hardlink.linkname = owned_file[0].name
        hardlink.uid = 0
        hardlink.gid = 42
        hardlink.mtime = 123456789
        first_layer = _layer([owned_file, (hardlink, None)])
        second_layer = _layer([_regular("root-owned.txt", b"unchanged")])

        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            source_manifest, source_config = self._write_cache(
                cache_root,
                [first_layer, second_layer],
            )
            result = normalize_registry_cache(
                cache_root,
                "source",
                "singleuid",
                progress=None,
            )

            target_manifest = json.loads(
                (cache_root / "manifests" / "singleuid").read_bytes()
            )
            manifest_payload = (
                cache_root / "manifests" / "singleuid"
            ).read_bytes()
            manifest_digest = "sha256:" + hashlib.sha256(
                manifest_payload
            ).hexdigest()
            self.assertEqual(
                (
                    cache_root / "manifests" / manifest_digest
                ).read_bytes(),
                manifest_payload,
            )
            self.assertEqual(result["changed_layer_positions"], [1])
            self.assertEqual(result["changed_owner_entries"], 2)
            self.assertNotEqual(
                target_manifest["layers"][0]["digest"],
                source_manifest["layers"][0]["digest"],
            )
            self.assertEqual(
                target_manifest["layers"][1],
                source_manifest["layers"][1],
            )

            target_config_digest = target_manifest["config"]["digest"].split(":")[1]
            target_config = json.loads(
                (cache_root / "blobs" / "sha256" / target_config_digest).read_bytes()
            )
            self.assertNotEqual(
                target_config["rootfs"]["diff_ids"][0],
                source_config["rootfs"]["diff_ids"][0],
            )
            self.assertEqual(
                target_config["rootfs"]["diff_ids"][1],
                source_config["rootfs"]["diff_ids"][1],
            )

            target_layer_digest = target_manifest["layers"][0]["digest"].split(":")[1]
            target_layer = cache_root / "blobs" / "sha256" / target_layer_digest
            with tarfile.open(target_layer, mode="r:gz") as archive:
                members = archive.getmembers()
                self.assertTrue(all(member.uid == 0 for member in members))
                self.assertTrue(all(member.gid == 0 for member in members))
                extracted = archive.extractfile(members[0])
                self.assertIsNotNone(extracted)
                self.assertEqual(extracted.read(), b"payload")
                self.assertEqual(
                    members[0].pax_headers[
                        "SCHILY.xattr.security.capability"
                    ],
                    capability,
                )
                self.assertEqual(members[1].type, tarfile.LNKTYPE)
                self.assertEqual(members[1].linkname, members[0].name)

    def test_rejects_special_tar_member(self):
        device = tarfile.TarInfo("device")
        device.type = tarfile.CHRTYPE
        device.devmajor = 1
        device.devminor = 3
        device.uid = 0
        device.gid = 42
        layer = _layer([(device, None)])

        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            self._write_cache(cache_root, [layer])
            with self.assertRaisesRegex(
                NormalizationError,
                "unsupported tar member type",
            ):
                normalize_registry_cache(
                    cache_root,
                    "source",
                    "singleuid",
                    progress=None,
                )

    def test_rejects_corrupt_source_blob(self):
        layer = _layer([_regular("file", b"data", gid=42)])
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            manifest, _ = self._write_cache(cache_root, [layer])
            digest = manifest["layers"][0]["digest"].split(":")[1]
            with (cache_root / "blobs" / "sha256" / digest).open("ab") as output:
                output.write(b"corruption")
            with self.assertRaisesRegex(NormalizationError, "blob size mismatch"):
                normalize_registry_cache(
                    cache_root,
                    "source",
                    "singleuid",
                    progress=None,
                )

    def test_rejects_source_manifest_digest_mismatch(self):
        layer = _layer([_regular("file", b"data", gid=42)])
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            self._write_cache(cache_root, [layer])
            with self.assertRaisesRegex(
                NormalizationError,
                "source manifest digest mismatch",
            ):
                normalize_registry_cache(
                    cache_root,
                    "source",
                    "singleuid",
                    expected_source_manifest_digest="sha256:" + "0" * 64,
                    progress=None,
                )

    def test_rejects_uncompressed_diff_id_mismatch(self):
        layer = _layer([_regular("file", b"data", gid=42)])
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            manifest, _ = self._write_cache(cache_root, [layer])
            config_digest = manifest["config"]["digest"].split(":")[1]
            config_path = cache_root / "blobs" / "sha256" / config_digest
            config = json.loads(config_path.read_bytes())
            config["rootfs"]["diff_ids"][0] = "sha256:" + "0" * 64
            config_payload = json.dumps(config, separators=(",", ":")).encode()
            replacement_digest = hashlib.sha256(config_payload).hexdigest()
            (cache_root / "blobs" / "sha256" / replacement_digest).write_bytes(
                config_payload
            )
            manifest["config"]["digest"] = f"sha256:{replacement_digest}"
            manifest["config"]["size"] = len(config_payload)
            (cache_root / "manifests" / "source").write_bytes(
                json.dumps(manifest, separators=(",", ":")).encode()
            )
            with self.assertRaisesRegex(
                NormalizationError,
                "uncompressed digest mismatch",
            ):
                normalize_registry_cache(
                    cache_root,
                    "source",
                    "singleuid",
                    progress=None,
                )


if __name__ == "__main__":
    unittest.main()
