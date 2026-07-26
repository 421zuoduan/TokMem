"""Build a single-UID-compatible image manifest from a registry cache.

The no-sudo runtime maps only container UID/GID 0.  A normal Docker image can
therefore fail while unpacking a layer that contains any other numeric owner.
This module rewrites only the numeric UID/GID fields of affected tar members,
verifies that every other relevant member property and file byte is unchanged,
and publishes new layer/config/manifest blobs into the same local registry
cache.

The resulting image is deliberately a derived compatibility image, not the
byte-identical upstream image.  It is suitable only after the caller accepts
the loss of multi-user ownership semantics.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import os
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


_DIGEST_PATTERN = re.compile(r"sha256:([0-9a-f]{64})\Z")
_REFERENCE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
OFFICIAL_TOOLATHLON_MANIFEST_DIGEST = (
    "sha256:4d04fe4e0a6fdb4946f51bb05120cb44a0eef980231c11252f93b62897afcb9f"
)
OFFICIAL_TOOLATHLON_CONFIG_DIGEST = (
    "sha256:e2a967606d6e99522a9c8ebc785cd40273235a89836438101cb04283024cd181"
)
OFFICIAL_TOOLATHLON_CHANGED_LAYER_POSITIONS = [1, 2, 3, 4, 5, 6, 9]
OFFICIAL_TOOLATHLON_CHANGED_OWNER_ENTRIES = 37
NORMALIZER_VERSION = "raw-header-v1"
_SUPPORTED_MEMBER_TYPES = {
    tarfile.REGTYPE,
    tarfile.AREGTYPE,
    tarfile.DIRTYPE,
    tarfile.SYMTYPE,
    tarfile.LNKTYPE,
}
_RAW_METADATA_TYPES = {tarfile.XHDTYPE}
_ALLOWED_PAX_KEYS = {
    "path",
    "linkpath",
    "SCHILY.xattr.security.capability",
}
_TAR_BLOCK_SIZE = 512
_ZERO_TAR_BLOCK = b"\0" * _TAR_BLOCK_SIZE
_OWNERSHIP_PAX_KEYS = {
    "uid",
    "gid",
    "uname",
    "gname",
    "SCHILY.uid",
    "SCHILY.gid",
}
_REPRESENTATION_PAX_KEYS = {"path", "linkpath"}


class NormalizationError(RuntimeError):
    """Raised when an input or transformed image violates a safety invariant."""


@dataclass(frozen=True)
class LayerScan:
    entries: int
    nonzero_owners: int


@dataclass(frozen=True)
class NormalizedLayer:
    digest: str
    size: int
    diff_id: str
    entries: int
    changed_owners: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_gzip_payload(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with gzip.open(path, mode="rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except (OSError, EOFError) as error:
        raise NormalizationError(
            f"cannot hash uncompressed layer {path}: {error}"
        ) from error
    return digest.hexdigest()


def _descriptor_digest(descriptor: dict[str, Any]) -> str:
    declared = descriptor.get("digest")
    if not isinstance(declared, str):
        raise NormalizationError("descriptor is missing a string digest")
    match = _DIGEST_PATTERN.fullmatch(declared)
    if match is None:
        raise NormalizationError(f"unsupported descriptor digest: {declared!r}")
    return match.group(1)


def _validate_blob(cache_root: Path, descriptor: dict[str, Any]) -> Path:
    digest = _descriptor_digest(descriptor)
    path = cache_root / "blobs" / "sha256" / digest
    if not path.is_file():
        raise NormalizationError(f"missing blob: {path}")
    declared_size = descriptor.get("size")
    if not isinstance(declared_size, int) or declared_size < 0:
        raise NormalizationError(f"invalid blob size for sha256:{digest}")
    actual_size = path.stat().st_size
    if actual_size != declared_size:
        raise NormalizationError(
            f"blob size mismatch for sha256:{digest}: "
            f"{actual_size} != {declared_size}"
        )
    actual_digest = _sha256_file(path)
    if actual_digest != digest:
        raise NormalizationError(
            f"blob digest mismatch for sha256:{digest}: sha256:{actual_digest}"
        )
    return path


def _validate_member(member: tarfile.TarInfo, layer_path: Path) -> None:
    if member.type not in _SUPPORTED_MEMBER_TYPES:
        raise NormalizationError(
            f"unsupported tar member type {member.type!r} for "
            f"{member.name!r} in {layer_path}"
        )
    if member.sparse:
        raise NormalizationError(
            f"sparse tar member is unsupported: {member.name!r} in {layer_path}"
        )
    if any(key.startswith("GNU.sparse") for key in member.pax_headers):
        raise NormalizationError(
            f"GNU sparse PAX metadata is unsupported: "
            f"{member.name!r} in {layer_path}"
        )


def scan_layer(layer_path: Path) -> LayerScan:
    entries = 0
    nonzero_owners = 0
    try:
        with tarfile.open(layer_path, mode="r|gz") as archive:
            for member in archive:
                _validate_member(member, layer_path)
                entries += 1
                nonzero_owners += bool(member.uid or member.gid)
    except (OSError, tarfile.TarError) as error:
        raise NormalizationError(f"cannot scan layer {layer_path}: {error}") from error
    return LayerScan(entries=entries, nonzero_owners=nonzero_owners)


def _normalized_pax_headers(member: tarfile.TarInfo) -> dict[str, str]:
    headers = dict(member.pax_headers)
    for key in _OWNERSHIP_PAX_KEYS:
        headers.pop(key, None)
    return headers


def _semantic_pax_headers(member: tarfile.TarInfo) -> dict[str, str]:
    headers = _normalized_pax_headers(member)
    for key in _REPRESENTATION_PAX_KEYS:
        headers.pop(key, None)
    return headers


def _parse_tar_number(field: bytes, label: str) -> int:
    if field and field[0] & 0x80:
        raise NormalizationError(f"base-256 tar {label} is unsupported")
    value = field.rstrip(b"\0 ").lstrip(b" ")
    if not value:
        return 0
    if any(byte not in b"01234567" for byte in value):
        raise NormalizationError(f"invalid octal tar {label}: {field!r}")
    return int(value, 8)


def _tar_checksum(header: bytes) -> int:
    return sum(header[:148]) + (8 * ord(" ")) + sum(header[156:])


def _validate_raw_header(header: bytes, source_path: Path) -> None:
    if len(header) != _TAR_BLOCK_SIZE:
        raise NormalizationError(f"short tar header in {source_path}")
    stored = _parse_tar_number(header[148:156], "checksum")
    calculated = _tar_checksum(header)
    if stored != calculated:
        raise NormalizationError(
            f"tar checksum mismatch in {source_path}: {stored} != {calculated}"
        )


def _parse_pax_keys(payload: bytes, source_path: Path) -> set[str]:
    keys: set[str] = set()
    offset = 0
    while offset < len(payload):
        separator = payload.find(b" ", offset)
        if separator < 0:
            raise NormalizationError(f"invalid PAX record length in {source_path}")
        try:
            record_size = int(payload[offset:separator])
        except ValueError as error:
            raise NormalizationError(
                f"invalid PAX record length in {source_path}"
            ) from error
        if record_size <= 0 or offset + record_size > len(payload):
            raise NormalizationError(f"invalid PAX record bounds in {source_path}")
        record = payload[separator + 1 : offset + record_size]
        if not record.endswith(b"\n") or b"=" not in record:
            raise NormalizationError(f"invalid PAX record in {source_path}")
        key, _ = record[:-1].split(b"=", 1)
        try:
            keys.add(key.decode("ascii"))
        except UnicodeDecodeError as error:
            raise NormalizationError(
                f"non-ASCII PAX key in {source_path}"
            ) from error
        offset += record_size
    return keys


def _read_exact(source: Any, size: int, label: str) -> bytes:
    payload = source.read(size)
    if len(payload) != size:
        raise NormalizationError(f"short read while copying {label}")
    return payload


def _copy_exact(source: Any, target: Any, size: int, label: str) -> None:
    remaining = size
    while remaining:
        chunk = _read_exact(source, min(1024 * 1024, remaining), label)
        target.write(chunk)
        remaining -= len(chunk)


def _patch_owner_header(header: bytes, uid: int, gid: int) -> bytes:
    if uid == 0 and gid == 0:
        return header
    patched = bytearray(header)
    if uid:
        patched[108:116] = b"0000000\0"
    if gid:
        patched[116:124] = b"0000000\0"
    patched[148:156] = b"        "
    checksum = _tar_checksum(patched)
    encoded_checksum = f"{checksum:06o}\0 ".encode("ascii")
    if len(encoded_checksum) != 8:
        raise NormalizationError(f"tar checksum overflow: {checksum}")
    patched[148:156] = encoded_checksum
    return bytes(patched)


def _rewrite_layer_to_tar(source_path: Path, target_tar: Path) -> LayerScan:
    entries = 0
    changed_owners = 0
    try:
        with gzip.open(source_path, mode="rb") as source:
            with target_tar.open("wb") as target:
                while True:
                    header = source.read(_TAR_BLOCK_SIZE)
                    if not header:
                        raise NormalizationError(
                            f"tar end-of-archive block is missing in {source_path}"
                        )
                    if header == _ZERO_TAR_BLOCK:
                        target.write(header)
                        shutil.copyfileobj(
                            source,
                            target,
                            length=1024 * 1024,
                        )
                        break
                    _validate_raw_header(header, source_path)
                    member_type = header[156:157]
                    member_size = _parse_tar_number(
                        header[124:136],
                        "member size",
                    )
                    padded_size = (
                        (member_size + _TAR_BLOCK_SIZE - 1)
                        // _TAR_BLOCK_SIZE
                        * _TAR_BLOCK_SIZE
                    )

                    if member_type in _RAW_METADATA_TYPES:
                        target.write(header)
                        payload = _read_exact(
                            source,
                            padded_size,
                            f"PAX payload in {source_path}",
                        )
                        pax_keys = _parse_pax_keys(
                            payload[:member_size],
                            source_path,
                        )
                        unexpected = pax_keys - _ALLOWED_PAX_KEYS
                        if unexpected:
                            raise NormalizationError(
                                f"unsupported PAX keys in {source_path}: "
                                f"{sorted(unexpected)}"
                            )
                        target.write(payload)
                        continue

                    if member_type not in _SUPPORTED_MEMBER_TYPES:
                        raise NormalizationError(
                            f"unsupported raw tar member type "
                            f"{member_type!r} in {source_path}"
                        )
                    uid = _parse_tar_number(header[108:116], "uid")
                    gid = _parse_tar_number(header[116:124], "gid")
                    entries += 1
                    changed_owners += bool(uid or gid)
                    target.write(_patch_owner_header(header, uid, gid))
                    _copy_exact(
                        source,
                        target,
                        padded_size,
                        f"member payload in {source_path}",
                    )
    except (OSError, tarfile.TarError) as error:
        raise NormalizationError(
            f"cannot rewrite layer {source_path}: {error}"
        ) from error
    return LayerScan(entries=entries, nonzero_owners=changed_owners)


def _member_semantics(member: tarfile.TarInfo) -> tuple[Any, ...]:
    return (
        member.name,
        member.type,
        member.linkname,
        member.mode,
        member.size,
        member.mtime,
        member.uname,
        member.gname,
        member.devmajor,
        member.devminor,
        _semantic_pax_headers(member),
    )


def _compare_regular_contents(
    source_archive: tarfile.TarFile,
    target_archive: tarfile.TarFile,
    source_member: tarfile.TarInfo,
    target_member: tarfile.TarInfo,
) -> None:
    source_file = source_archive.extractfile(source_member)
    target_file = target_archive.extractfile(target_member)
    if source_file is None or target_file is None:
        raise NormalizationError(
            f"regular member has no verification stream: {source_member.name!r}"
        )
    with source_file, target_file:
        while True:
            source_chunk = source_file.read(1024 * 1024)
            target_chunk = target_file.read(1024 * 1024)
            if source_chunk != target_chunk:
                raise NormalizationError(
                    f"file content changed during normalization: "
                    f"{source_member.name!r}"
                )
            if not source_chunk:
                return


def verify_normalized_layer(source_path: Path, target_path: Path) -> LayerScan:
    entries = 0
    changed_owners = 0
    sentinel = object()
    try:
        with tarfile.open(source_path, mode="r|gz") as source_archive:
            with tarfile.open(target_path, mode="r|gz") as target_archive:
                pairs = itertools.zip_longest(
                    source_archive,
                    target_archive,
                    fillvalue=sentinel,
                )
                for source_member, target_member in pairs:
                    if source_member is sentinel or target_member is sentinel:
                        raise NormalizationError(
                            "tar entry count changed during normalization"
                        )
                    _validate_member(source_member, source_path)
                    _validate_member(target_member, target_path)
                    entries += 1
                    changed_owners += bool(source_member.uid or source_member.gid)
                    if target_member.uid != 0 or target_member.gid != 0:
                        raise NormalizationError(
                            f"nonzero owner remains after normalization: "
                            f"{target_member.name!r}"
                        )
                    if _member_semantics(source_member) != _member_semantics(
                        target_member
                    ):
                        raise NormalizationError(
                            f"tar metadata changed beyond ownership: "
                            f"{source_member.name!r}"
                        )
                    if source_member.isreg():
                        _compare_regular_contents(
                            source_archive,
                            target_archive,
                            source_member,
                            target_member,
                        )
    except (OSError, tarfile.TarError) as error:
        raise NormalizationError(
            f"cannot verify normalized layer {target_path}: {error}"
        ) from error
    return LayerScan(entries=entries, nonzero_owners=changed_owners)


def _gzip_tar(source_tar: Path, target_gzip: Path) -> None:
    with source_tar.open("rb") as source:
        with target_gzip.open("wb") as raw_target:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=6,
                fileobj=raw_target,
                mtime=0,
            ) as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)


def _publish_blob(cache_root: Path, temporary_blob: Path, digest: str) -> Path:
    target = cache_root / "blobs" / "sha256" / digest
    if target.exists():
        if not target.is_file() or _sha256_file(target) != digest:
            raise NormalizationError(f"conflicting target blob: {target}")
        temporary_blob.unlink()
        return target
    os.replace(temporary_blob, target)
    return target


def normalize_layer(
    cache_root: Path,
    source_path: Path,
    temporary_root: Path,
) -> NormalizedLayer:
    uncompressed_tar = temporary_root / "normalized-layer.tar"
    compressed_layer = temporary_root / "normalized-layer.tar.gz"
    rewrite_scan = _rewrite_layer_to_tar(source_path, uncompressed_tar)
    diff_digest = _sha256_file(uncompressed_tar)
    _gzip_tar(uncompressed_tar, compressed_layer)
    compressed_digest = _sha256_file(compressed_layer)
    published = _publish_blob(
        cache_root,
        compressed_layer,
        compressed_digest,
    )
    verification = verify_normalized_layer(source_path, published)
    if verification != rewrite_scan:
        raise NormalizationError(
            f"layer verification counts differ: "
            f"{verification!r} != {rewrite_scan!r}"
        )
    return NormalizedLayer(
        digest=f"sha256:{compressed_digest}",
        size=published.stat().st_size,
        diff_id=f"sha256:{diff_digest}",
        entries=verification.entries,
        changed_owners=verification.nonzero_owners,
    )


def _publish_bytes(cache_root: Path, payload: bytes) -> tuple[str, int]:
    digest = hashlib.sha256(payload).hexdigest()
    temporary_root = cache_root / "tmp"
    temporary_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=temporary_root,
        prefix="blob-",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    _publish_blob(cache_root, temporary_path, digest)
    return f"sha256:{digest}", len(payload)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _publish_manifest_path(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise NormalizationError(f"conflicting target manifest: {path}")
        return
    temporary_manifest = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_manifest.write_bytes(payload)
    os.replace(temporary_manifest, path)


def _load_manifest(
    cache_root: Path,
    reference: str,
    expected_digest: str | None,
) -> dict[str, Any]:
    if _REFERENCE_PATTERN.fullmatch(reference) is None:
        raise NormalizationError(f"unsafe manifest reference: {reference!r}")
    path = cache_root / "manifests" / reference
    if not path.is_file():
        raise NormalizationError(f"missing manifest: {path}")
    try:
        payload = path.read_bytes()
        manifest = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise NormalizationError(f"invalid manifest {path}: {error}") from error
    actual_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    if expected_digest is not None:
        if _DIGEST_PATTERN.fullmatch(expected_digest) is None:
            raise NormalizationError(
                f"invalid expected source manifest digest: {expected_digest!r}"
            )
        if actual_digest != expected_digest:
            raise NormalizationError(
                f"source manifest digest mismatch: "
                f"{actual_digest} != {expected_digest}"
            )
    if manifest.get("schemaVersion") != 2:
        raise NormalizationError("only Docker/OCI schemaVersion 2 is supported")
    if not isinstance(manifest.get("config"), dict):
        raise NormalizationError("manifest config descriptor is missing")
    if not isinstance(manifest.get("layers"), list):
        raise NormalizationError("manifest layers are missing")
    return manifest


def normalize_registry_cache(
    cache_root: Path,
    source_reference: str,
    target_reference: str,
    *,
    expected_source_manifest_digest: str | None = None,
    expected_source_config_digest: str | None = None,
    expected_changed_layer_positions: list[int] | None = None,
    expected_changed_owner_entries: int | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    cache_root = cache_root.resolve(strict=True)
    if source_reference == target_reference:
        raise NormalizationError("source and target references must differ")
    if _REFERENCE_PATTERN.fullmatch(target_reference) is None:
        raise NormalizationError(
            f"unsafe target manifest reference: {target_reference!r}"
        )

    manifest = _load_manifest(
        cache_root,
        source_reference,
        expected_source_manifest_digest,
    )
    source_config_digest = manifest["config"].get("digest")
    if (
        expected_source_config_digest is not None
        and source_config_digest != expected_source_config_digest
    ):
        raise NormalizationError(
            f"source config digest mismatch: "
            f"{source_config_digest} != {expected_source_config_digest}"
        )
    config_path = _validate_blob(cache_root, manifest["config"])
    try:
        config = json.loads(config_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise NormalizationError(f"invalid image config: {error}") from error

    layers = manifest["layers"]
    rootfs = config.get("rootfs")
    if not isinstance(rootfs, dict) or rootfs.get("type") != "layers":
        raise NormalizationError("image config rootfs.type must be 'layers'")
    diff_ids = rootfs.get("diff_ids")
    if not isinstance(diff_ids, list) or len(diff_ids) != len(layers):
        raise NormalizationError(
            "image config diff_ids must match the manifest layer count"
        )
    for diff_id in diff_ids:
        if not isinstance(diff_id, str) or _DIGEST_PATTERN.fullmatch(diff_id) is None:
            raise NormalizationError(f"invalid rootfs diff_id: {diff_id!r}")

    target_layers: list[dict[str, Any]] = []
    target_diff_ids = list(diff_ids)
    scans: dict[str, LayerScan] = {}
    source_diff_ids: dict[str, str] = {}
    normalized: dict[str, NormalizedLayer] = {}
    changed_positions: list[int] = []
    total_changed_owners = 0
    layer_mappings: list[dict[str, Any]] = []

    temporary_parent = cache_root / "tmp"
    temporary_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    for position, descriptor in enumerate(layers):
        if not isinstance(descriptor, dict):
            raise NormalizationError(f"layer {position + 1} is not a descriptor")
        source_path = _validate_blob(cache_root, descriptor)
        source_digest = _descriptor_digest(descriptor)
        if source_digest not in scans:
            scans[source_digest] = scan_layer(source_path)
            source_diff_ids[source_digest] = (
                "sha256:" + _sha256_gzip_payload(source_path)
            )
        scan = scans[source_digest]
        if source_diff_ids[source_digest] != diff_ids[position]:
            raise NormalizationError(
                f"uncompressed digest mismatch for layer {position + 1}: "
                f"{source_diff_ids[source_digest]} != {diff_ids[position]}"
            )
        if progress is not None:
            progress(
                f"layer {position + 1}/{len(layers)} "
                f"entries={scan.entries} nonzero_owners={scan.nonzero_owners}"
            )

        if not scan.nonzero_owners:
            target_layers.append(dict(descriptor))
            layer_mappings.append(
                {
                    "position": position + 1,
                    "entries": scan.entries,
                    "changed_owners": 0,
                    "source_blob": descriptor["digest"],
                    "target_blob": descriptor["digest"],
                    "source_diff_id": diff_ids[position],
                    "target_diff_id": diff_ids[position],
                }
            )
            continue
        if source_digest not in normalized:
            with tempfile.TemporaryDirectory(
                dir=temporary_parent,
                prefix=f"normalize-{source_digest[:12]}-",
            ) as temporary:
                normalized[source_digest] = normalize_layer(
                    cache_root,
                    source_path,
                    Path(temporary),
                )
        result = normalized[source_digest]
        target_descriptor = dict(descriptor)
        target_descriptor["digest"] = result.digest
        target_descriptor["size"] = result.size
        target_layers.append(target_descriptor)
        target_diff_ids[position] = result.diff_id
        changed_positions.append(position + 1)
        total_changed_owners += result.changed_owners
        layer_mappings.append(
            {
                "position": position + 1,
                "entries": result.entries,
                "changed_owners": result.changed_owners,
                "source_blob": descriptor["digest"],
                "target_blob": result.digest,
                "source_diff_id": diff_ids[position],
                "target_diff_id": result.diff_id,
            }
        )

    if (
        expected_changed_layer_positions is not None
        and changed_positions != expected_changed_layer_positions
    ):
        raise NormalizationError(
            f"changed layer inventory mismatch: "
            f"{changed_positions} != {expected_changed_layer_positions}"
        )
    if (
        expected_changed_owner_entries is not None
        and total_changed_owners != expected_changed_owner_entries
    ):
        raise NormalizationError(
            f"changed owner inventory mismatch: "
            f"{total_changed_owners} != {expected_changed_owner_entries}"
        )

    target_config = dict(config)
    target_rootfs = dict(rootfs)
    target_rootfs["diff_ids"] = target_diff_ids
    target_config["rootfs"] = target_rootfs
    config_semantic_check = dict(target_config)
    config_semantic_rootfs = dict(target_rootfs)
    config_semantic_rootfs["diff_ids"] = diff_ids
    config_semantic_check["rootfs"] = config_semantic_rootfs
    if config_semantic_check != config:
        raise NormalizationError(
            "target config changed beyond rootfs.diff_ids"
        )
    config_digest, config_size = _publish_bytes(
        cache_root,
        _json_bytes(target_config),
    )

    target_manifest = dict(manifest)
    target_config_descriptor = dict(manifest["config"])
    target_config_descriptor["digest"] = config_digest
    target_config_descriptor["size"] = config_size
    target_manifest["config"] = target_config_descriptor
    target_manifest["layers"] = target_layers
    manifest_semantic_check = dict(target_manifest)
    manifest_semantic_check["config"] = manifest["config"]
    manifest_semantic_check["layers"] = manifest["layers"]
    if manifest_semantic_check != manifest:
        raise NormalizationError(
            "target manifest changed beyond config/layer descriptors"
        )
    manifest_payload = _json_bytes(target_manifest)
    manifest_digest = "sha256:" + hashlib.sha256(manifest_payload).hexdigest()

    target_path = cache_root / "manifests" / target_reference
    _publish_manifest_path(target_path, manifest_payload)
    # Docker may resolve a tag with HEAD and then request the canonical
    # manifest digest.  The minimal loopback registry serves files by their
    # reference, so publish the digest alias as well as the human-readable tag.
    _publish_manifest_path(
        cache_root / "manifests" / manifest_digest,
        manifest_payload,
    )

    return {
        "normalizer_version": NORMALIZER_VERSION,
        "source_reference": source_reference,
        "source_manifest_digest": (
            "sha256:"
            + _sha256_file(cache_root / "manifests" / source_reference)
        ),
        "source_config_digest": source_config_digest,
        "target_reference": target_reference,
        "target_manifest_digest": manifest_digest,
        "target_config_digest": config_digest,
        "layer_count": len(layers),
        "changed_layer_positions": changed_positions,
        "changed_owner_entries": total_changed_owners,
        "layers": layer_mappings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize layer UID/GID metadata in a local Docker registry cache "
            "for the single-UID Toolathlon runtime."
        )
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-reference", default="1016beta")
    parser.add_argument("--target-reference", default="1016beta-singleuid-raw")
    parser.add_argument(
        "--expected-source-manifest-digest",
        default=OFFICIAL_TOOLATHLON_MANIFEST_DIGEST,
    )
    args = parser.parse_args()
    result = normalize_registry_cache(
        args.cache_root,
        args.source_reference,
        args.target_reference,
        expected_source_manifest_digest=args.expected_source_manifest_digest,
        expected_source_config_digest=OFFICIAL_TOOLATHLON_CONFIG_DIGEST,
        expected_changed_layer_positions=(
            OFFICIAL_TOOLATHLON_CHANGED_LAYER_POSITIONS
        ),
        expected_changed_owner_entries=(
            OFFICIAL_TOOLATHLON_CHANGED_OWNER_ENTRIES
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
