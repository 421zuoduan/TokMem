#!/usr/bin/env bash
set -euo pipefail

PARENT_MOUNT_NAMESPACE="${TOOLATHLON_ROOTLESS_PARENT_MOUNT_NAMESPACE:-}"
CURRENT_MOUNT_NAMESPACE="$(readlink /proc/self/ns/mnt)"
if [[ "${_DOCKERD_ROOTLESS_CHILD:-}" != "1" ]] \
    || [[ -z "${ROOTLESSKIT_STATE_DIR:-}" ]] \
    || [[ -z "${ROOTLESSKIT_PARENT_EUID:-}" ]] \
    || [[ "${ROOTLESSKIT_PARENT_EUID}" == "0" ]] \
    || [[ -z "$PARENT_MOUNT_NAMESPACE" ]] \
    || [[ "$CURRENT_MOUNT_NAMESPACE" == "$PARENT_MOUNT_NAMESPACE" ]] \
    || [[ "$(id -u)" != "0" ]]; then
    echo "Refusing to alter /run outside a RootlessKit child namespace" >&2
    exit 2
fi

DOCKERD_BIN="${TOOLATHLON_ROOTLESS_DOCKERD:?missing rootless dockerd path}"
if [[ ! -x "$DOCKERD_BIN" ]]; then
    echo "Rootless dockerd is not executable: $DOCKERD_BIN" >&2
    exit 2
fi

# The parent and child mount namespace IDs differ, and /run is a RootlessKit
# copy-up mount in the child. The second dockerd listener recreates this path
# privately so the unmodified Toolathlon runner can mount the conventional
# socket path into its task container.
rm -f -- /run/docker.sock
exec "$DOCKERD_BIN" "$@"
