#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="${TOOLATHLON_ROOTLESS_BIN_DIR:-$TRACK_DIR/artifacts/rootless-docker/bin}"
HELPER_DIR="$TRACK_DIR/rootless_helpers"
DEFAULT_RUNTIME_PARENT="${XDG_RUNTIME_DIR:-/tmp}"
RUNTIME_ROOT="${TOOLATHLON_ROOTLESS_RUNTIME_ROOT:-$DEFAULT_RUNTIME_PARENT/compositional-toolathlon-rootless}"
DATA_ROOT="${TOOLATHLON_ROOTLESS_DATA_ROOT:-$TRACK_DIR/artifacts/rootless-docker-data}"
CLIENT_SOCKET="$RUNTIME_ROOT/docker.sock"
CURRENT_UID="$(id -u)"
MANAGED_MARKER=".toolathlon-rootless-managed"
MARKER_VERSION="toolathlon-rootless-managed-v1"
DOCKERD_EXTRA_ARGS=()

umask 077

for argument in "$@"; do
    if [[ "$argument" =~ ^--registry-mirror=https://[^[:space:]]+$ ]]; then
        DOCKERD_EXTRA_ARGS+=("$argument")
    elif [[ "$argument" =~ ^--max-concurrent-downloads=([1-9][0-9]?)$ ]] \
        && ((BASH_REMATCH[1] <= 32)); then
        DOCKERD_EXTRA_ARGS+=("$argument")
    else
        echo "Unsupported rootless dockerd argument: $argument" >&2
        echo "Allowed: --registry-mirror=https://... and --max-concurrent-downloads=1..32" >&2
        exit 2
    fi
done

if ! command -v realpath >/dev/null; then
    echo "realpath is required" >&2
    exit 2
fi
for configured_path in "$BIN_DIR" "$RUNTIME_ROOT" "$DATA_ROOT"; do
    if [[ "$configured_path" != /* ]]; then
        echo "Rootless Docker paths must be absolute: $configured_path" >&2
        exit 2
    fi
    if [[ -L "$configured_path" ]]; then
        echo "Refusing symlink directory: $configured_path" >&2
        exit 2
    fi
done
BIN_DIR="$(realpath -m -- "$BIN_DIR")"
RUNTIME_ROOT="$(realpath -m -- "$RUNTIME_ROOT")"
DATA_ROOT="$(realpath -m -- "$DATA_ROOT")"
CLIENT_SOCKET="$RUNTIME_ROOT/docker.sock"

paths_overlap() {
    local first="$1"
    local second="$2"
    [[ "$first" == "$second" ]] \
        || [[ "$first" == "$second/"* ]] \
        || [[ "$second" == "$first/"* ]]
}

if paths_overlap "$BIN_DIR" "$RUNTIME_ROOT" \
    || paths_overlap "$BIN_DIR" "$DATA_ROOT" \
    || paths_overlap "$RUNTIME_ROOT" "$DATA_ROOT"; then
    echo "Rootless Docker bin, runtime, and data paths must not overlap" >&2
    exit 2
fi

prepare_managed_directory() {
    local directory="$1"
    local role="$2"
    local marker="$directory/$MANAGED_MARKER"
    local expected_marker="$MARKER_VERSION:$role"
    local first_entry=""
    if [[ -L "$directory" ]]; then
        echo "Refusing symlink directory: $directory" >&2
        exit 2
    fi
    if [[ -d "$directory" ]] && [[ ! -e "$marker" ]]; then
        first_entry="$(find "$directory" -mindepth 1 -maxdepth 1 -print -quit)"
    fi
    if [[ -n "$first_entry" ]]; then
        echo "Refusing non-empty directory without rootless marker: $directory" >&2
        exit 2
    fi
    mkdir -p "$directory"
    if [[ ! -d "$directory" ]] \
        || [[ "$(stat -c %u "$directory")" != "$CURRENT_UID" ]]; then
        echo "Directory is not owned by the current user: $directory" >&2
        exit 2
    fi
    chmod 700 "$directory"
    if [[ -e "$marker" || -L "$marker" ]]; then
        if [[ ! -f "$marker" ]] \
            || [[ -L "$marker" ]] \
            || [[ "$(stat -c %u "$marker")" != "$CURRENT_UID" ]] \
            || [[ "$(<"$marker")" != "$expected_marker" ]]; then
            echo "Invalid rootless directory marker: $marker" >&2
            exit 2
        fi
    else
        printf '%s\n' "$expected_marker" >"$marker"
        chmod 600 "$marker"
    fi
}

prepare_private_subdirectory() {
    local directory="$1"
    if [[ -L "$directory" ]]; then
        echo "Refusing symlink directory: $directory" >&2
        exit 2
    fi
    mkdir -p "$directory"
    if [[ ! -d "$directory" ]] \
        || [[ "$(stat -c %u "$directory")" != "$CURRENT_UID" ]]; then
        echo "Directory is not owned by the current user: $directory" >&2
        exit 2
    fi
    chmod 700 "$directory"
}

for executable in \
    dockerd \
    dockerd-rootless.sh \
    rootlesskit \
    runc \
    containerd \
    containerd-shim-runc-v2; do
    if [[ ! -x "$BIN_DIR/$executable" ]]; then
        echo "Missing rootless Docker component: $BIN_DIR/$executable" >&2
        exit 2
    fi
done
for helper in newuidmap newgidmap runc_singleuid.py; do
    if [[ ! -x "$HELPER_DIR/$helper" ]]; then
        echo "Missing executable helper: $HELPER_DIR/$helper" >&2
        exit 2
    fi
done

prepare_managed_directory "$RUNTIME_ROOT" runtime
prepare_managed_directory "$DATA_ROOT" data
if [[ "$(stat -Lc %d:%i "$RUNTIME_ROOT")" == \
    "$(stat -Lc %d:%i "$DATA_ROOT")" ]]; then
    echo "Runtime and data roots resolve to the same directory" >&2
    exit 2
fi
prepare_private_subdirectory "$RUNTIME_ROOT/state"
prepare_private_subdirectory "$RUNTIME_ROOT/exec"

# The upstream dockerd-rootless.sh requires this variable even though the
# RootlessKit state directory is explicitly configured below.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-$RUNTIME_ROOT}"

export PATH="$HELPER_DIR:$BIN_DIR:$PATH"
export TOOLATHLON_ROOTLESS_DOCKERD="$BIN_DIR/dockerd"
export TOOLATHLON_ROOTLESS_RUNC="$BIN_DIR/runc"
TOOLATHLON_ROOTLESS_PARENT_MOUNT_NAMESPACE="$(readlink /proc/self/ns/mnt)"
export TOOLATHLON_ROOTLESS_PARENT_MOUNT_NAMESPACE
export DOCKERD="$SCRIPT_DIR/rootless_dockerd_child.sh"
export DOCKERD_ROOTLESS_ROOTLESSKIT_STATE_DIR="$RUNTIME_ROOT/state"
# The daemon needs its own network namespace so it has CAP_NET_ADMIN there.
# Using RootlessKit "host" networking leaves the daemon in the real host
# network namespace and makes even `docker run --network host` fail while
# creating libnetwork's sandbox.  The pinned rootless extras archive includes
# vpnkit, which is the upstream no-slirp4netns fallback.
export DOCKERD_ROOTLESS_ROOTLESSKIT_NET=vpnkit
export DOCKERD_ROOTLESS_ROOTLESSKIT_MTU=1500
export DOCKERD_ROOTLESS_ROOTLESSKIT_PORT_DRIVER=builtin
export DOCKERD_ROOTLESS_ROOTLESSKIT_DISABLE_HOST_LOOPBACK=false

echo "Rootless Docker client socket: unix://$CLIENT_SOCKET"
echo "Client command: export DOCKER_HOST=unix://$CLIENT_SOCKET"
echo "Rootless Docker data root: $DATA_ROOT"
exec "$BIN_DIR/dockerd-rootless.sh" \
    --host="unix://$CLIENT_SOCKET" \
    --host=unix:///var/run/docker.sock \
    --group=root \
    --data-root="$DATA_ROOT" \
    --exec-root="$RUNTIME_ROOT/exec" \
    --pidfile="$RUNTIME_ROOT/docker.pid" \
    --storage-driver=vfs \
    --iptables=false \
    --ip6tables=false \
    --bridge=none \
    --add-runtime="singleuid=$HELPER_DIR/runc_singleuid.py" \
    --default-runtime=singleuid \
    --log-level=info \
    "${DOCKERD_EXTRA_ARGS[@]}"
