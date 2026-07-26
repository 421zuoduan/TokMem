#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="${TOOLATHLON_ROOTLESS_BIN_DIR:-$TRACK_DIR/artifacts/rootless-docker/bin}"
DEFAULT_RUNTIME_PARENT="${XDG_RUNTIME_DIR:-/tmp}"
RUNTIME_ROOT="${TOOLATHLON_ROOTLESS_RUNTIME_ROOT:-$DEFAULT_RUNTIME_PARENT/compositional-toolathlon-rootless}"
CLIENT_SOCKET="$RUNTIME_ROOT/docker.sock"
IMAGE="${1:-lockon0927/toolathlon-task-image:1016beta}"

for configured_path in "$BIN_DIR" "$RUNTIME_ROOT"; do
    if [[ "$configured_path" != /* ]]; then
        echo "Rootless Docker paths must be absolute: $configured_path" >&2
        exit 2
    fi
done
BIN_DIR="$(realpath -m -- "$BIN_DIR")"
RUNTIME_ROOT="$(realpath -m -- "$RUNTIME_ROOT")"
CLIENT_SOCKET="$RUNTIME_ROOT/docker.sock"
if [[ ! -x "$BIN_DIR/docker" ]]; then
    echo "Rootless Docker client is missing: $BIN_DIR/docker" >&2
    exit 2
fi
if [[ ! -S "$CLIENT_SOCKET" ]]; then
    echo "Rootless Docker socket is missing: $CLIENT_SOCKET" >&2
    exit 2
fi

DOCKER=("$BIN_DIR/docker" --host "unix://$CLIENT_SOCKET")
"${DOCKER[@]}" info >/dev/null
"${DOCKER[@]}" image inspect "$IMAGE" >/dev/null
"${DOCKER[@]}" run --rm \
    --name "toolathlon-rootless-check-$$" \
    --network host \
    --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
    "$IMAGE" \
    bash -lc '
        set -euo pipefail
        test "$(id -u)" = 0
        docker version --format \
            "nested-client={{.Client.Version}} nested-server={{.Server.Version}}"
    '
