#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
    echo "Usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CALLER_WORKDIR="$(pwd -P)"
BIN_DIR="${TOOLATHLON_ROOTLESS_BIN_DIR:-$TRACK_DIR/artifacts/rootless-docker/bin}"
DEFAULT_RUNTIME_PARENT="${XDG_RUNTIME_DIR:-/tmp}"
RUNTIME_ROOT="${TOOLATHLON_ROOTLESS_RUNTIME_ROOT:-$DEFAULT_RUNTIME_PARENT/compositional-toolathlon-rootless}"
CLIENT_SOCKET="$RUNTIME_ROOT/docker.sock"
CHILD_PID_FILE="$RUNTIME_ROOT/state/child_pid"
for configured_path in "$BIN_DIR" "$RUNTIME_ROOT"; do
    if [[ "$configured_path" != /* ]]; then
        echo "Rootless Docker paths must be absolute: $configured_path" >&2
        exit 2
    fi
done
BIN_DIR="$(realpath -m -- "$BIN_DIR")"
RUNTIME_ROOT="$(realpath -m -- "$RUNTIME_ROOT")"
CLIENT_SOCKET="$RUNTIME_ROOT/docker.sock"
CHILD_PID_FILE="$RUNTIME_ROOT/state/child_pid"
if [[ ! -x "$BIN_DIR/docker" ]]; then
    echo "Rootless Docker client is missing: $BIN_DIR/docker" >&2
    exit 2
fi
if [[ ! -S "$CLIENT_SOCKET" ]]; then
    echo "Rootless Docker socket is missing: $CLIENT_SOCKET" >&2
    exit 2
fi
if [[ ! -r "$CHILD_PID_FILE" ]]; then
    echo "RootlessKit child PID is missing: $CHILD_PID_FILE" >&2
    exit 2
fi
ROOTLESSKIT_CHILD_PID="$(<"$CHILD_PID_FILE")"
if [[ ! "$ROOTLESSKIT_CHILD_PID" =~ ^[0-9]+$ ]] || \
   [[ ! -e "/proc/$ROOTLESSKIT_CHILD_PID/ns/user" ]] || \
   [[ ! -e "/proc/$ROOTLESSKIT_CHILD_PID/ns/mnt" ]] || \
   [[ ! -e "/proc/$ROOTLESSKIT_CHILD_PID/ns/net" ]]; then
    echo "RootlessKit child PID is stale or invalid: $ROOTLESSKIT_CHILD_PID" >&2
    exit 2
fi
if ! command -v nsenter >/dev/null; then
    echo "nsenter is required to enter the RootlessKit runtime namespace" >&2
    exit 2
fi

export PATH="$BIN_DIR:$PATH"
export DOCKER_HOST="unix://$CLIENT_SOCKET"
NSENTER=(
    nsenter
    --target "$ROOTLESSKIT_CHILD_PID"
    --user
    --mount
    --net
    "--wd=$CALLER_WORKDIR"
    --preserve-credentials
)
"${NSENTER[@]}" "$BIN_DIR/docker" info >/dev/null
exec "${NSENTER[@]}" env \
    PATH="$PATH" \
    DOCKER_HOST="$DOCKER_HOST" \
    "$@"
