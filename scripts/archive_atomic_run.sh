#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="$1"

mkdir -p "$ROOT_DIR/results"
cp -a "$ROOT_DIR/atomic/runs/$RUN_NAME" "$ROOT_DIR/results/$RUN_NAME"
