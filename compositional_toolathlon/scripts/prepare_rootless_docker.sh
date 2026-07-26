#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="${TOOLATHLON_ROOTLESS_BIN_DIR:-$TRACK_DIR/artifacts/rootless-docker/bin}"
DOCKER_VERSION="26.1.3"
DOWNLOAD_BASE_URL="https://download.docker.com/linux/static/stable/x86_64"
DOCKER_ARCHIVE="docker-$DOCKER_VERSION.tgz"
EXTRAS_ARCHIVE="docker-rootless-extras-$DOCKER_VERSION.tgz"
DOCKER_SHA256="a50076d372d3bbe955664707af1a4ce4f5df6b2d896e68b12ecc74e724d1db31"
EXTRAS_SHA256="864852d0210582ef6618f8f3a1de786c31ec53581541769b0db05d71ebfda97b"
CURRENT_UID="$(id -u)"

if [[ "$BIN_DIR" != /* ]]; then
    echo "TOOLATHLON_ROOTLESS_BIN_DIR must be an absolute path: $BIN_DIR" >&2
    exit 2
fi
if ! command -v realpath >/dev/null; then
    echo "realpath is required" >&2
    exit 2
fi
if [[ -L "$BIN_DIR" ]]; then
    echo "Refusing symlink directory: $BIN_DIR" >&2
    exit 2
fi
BIN_DIR="$(realpath -m -- "$BIN_DIR")"

DOCKER_FILES=(
    containerd
    containerd-shim-runc-v2
    ctr
    docker
    docker-init
    docker-proxy
    dockerd
    runc
)
EXTRAS_FILES=(
    dockerd-rootless-setuptool.sh
    dockerd-rootless.sh
    rootlesskit
    rootlesskit-docker-proxy
    vpnkit
)
EXPECTED_FILES=("${DOCKER_FILES[@]}" "${EXTRAS_FILES[@]}")

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "The pinned archives are only for x86_64 hosts" >&2
    exit 2
fi
if ! command -v sha256sum >/dev/null; then
    echo "sha256sum is required" >&2
    exit 2
fi

umask 077
TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/toolathlon-rootless-docker.XXXXXX")"
cleanup() {
    rm -rf -- "$TEMP_ROOT"
}
trap cleanup EXIT

download() {
    local url="$1"
    local output="$2"
    if command -v curl >/dev/null; then
        curl \
            --fail \
            --location \
            --retry 4 \
            --retry-all-errors \
            --connect-timeout 30 \
            --output "$output" \
            "$url"
    elif command -v wget >/dev/null; then
        wget \
            --https-only \
            --tries=5 \
            --timeout=30 \
            --output-document="$output" \
            "$url"
    else
        echo "curl or wget is required" >&2
        exit 2
    fi
}

verify_digest() {
    local archive="$1"
    local expected="$2"
    local actual
    actual="$(sha256sum "$archive")"
    actual="${actual%% *}"
    if [[ "$actual" != "$expected" ]]; then
        echo "SHA-256 mismatch for $(basename "$archive")" >&2
        echo "expected: $expected" >&2
        echo "actual:   $actual" >&2
        exit 2
    fi
}

validate_archive_layout() {
    local archive="$1"
    local prefix="$2"
    shift 2
    local expected_listing="$TEMP_ROOT/$(basename "$archive").expected"
    local actual_listing="$TEMP_ROOT/$(basename "$archive").actual"
    {
        printf '%s/\n' "$prefix"
        local filename
        for filename in "$@"; do
            printf '%s/%s\n' "$prefix" "$filename"
        done
    } | LC_ALL=C sort >"$expected_listing"
    tar -tzf "$archive" | LC_ALL=C sort >"$actual_listing"
    if ! cmp -s "$expected_listing" "$actual_listing"; then
        echo "Unexpected entries in $(basename "$archive"); refusing extraction" >&2
        diff -u "$expected_listing" "$actual_listing" >&2 || true
        exit 2
    fi
}

is_expected_filename() {
    local candidate="$1"
    local expected
    for expected in "${EXPECTED_FILES[@]}"; do
        if [[ "$candidate" == "$expected" ]]; then
            return 0
        fi
    done
    return 1
}

prepare_owned_directory() {
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
}

DOCKER_PATH="$TEMP_ROOT/$DOCKER_ARCHIVE"
EXTRAS_PATH="$TEMP_ROOT/$EXTRAS_ARCHIVE"
echo "Downloading Docker $DOCKER_VERSION static binaries"
download "$DOWNLOAD_BASE_URL/$DOCKER_ARCHIVE" "$DOCKER_PATH"
download "$DOWNLOAD_BASE_URL/$EXTRAS_ARCHIVE" "$EXTRAS_PATH"
verify_digest "$DOCKER_PATH" "$DOCKER_SHA256"
verify_digest "$EXTRAS_PATH" "$EXTRAS_SHA256"
validate_archive_layout "$DOCKER_PATH" docker "${DOCKER_FILES[@]}"
validate_archive_layout \
    "$EXTRAS_PATH" \
    docker-rootless-extras \
    "${EXTRAS_FILES[@]}"

EXTRACT_ROOT="$TEMP_ROOT/extracted"
STAGED_BIN="$TEMP_ROOT/bin"
mkdir -p "$EXTRACT_ROOT" "$STAGED_BIN"
tar -xzf "$DOCKER_PATH" -C "$EXTRACT_ROOT" --no-same-owner
tar -xzf "$EXTRAS_PATH" -C "$EXTRACT_ROOT" --no-same-owner
for filename in "${DOCKER_FILES[@]}"; do
    install -m 0755 "$EXTRACT_ROOT/docker/$filename" "$STAGED_BIN/$filename"
done
for filename in "${EXTRAS_FILES[@]}"; do
    install \
        -m 0755 \
        "$EXTRACT_ROOT/docker-rootless-extras/$filename" \
        "$STAGED_BIN/$filename"
done

prepare_owned_directory "$BIN_DIR"
while IFS= read -r -d '' existing; do
    filename="${existing##*/}"
    if ! is_expected_filename "$filename" \
        || [[ ! -f "$existing" ]] \
        || [[ -L "$existing" ]]; then
        echo "Unexpected entry in rootless Docker bin directory: $existing" >&2
        exit 2
    fi
done < <(find "$BIN_DIR" -mindepth 1 -maxdepth 1 -print0)
chmod 700 "$BIN_DIR"

for filename in "${EXPECTED_FILES[@]}"; do
    destination="$BIN_DIR/$filename"
    staged="$STAGED_BIN/$filename"
    if [[ -e "$destination" || -L "$destination" ]]; then
        if [[ ! -f "$destination" ]] \
            || [[ -L "$destination" ]] \
            || ! cmp -s "$staged" "$destination"; then
            echo "Refusing to overwrite different file: $destination" >&2
            exit 2
        fi
    fi
done

for filename in "${EXPECTED_FILES[@]}"; do
    destination="$BIN_DIR/$filename"
    staged="$STAGED_BIN/$filename"
    if [[ -e "$destination" ]]; then
        chmod 0755 "$destination"
    else
        install -m 0755 "$staged" "$destination"
    fi
done

echo "Installed verified Docker $DOCKER_VERSION binaries in $BIN_DIR"
"$BIN_DIR/docker" --version
"$BIN_DIR/dockerd" --version
"$BIN_DIR/rootlesskit" --version
