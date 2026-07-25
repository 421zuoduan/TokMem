#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$TRACK_DIR/.." && pwd)"
TARGET_DIR="${TOOLATHLON_VENDOR_DIR:-$TRACK_DIR/vendor/toolathlon}"
REVISION="2aed2468858f15818acafa178518390cc4b0f5cb"
SOURCE_URL="https://github.com/hkust-nlp/Toolathlon.git"
ARCHIVE_URL="https://codeload.github.com/hkust-nlp/Toolathlon/tar.gz/$REVISION"
TREE_URL="https://api.github.com/repos/hkust-nlp/Toolathlon/git/trees/$REVISION?recursive=1"

configure_local_templates() {
    local source_root="$1"
    if [[ ! -f "$source_root/configs/global_configs.py" ]]; then
        cp \
            "$source_root/configs/global_configs_example.py" \
            "$source_root/configs/global_configs.py"
    fi
    if [[ ! -f "$source_root/configs/token_key_session.py" ]]; then
        cp \
            "$source_root/configs/token_key_session_example.py" \
            "$source_root/configs/token_key_session.py"
    fi
}

if [[ -e "$TARGET_DIR" ]]; then
    if [[ -d "$TARGET_DIR/.git" ]]; then
        ACTUAL_REVISION="$(git -C "$TARGET_DIR" rev-parse HEAD)"
        SOURCE_KIND="git"
    elif [[ -f "$TARGET_DIR/.toolathlon-source.json" ]]; then
        readarray -t SOURCE_METADATA < <(
            python -c \
                'import json,sys; p=json.load(open(sys.argv[1])); print(p["revision"]); print(p["source_kind"])' \
                "$TARGET_DIR/.toolathlon-source.json"
        )
        ACTUAL_REVISION="${SOURCE_METADATA[0]}"
        SOURCE_KIND="${SOURCE_METADATA[1]}"
    else
        echo "Refusing to overwrite unverified path: $TARGET_DIR" >&2
        exit 2
    fi
    if [[ "$ACTUAL_REVISION" != "$REVISION" ]]; then
        echo "Existing snapshot has unexpected revision: $ACTUAL_REVISION" >&2
        exit 2
    fi
    configure_local_templates "$TARGET_DIR"
    echo "Toolathlon $SOURCE_KIND snapshot already pinned at $REVISION"
    exit 0
fi

mkdir -p "$(dirname "$TARGET_DIR")"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
GIT_DIR="$TEMP_DIR/git"

if git -c http.version=HTTP/1.1 clone \
        --filter=blob:none \
        --no-checkout \
        "$SOURCE_URL" \
        "$GIT_DIR" \
    && git -c http.version=HTTP/1.1 -C "$GIT_DIR" \
        fetch --depth 1 origin "$REVISION" \
    && git -C "$GIT_DIR" sparse-checkout init --cone \
    && git -C "$GIT_DIR" sparse-checkout set \
        configs \
        deployment \
        global_preparation \
        local_binary \
        scripts \
        utils \
        tasks/finalpool/arrange-workspace \
        tasks/finalpool/courses-ta-hws \
        tasks/finalpool/detect-revised-terms \
        tasks/finalpool/excel-data-transformation \
        tasks/finalpool/excel-market-research \
        tasks/finalpool/imagenet \
        tasks/finalpool/paper-checker \
        tasks/finalpool/privacy-desensitization \
        tasks/finalpool/reimbursement-form-filler \
        tasks/finalpool/university-course-selection \
    && git -C "$GIT_DIR" checkout --detach "$REVISION"; then
    git -C "$GIT_DIR" status --short
    configure_local_templates "$GIT_DIR"
    mv "$GIT_DIR" "$TARGET_DIR"
    echo "Pinned Toolathlon Git checkout created at $TARGET_DIR"
    exit 0
fi

echo "Git transport failed; downloading a blob-verified code overlay." >&2
TREE_JSON="$TEMP_DIR/tree.json"
TREE_VALID=false
for _ in $(seq 1 8); do
    curl \
        --http1.1 \
        --fail \
        --location \
        --continue-at - \
        --retry 0 \
        --max-time 120 \
        --output "$TREE_JSON" \
        "$TREE_URL" || true
    if python -c \
        'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("truncated") is False and isinstance(p.get("tree"),list)' \
        "$TREE_JSON" 2>/dev/null; then
        TREE_VALID=true
        break
    fi
done
if [[ "$TREE_VALID" = true ]] && \
    python "$SCRIPT_DIR/fetch_pinned_source_overlay.py" \
        --tree-json "$TREE_JSON" \
        --target "$TARGET_DIR" \
        --task-root "$REPO_DIR/datasets/toolathlon/tasks"; then
    configure_local_templates "$TARGET_DIR"
    echo "Pinned Toolathlon blob-verified source overlay created at $TARGET_DIR"
    exit 0
fi

echo "Sparse source download failed; trying the immutable codeload archive." >&2
ARCHIVE="$TEMP_DIR/toolathlon.tar.gz"
SNAPSHOT_DIR="$TEMP_DIR/snapshot"
mkdir -p "$SNAPSHOT_DIR"
curl \
    --fail \
    --location \
    --retry 0 \
    --max-time 1800 \
    --speed-limit 51200 \
    --speed-time 30 \
    --output "$ARCHIVE" \
    "$ARCHIVE_URL"
ARCHIVE_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
tar -xzf "$ARCHIVE" --strip-components=1 -C "$SNAPSHOT_DIR"
configure_local_templates "$SNAPSHOT_DIR"
python -c \
    'import json,sys; json.dump({"source_kind":"github_codeload","source_url":sys.argv[2],"revision":sys.argv[3],"archive_sha256":sys.argv[4]},open(sys.argv[1],"w"),indent=2); open(sys.argv[1],"a").write("\n")' \
    "$SNAPSHOT_DIR/.toolathlon-source.json" \
    "$ARCHIVE_URL" \
    "$REVISION" \
    "$ARCHIVE_SHA256"
mv "$SNAPSHOT_DIR" "$TARGET_DIR"
echo "Pinned Toolathlon codeload snapshot created at $TARGET_DIR"
echo "archive_sha256=$ARCHIVE_SHA256"
