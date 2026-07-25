#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="lockon0927/toolathlon-task-image:1016beta"
OUTPUT="${1:-$TRACK_DIR/artifacts/toolathlon_image_digest.json}"

docker info >/dev/null
docker pull "$IMAGE"
mkdir -p "$(dirname "$OUTPUT")"

DIGESTS="$(docker image inspect "$IMAGE" --format '{{json .RepoDigests}}')"
IMAGE_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
CREATED="$(docker image inspect "$IMAGE" --format '{{.Created}}')"

python - "$OUTPUT" "$IMAGE" "$DIGESTS" "$IMAGE_ID" "$CREATED" <<'PY'
import json
import sys
from pathlib import Path

output, image, digests, image_id, created = sys.argv[1:]
payload = {
    "image": image,
    "repo_digests": json.loads(digests),
    "image_id": image_id,
    "created": created,
}
path = Path(output)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(path)
PY
