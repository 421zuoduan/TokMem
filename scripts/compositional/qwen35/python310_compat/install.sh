#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SOURCE_MODULE="$ROOT_DIR/scripts/compositional/qwen35/python310_compat/tokmem_qwen35_inspect_patch.py"
ENV_PREFIX="/home/shilong/anaconda3/envs/tokmem-qwen35"
ENV_PYTHON="$ENV_PREFIX/bin/python"

if [[ ! -x "$ENV_PYTHON" ]]; then
    echo "Expected conda environment is missing: $ENV_PREFIX" >&2
    exit 2
fi

PURELIB="$("$ENV_PYTHON" -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
TARGET_MODULE="$PURELIB/tokmem_qwen35_inspect_patch.py"
TARGET_PTH="$PURELIB/tokmem_qwen35_inspect_patch.pth"
PTH_STAGING="$(mktemp /tmp/tokmem-qwen35-inspect-patch.XXXXXX)"
trap 'rm -f "$PTH_STAGING"' EXIT
printf 'import tokmem_qwen35_inspect_patch\n' > "$PTH_STAGING"

if [[ -e "$TARGET_MODULE" ]] && ! cmp -s "$SOURCE_MODULE" "$TARGET_MODULE"; then
    echo "Refusing to overwrite a different compatibility module: $TARGET_MODULE" >&2
    exit 2
fi
if [[ -e "$TARGET_PTH" ]] && ! cmp -s "$PTH_STAGING" "$TARGET_PTH"; then
    echo "Refusing to overwrite a different .pth file: $TARGET_PTH" >&2
    exit 2
fi

install -m 0644 "$SOURCE_MODULE" "$TARGET_MODULE"
install -m 0644 "$PTH_STAGING" "$TARGET_PTH"

cmp -s "$SOURCE_MODULE" "$TARGET_MODULE"
"$ENV_PYTHON" -c '
import inspect
import sys

if (
    sys.version_info[:2] == (3, 10)
    and hasattr(inspect.BlockFinder(), "decoratorhasargs")
):
    assert (
        inspect.BlockFinder.tokeneater.__module__
        == "tokmem_qwen35_inspect_patch"
    )
source = """\
@triton.heuristics({"NV": lambda args: triton.cdiv(inner(args["N"]), 64)})
def kernel():
    return 1
"""
block = inspect.getblock(source.splitlines(keepends=True))
assert block[-1].strip() == "return 1"
'

echo "Installed the Qwen3.5 Python 3.10 compatibility module in $ENV_PREFIX"
