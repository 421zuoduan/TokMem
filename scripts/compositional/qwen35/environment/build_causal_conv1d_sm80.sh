#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ARTIFACT_DIR="$ROOT_DIR/results/compositional/qwen35_environment_artifacts"
SOURCE_ARCHIVE="$ARTIFACT_DIR/sources/causal_conv1d-1.6.2.post1.tar.gz"
OUTPUT_DIR="$ARTIFACT_DIR/rebuilt-wheels"
PATCH_FILE="$ROOT_DIR/scripts/compositional/qwen35/environment/causal_conv1d-sm80.patch"
BUILD_ENV_PREFIX="/home/shilong/anaconda3/envs/tokmem-qwen35"
CUDA_TOOLKIT_PREFIX="/home/shilong/anaconda3/envs/draftOPRD"
EXPECTED_SOURCE_SHA256="245e314ea21064ded7a5bf6b3b842b644aa6f92e45cecfe3e935629744c35ff4"

if [[ ! -f "$SOURCE_ARCHIVE" ]]; then
    echo "Missing pinned source archive: $SOURCE_ARCHIVE" >&2
    echo "Download causal-conv1d==1.6.2.post1 sdist and verify its SHA-256 first." >&2
    exit 2
fi
if [[ "$(sha256sum "$SOURCE_ARCHIVE" | awk '{print $1}')" != \
    "$EXPECTED_SOURCE_SHA256" ]]; then
    echo "Source archive checksum mismatch: $SOURCE_ARCHIVE" >&2
    exit 2
fi
if [[ ! -x "$BUILD_ENV_PREFIX/bin/python" ]]; then
    echo "Expected Python build environment is incomplete: $BUILD_ENV_PREFIX" >&2
    exit 2
fi
if [[ ! -x "$CUDA_TOOLKIT_PREFIX/bin/nvcc" ]]; then
    echo "Expected CUDA 12.8 toolkit is incomplete: $CUDA_TOOLKIT_PREFIX" >&2
    exit 2
fi

BUILD_ROOT="$(mktemp -d /tmp/tokmem-qwen35-causal-build.XXXXXX)"
trap 'rm -rf "$BUILD_ROOT"' EXIT
tar -xzf "$SOURCE_ARCHIVE" -C "$BUILD_ROOT"
SOURCE_DIR="$BUILD_ROOT/causal_conv1d-1.6.2.post1"
BUILD_OUTPUT_DIR="$BUILD_ROOT/wheels"
patch -d "$SOURCE_DIR" -p1 < "$PATCH_FILE"
mkdir -p "$BUILD_OUTPUT_DIR"

source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate "$BUILD_ENV_PREFIX"
if [[ "$(python -c 'import platform; print(platform.python_version())')" != \
    "3.10.0" ]]; then
    echo "The build environment must use Python 3.10.0." >&2
    exit 2
fi
if [[ "$(python -c 'import torch; print(torch.__version__)')" != \
    "2.7.1+cu128" ]]; then
    echo "The build environment must contain torch 2.7.1+cu128." >&2
    exit 2
fi
if [[ "$(python -c 'import torch; print(torch.version.cuda)')" != "12.8" ]]; then
    echo "The build environment must use the CUDA 12.8 PyTorch runtime." >&2
    exit 2
fi
if [[ "$(python -c 'import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)')" != \
    "True" ]]; then
    echo "The build environment must use PyTorch's CXX11 ABI." >&2
    exit 2
fi
if [[ "$("$CUDA_TOOLKIT_PREFIX/bin/nvcc" --version)" != *"release 12.8"* ]]; then
    echo "The build toolkit must provide nvcc 12.8." >&2
    exit 2
fi

export CUDA_HOME="$CUDA_TOOLKIT_PREFIX"
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE
export MAX_JOBS=4
python -m pip wheel \
    "$SOURCE_DIR" \
    --no-build-isolation \
    --no-deps \
    --wheel-dir "$BUILD_OUTPUT_DIR"

mapfile -t built_wheels < <(
    find "$BUILD_OUTPUT_DIR" -maxdepth 1 -type f -name '*.whl' \
        -print | LC_ALL=C sort
)
if [[ "${#built_wheels[@]}" -ne 1 ]]; then
    echo "Expected exactly one rebuilt wheel, found ${#built_wheels[@]}." >&2
    exit 2
fi
BUILT_WHEEL="${built_wheels[0]}"
EXPECTED_WHEEL_NAME="causal_conv1d-1.6.2.post1-cp310-cp310-linux_x86_64.whl"
if [[ "$(basename "$BUILT_WHEEL")" != "$EXPECTED_WHEEL_NAME" ]]; then
    echo "Unexpected rebuilt wheel tag: $(basename "$BUILT_WHEEL")" >&2
    exit 2
fi

EXTRACTED_DIR="$BUILD_ROOT/extracted"
mkdir -p "$EXTRACTED_DIR"
unzip -q -j "$BUILT_WHEEL" 'causal_conv1d_cuda*.so' -d "$EXTRACTED_DIR"
mapfile -t built_extensions < <(
    find "$EXTRACTED_DIR" -maxdepth 1 -type f \
        -name 'causal_conv1d_cuda*.so' -print
)
if [[ "${#built_extensions[@]}" -ne 1 ]]; then
    echo "Expected exactly one causal-conv1d extension in the wheel." >&2
    exit 2
fi
BUILT_EXTENSION="${built_extensions[0]}"
CUBIN_NAMES="$(
    "$CUDA_TOOLKIT_PREFIX/bin/cuobjdump" --list-elf "$BUILT_EXTENSION" \
        | awk '/ELF file/ {print $NF}' \
        | LC_ALL=C sort
)"
EXPECTED_CUBIN_NAMES="$(
    printf '%s\n' \
        causal_conv1d_bwd.sm_80.cubin \
        causal_conv1d_fwd.sm_80.cubin \
        causal_conv1d_update.sm_80.cubin
)"
if [[ "$CUBIN_NAMES" != "$EXPECTED_CUBIN_NAMES" ]]; then
    echo "The rebuilt extension does not contain exactly the three SM80 cubins." >&2
    exit 2
fi
HIGHEST_GLIBC="$(
    readelf --version-info "$BUILT_EXTENSION" \
        | grep -o 'GLIBC_[0-9.]*' \
        | sort -Vu \
        | tail -n 1
)"
if [[ "$(printf '%s\n' "GLIBC_2.14" "$HIGHEST_GLIBC" \
    | sort -V | tail -n 1)" != "GLIBC_2.14" ]]; then
    echo "The rebuilt extension requires unsupported $HIGHEST_GLIBC." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
install -m 0644 "$BUILT_WHEEL" "$OUTPUT_DIR/$EXPECTED_WHEEL_NAME"
sha256sum "$OUTPUT_DIR/$EXPECTED_WHEEL_NAME"
