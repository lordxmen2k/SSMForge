#!/usr/bin/env bash
# Build the vendored llama.cpp fork's tools (llama-quantize, llama-cli, etc).
#
# Usage: scripts/build_vendor.sh [--cuda|--metal|--cpu-only]
#
# This builds the C++ binaries we shell out to for GGUF quantization.
# The ssmforge Python package expects to find these at:
#   vendor/llama.cpp/build/bin/llama-quantize
#   vendor/llama.cpp/build/bin/llama-cli
#
# On Windows (Git Bash / MINGW64): use this script from Git Bash so POSIX paths work.
# On Linux/macOS: just run as a normal shell script.

set -e

cd "$(dirname "$0")/.."  # script lives in scripts/, cd up to repo root

VENDOR_DIR="vendor/llama.cpp"
BUILD_DIR="$VENDOR_DIR/build"

if [ ! -d "$VENDOR_DIR" ]; then
    echo "Error: $VENDOR_DIR not found. Run:"
    echo "  git submodule update --init --recursive"
    exit 1
fi

# Detect accelerator preference (default: CPU-only for portability).
GPU_FLAGS=""
case "${1:-}" in
    --cuda)
        GPU_FLAGS="-DGGML_CUDA=ON"
        echo "Building with CUDA support..."
        ;;
    --metal)
        GPU_FLAGS="-DGGML_METAL=ON"
        echo "Building with Metal support..."
        ;;
    --cpu-only|"")
        echo "Building CPU-only (default; add --cuda or --metal for accelerators)..."
        ;;
    *)
        echo "Unknown flag: $1"
        echo "Usage: $0 [--cuda|--metal|--cpu-only]"
        exit 1
        ;;
esac

# Configure
cmake -S "$VENDOR_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release $GPU_FLAGS

# Build just the binaries we need (skip server, mtmd, etc for build speed)
cmake --build "$BUILD_DIR" --config Release \
    --target llama-quantize llama-cli gguf-dump gguf-convert \
    -j$(nproc 2>/dev/null || echo 4)

echo ""
echo "Built binaries:"
ls -la "$BUILD_DIR/bin/"
echo ""
echo "Make sure ssmforge can find them:"
echo "  export LLAMA_QUANTIZE_BIN=\"\$(pwd)/$BUILD_DIR/bin/llama-quantize\""
echo "  export LLAMA_CLI_BIN=\"\$(pwd)/$BUILD_DIR/bin/llama-cli\""
