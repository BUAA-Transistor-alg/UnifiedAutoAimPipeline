#!/bin/bash
set -e

# 默认使用所有CPU核心
JOBS=$(nproc)

# 解析 -j 参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j*)
            value="${1#-j}"
            if [[ -z "$value" ]]; then
                shift || true
                value="${1:-}"
            fi
            if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
                echo "错误: -j 需要正整数参数" >&2
                echo "Usage: $0 [-j<N>]" >&2
                exit 1
            fi
            JOBS="$value"
            ;;
        *)
            echo "Usage: $0 [-j<N>]"
            echo "  -j<N>  并行编译线程数 (默认: $(nproc))"
            exit 1
            ;;
    esac
    shift
done

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

echo "========================================"
echo "Project: $PROJECT_DIR"
echo "Build:   $BUILD_DIR"
echo "Jobs:    $JOBS"
echo "========================================"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "[1/2] Running cmake..."
cmake "$PROJECT_DIR"

echo ""
echo "[2/2] Running make -j$JOBS..."
make -j"$JOBS"

echo ""
echo "========================================"
echo "Build succeeded!"
echo "========================================"
