#!/usr/bin/env bash
# run_roi_planner_test.sh — ArmorRoiPlanner 单元测试：编译 + 运行
#
# 用法（在项目根目录执行）：bash test/run_roi_planner_test.sh
# 编译产物放在 cache/roi_planner_test/（cache/ 已 gitignore）。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/cache/roi_planner_test"
mkdir -p "$OUT"

echo "=== 编译 ArmorRoiPlanner 单元测试 ==="
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
    -I"$ROOT/include" \
    "$ROOT/test/roi_planner_test.cpp" \
    "$ROOT/src/Armor/ArmorRoiPlanner.cpp" \
    -o "$OUT/roi_planner_test" \
    $(pkg-config --cflags --libs opencv4) \
    -pthread

echo "=== 运行 ArmorRoiPlanner 单元测试 ==="
"$OUT/roi_planner_test"
echo "=== ArmorRoiPlanner 单元测试全部通过 ==="
