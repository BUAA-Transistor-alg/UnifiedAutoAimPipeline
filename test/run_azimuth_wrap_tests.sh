#!/usr/bin/env bash
# run_azimuth_wrap_tests.sh — 「整圈表示差」修复的两个回归测试一键编译+运行
#
# 背景（详见 analysis/yaw_pulse/REPORT.md）：
#   子模组里 chassis_azimuth 曾是 atan2 的 (−π,π] 包裹值，而 MPC 按
#   psi_b = chassis_azimuth + θ_b（θ_b 多圈）使用、参考序列又只在每帧 set() 对齐圈数 ⇒
#   底盘每转一圈，MPC 代价里出现 2π 量级残差 ⇒ 每圈一个恒定幅值力矩脉冲。
#
#   本脚本跑两个测试：
#     azimuth_wrap_test   ：angle_wrap 工具 + FullStrictPoseBuilder 解卷绕（只依赖 libstdc++）
#     mpc_angle_align_test：MPC 每拍整圈对齐（链接 build/libtcbs.so，需先构建 tcbs 目标）
#
# 用法（在项目根目录）：bash test/run_azimuth_wrap_tests.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SUB="sub_module/TorqueControllerForBigSmallYaw_v2"
OUT="${TMPDIR:-/tmp}/azimuth_wrap_tests"
mkdir -p "$OUT"

echo "=== [1/2] azimuth_wrap_test（不需要 libtcbs.so） ==="
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
    -I"$SUB/include" -I"$SUB/include/com" -I"$SUB/include/mpc" \
    test/azimuth_wrap_test.cpp \
    "$SUB/src/com/FullStrictPoseBuilder.cpp" \
    -o "$OUT/azimuth_wrap_test" -lpthread
"$OUT/azimuth_wrap_test"

echo
echo "=== [2/2] mpc_angle_align_test（需要 build/libtcbs.so） ==="
if [[ ! -f build/libtcbs.so ]]; then
    echo "build/libtcbs.so 不存在，先构建：cmake --build build --target tcbs -j\"\$(nproc)\""
    exit 1
fi
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
    -I"$SUB/include" -I"$SUB/include/mpc" -I"$SUB/include/dm" \
    -I"$SUB/include/com" -I"$SUB/include/common" -I/usr/include/eigen3 \
    test/mpc_angle_align_test.cpp \
    -o "$OUT/mpc_angle_align_test" \
    -Lbuild -ltcbs -lceres -lglog -ludev -lpthread
LD_LIBRARY_PATH=build "$OUT/mpc_angle_align_test"
