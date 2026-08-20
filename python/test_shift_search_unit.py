#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单元测试：帧平移自动搜索 / 公共段 / 插值采样 / 平均 dt 计算（不依赖视频与棋盘格）。

用法: python3 python/test_shift_search_unit.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_record as ar  # noqa: E402


def make_curves(total, n_info, delay, valid_every=1, seed=0):
    """构造合成数据：
    - checker_yaw_full[k] = true(k)（长度为 total，forward-filled 后形态）；
    - imu_unwrapped_full[i] = true(i - delay)（文本行 i 记录的 IMU 比视频晚 delay 帧，
      用线性插值生成，因此脚本应搜到 shift ≈ -delay）；
    - valid_idx：每 valid_every 帧一个有效解析帧。
    """
    rng = np.random.default_rng(seed)
    k = np.arange(total)
    true = 0.35 * np.sin(2 * np.pi * 1.3 * k / total) \
        + 0.12 * np.sin(2 * np.pi * 4.7 * k / total + 0.8)
    # imu 文本行 i 对应 true(i - delay)：对 n_info 个行位置做线性插值
    imu = np.interp(np.arange(n_info) - delay, k, true)
    checker = true.copy()
    valid_idx = np.arange(0, total, valid_every)
    return checker, imu, valid_idx


def test_fractional_shift():
    total = 300
    n_info = 300
    delay = 2.5
    checker, imu, valid_idx = make_curves(total, n_info, delay, valid_every=2)
    best_s, mse = ar.search_best_shift(valid_idx, checker, imu, total, n_info,
                                       precision=0.1, progress=False)
    print(f"  [fractional] delay={delay} -> best_shift={best_s:.2f} mse={mse:.6f}")
    assert abs(best_s - (-delay)) <= 0.11, f"期望 {-delay}，得到 {best_s}"


def test_integer_shift():
    total = 300
    n_info = 300
    delay = 3.0
    checker, imu, valid_idx = make_curves(total, n_info, delay, valid_every=1)
    best_s, mse = ar.search_best_shift(valid_idx, checker, imu, total, n_info,
                                       precision=0.1, progress=False)
    print(f"  [integer] delay={delay} -> best_shift={best_s:.2f} mse={mse:.6f}")
    assert abs(best_s - (-delay)) <= 0.11, f"期望 {-delay}，得到 {best_s}"


def test_negative_fractional_shift():
    total = 300
    n_info = 300
    delay = -1.5  # IMU 超前 → 应搜到 +1.5
    checker, imu, valid_idx = make_curves(total, n_info, delay, valid_every=3)
    best_s, mse = ar.search_best_shift(valid_idx, checker, imu, total, n_info,
                                       precision=0.1, progress=False)
    print(f"  [negative-frac] delay={delay} -> best_shift={best_s:.2f} mse={mse:.6f}")
    assert abs(best_s - (-delay)) <= 0.11, f"期望 {-delay}，得到 {best_s}"


def test_n_info_less_than_total():
    # 文本行数少于视频帧数（部分帧未写入文本）
    total = 300
    n_info = 260
    delay = 2.0
    checker, imu, valid_idx = make_curves(total, n_info, delay, valid_every=1)
    max_shift = min(total, n_info) / 2.0
    assert max_shift == 130.0
    best_s, mse = ar.search_best_shift(valid_idx, checker, imu, total, n_info,
                                       precision=0.1, progress=False)
    print(f"  [n_info<total] delay={delay} -> best_shift={best_s:.2f} mse={mse:.6f}")
    assert abs(best_s - (-delay)) <= 0.11, f"期望 {-delay}，得到 {best_s}"


def test_common_segment_endpoint_discard():
    # 小数 shift=2.5, total=300, n_info=300:
    # k_start = ceil(2.5)=3, k_end = min(300, floor(299+2.5)+1) = 300
    k_start, k_end = ar.common_segment_range(2.5, 300, 300)
    assert (k_start, k_end) == (3, 300), (k_start, k_end)
    # 与整数 shift=2 相比丢掉了插值窗口越界的端点帧（少 1 帧）
    ks2, ke2 = ar.common_segment_range(2.0, 300, 300)
    assert (ks2, ke2) == (2, 302 - 0) or (ks2, ke2) == (2, 300), (ks2, ke2)
    # 负小数 shift
    k_start, k_end = ar.common_segment_range(-1.3, 300, 300)
    assert (k_start, k_end) == (0, 298), (k_start, k_end)


def test_sample_imu_integer_vs_fractional():
    imu = np.arange(10, dtype=np.float64)
    # 整数 shift=2：帧 k ↔ 文本 k-2
    v = ar.sample_imu(imu, 2.0, 2, 8)
    np.testing.assert_allclose(v, imu[0:6])
    # 小数 shift=1.5：帧 k ↔ 文本 k-1.5（线性插值）
    v = ar.sample_imu(imu, 1.5, 2, 8)
    np.testing.assert_allclose(v, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])


def test_avg_dt_range():
    # 模拟 analyze() 中 avg_dt 的取行逻辑：公共段 [k_start,k_end) ↔ 文本位置 [p_first,p_last]
    total, n_info, shift = 300, 300, 2.5
    k_start, k_end = ar.common_segment_range(shift, total, n_info)
    p_first = k_start - shift
    p_last = k_end - 1 - shift
    r0, r1 = int(math.floor(p_first)), int(math.ceil(p_last))
    assert (r0, r1) == (0, 297), (r0, r1)   # 文本位置 [0.5, 296.5] → 覆盖行 0..297
    # 整数 shift 时与旧逻辑 imu_start..imu_start+n_plot-1 一致
    k_start, k_end = ar.common_segment_range(3.0, 300, 300)
    n_plot = k_end - k_start
    imu_start = k_start - 3
    r0 = int(math.floor(k_start - 3.0))
    r1 = int(math.ceil(k_end - 1 - 3.0))
    assert r0 == imu_start and r1 == imu_start + n_plot - 1, (r0, r1, imu_start, n_plot)


if __name__ == "__main__":
    for fn in [test_fractional_shift, test_integer_shift, test_negative_fractional_shift,
               test_n_info_less_than_total, test_common_segment_endpoint_discard,
               test_sample_imu_integer_vs_fractional, test_avg_dt_range]:
        print(f"== {fn.__name__} ==")
        fn()
    print("ALL UNIT TESTS PASSED")
