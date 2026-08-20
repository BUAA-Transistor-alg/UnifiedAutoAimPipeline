#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_record.py — 解析 Record 模式录制数据 + 视频，对比棋盘格 PnP 欧拉角 yaw 与 IMU yaw

用法:
    python3 python/analyze_record.py <record_dir> [--mode camera|video]
                                     [--precision 0.1] [--show] [--output <png>]

    <record_dir>   Record 模式输出会话目录（record_YYYYmmdd_HHMMSS），内含
                   video_*.mkv 与 frame_info_*.txt
    --mode         config.yaml 中使用的相机参数段（camera_mode / video_mode，默认 camera）
    --precision    自动搜索 shift 的小数精度（默认 0.1，即按 0.1 帧步长细化）
    --show         额外调用 matplotlib 交互显示
    --output       结果图保存路径（默认 <record_dir>/checkerboard_imu_yaw_comparison.png）

功能（对应需求 3）:
  1. 对视频每一帧，参考 other_files/camera_calibration/dist.py 的棋盘格做法
     （棋盘格参数相同：角点数 (11, 8)，方块边长 0.03），尝试 findChessboardCorners；
  2. 成功获取角点后使用 solvePnP 解算位姿，并按 CameraProjection::solvePnP_Cam /
     pnpRvecToEuler 相同的约定获取 cam 系下欧拉角（yaw = atan2(-R02, R22)，
     pitch = asin(-R12)，roll = atan2(R10, R11)）；
  3. 将棋盘格 yaw 与 frame_info 中记录的 imu_euler_yaw 各自解缠绕，绘制在同一张图：
       - 视频未成功解析的帧使用前一帧数据（再往前推；若直到第一帧都无有效值则为 0）；
       - imu_euler_yaw 整体平移一个常数，使「视频成功解析的点」上两者平均值相等；
  4. 帧平移量（视频帧 k 对应文本行 k - shift）不再手动输入，而是自动搜索：
       - 搜索范围 |shift| <= min(视频帧数, 文本行数) / 2；
       - 目标为使「有效解析帧上的误差 MSE」最小（先做与最终绘图一致的均值对齐，
         因此等价于 (imu − checker) 在有效帧上的方差）；
       - 先整数粗搜全范围，再在最佳整数附近按 --precision（默认 0.1）细化，
         即同时搜索整数与小数 shift；
       - 小数 shift 采样 IMU yaw 曲线时使用线性插值（文本位置 k - shift），
         并抛弃插值窗口越出文本数据范围的端点帧；
  5. 平均 dt：取最终公共段（计算范围）对应文本行内所有帧 dt 的均值；
     结果图标题与汇总输出中同时显示使用的最优 shift 与 shift × 平均 dt（时间量）。

相机内参/畸变从 config/config.yaml 解析（经 python/path_resolver.py 的 Python 版
PathResolver 定位项目根目录）。
"""

import argparse
import glob
import math
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")  # 无显示环境也可保存图片
import matplotlib.pyplot as plt
import numpy as np
import yaml

# 使脚本可直接运行（python3 python/analyze_record.py）也能被 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_resolver import resolve_path  # noqa: E402

# ==================== 棋盘格参数（与 dist.py 相同） ====================
CHESSBOARD_SIZE = (11, 8)   # 棋盘格角点数目
SQUARE_SIZE = 0.03          # 每个小方块边长

# 棋盘格角点世界坐标（z=0 平面；PnP 只影响 tvec 尺度，不影响 rvec/yaw）
_objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), dtype=np.float32)
_objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
_objp[:, :2] *= SQUARE_SIZE


def pnp_rvec_to_euler(rvec):
    """与 CameraProjection::pnpRvecToEuler 完全一致：rvec → (yaw, pitch, roll)。"""
    rmat, _ = cv2.Rodrigues(rvec)          # 3x3 float64
    pitch = math.asin(-rmat[1, 2])
    epsilon = 1e-6
    if abs(math.cos(pitch)) > epsilon:
        yaw = math.atan2(-rmat[0, 2], rmat[2, 2])
        roll = math.atan2(rmat[1, 0], rmat[1, 1])
    else:
        roll = 0.0
        yaw = math.atan2(rmat[2, 0], rmat[0, 0])
    return yaw, pitch, roll


def load_camera_params(mode):
    """从 config/config.yaml 解析相机内参/畸变（mode: camera | video，对应配置段
    common.input_mode.camera_mode / common.input_mode.video_mode）。"""
    section = f"{mode}_mode"
    cfg_path = resolve_path("config/config.yaml")
    if not os.path.isfile(cfg_path):
        raise RuntimeError(f"找不到配置文件: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cam = cfg["common"]["input_mode"][section]
    cm = np.array(cam["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dc = np.array(cam["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    print(f"[config] {cfg_path}  ->  common.input_mode.{section}  (K shape {cm.shape}, dist {dc.ravel().tolist()})")
    return cm, dc


def parse_frame_info(info_path):
    """解析 Record 模式的 frame_info txt（v2，15 列），返回 (frame_index[], dt[],
    timestamp[], accepted[], imu_euler_yaw[])。表头/注释行以 '#' 开头被忽略。"""
    idx, dt, ts, accepted, imu_yaw = [], [], [], [], []
    with open(info_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            idx.append(int(parts[0]))
            dt.append(float(parts[1]))
            ts.append(float(parts[2]))
            accepted.append(int(parts[3]) != 0)
            imu_yaw.append(float(parts[4]))   # imu_euler_yaw
    return (np.array(idx), np.array(dt), np.array(ts),
            np.array(accepted, dtype=bool), np.array(imu_yaw))


def forward_fill(values, fill_zero=0.0):
    """视频未成功解析的帧使用前一帧数据（若还没有则继续往前推；
    直到第一帧都无有效值 → fill_zero）。输入 values: (N,) 或 None 的列表。"""
    out = np.empty(len(values), dtype=np.float64)
    last = None
    for i, v in enumerate(values):
        if v is not None:
            last = v
            out[i] = v
        else:
            out[i] = fill_zero if last is None else last
    return out


# ==================== 帧平移（自动搜索）相关 ====================

def common_segment_range(shift, total, n_info):
    """给定帧平移 shift，返回公共段视频帧范围 [k_start, k_end)（视频帧 k ↔ 文本位置
    p = k - shift，两侧数据都需存在：0 <= k < total 且 0 <= p <= n_info - 1）。

    对小数 shift，线性插值需要 p 的两个相邻整数点都在文本数据范围内，因此插值窗口
    越界的端点帧会被抛弃，等价于:
        k_start = max(0, ceil(shift))
        k_end   = min(total, floor(n_info - 1 + shift) + 1)
    当 shift 为整数时上式与旧逻辑 max(0, shift) / min(total, n_info + shift) 完全一致。
    """
    k_start = max(0, math.ceil(shift))
    k_end = min(total, math.floor(n_info - 1 + shift) + 1)
    return k_start, k_end


def sample_imu(imu_unwrapped_full, shift, k_start, k_end):
    """按帧平移 shift 采样 IMU yaw：视频帧 k 取文本位置 k - shift 处的值。
    整数 shift 直接取整数值；小数 shift 用 np.interp 做线性插值
    （插值窗口越界的端点帧已由 common_segment_range 抛弃）。
    返回长度 (k_end - k_start) 的数组。"""
    k = np.arange(k_start, k_end)
    p = k - shift
    if float(shift).is_integer():
        return imu_unwrapped_full[p.astype(np.int64)]
    return np.interp(p, np.arange(len(imu_unwrapped_full)), imu_unwrapped_full)


def mse_for_shift(shift, valid_idx, checker_yaw_full, imu_unwrapped_full,
                  total, n_info):
    """候选 shift 的误差 MSE：仅在棋盘格成功解析（valid_idx 命中）且位于公共段内的
    视频帧上计算；先做与最终绘图一致的均值对齐（有效帧上均值相等），因此 MSE 等价于
    (imu 采样 − checker) 在有效帧上的方差。公共段为空或无有效帧 → inf。"""
    k_start, k_end = common_segment_range(shift, total, n_info)
    if k_end <= k_start:
        return float("inf")
    lo = np.searchsorted(valid_idx, k_start)
    hi = np.searchsorted(valid_idx, k_end)
    if lo >= hi:
        return float("inf")
    kv = valid_idx[lo:hi]                 # 公共段内的有效（成功解析）视频帧
    p = kv - shift
    if float(shift).is_integer():
        imu_v = imu_unwrapped_full[p.astype(np.int64)]
    else:
        imu_v = np.interp(p, np.arange(n_info), imu_unwrapped_full)
    diff = imu_v - checker_yaw_full[kv]
    return float(np.mean((diff - diff.mean()) ** 2))


def search_best_shift(valid_idx, checker_yaw_full, imu_unwrapped_full,
                      total, n_info, precision=0.1, progress=True):
    """自动搜索使误差 MSE 最小的帧平移量 shift（视频帧 k ↔ 文本行 k - shift）。

    搜索范围 |shift| <= min(total, n_info) * 0.1。MSE 关于 shift 分段光滑
    （分段线性插值 + 整数点处公共段集合的微小跳变），故采用粗-细两级搜索：
      1. 整数粗搜：遍历范围内全部整数 shift；
      2. 小数细化：在最佳整数 ±1.0 内按 precision（默认 0.1）步长搜索。
    返回 (best_shift, best_mse)；找不到任何有限 MSE（如无有效解析帧）返回 (0.0, inf)。
    """
    max_shift = min(total, n_info) * 0.1
    print(f"[shift-search] 搜索范围 ±{max_shift:.1f} 帧（min(视频 {total}, 文本 {n_info}) / 2），"
          f"精度 {precision}")

    int_lo = -int(math.floor(max_shift))
    int_hi = int(math.floor(max_shift))
    n_cand = int_hi - int_lo + 1

    # ── 整数粗搜（全范围）──
    best_shift, best_mse = 0.0, float("inf")
    for j, s in enumerate(range(int_lo, int_hi + 1)):
        m = mse_for_shift(float(s), valid_idx, checker_yaw_full,
                          imu_unwrapped_full, total, n_info)
        if m < best_mse:
            best_mse, best_shift = m, float(s)
        if progress and n_cand > 4000 and (j % 2000 == 0 or j == n_cand - 1):
            print(f"    [coarse] {j + 1}/{n_cand}  (best: shift={best_shift}, mse={best_mse:.6f})")

    # ── 小数细化：最佳整数 ±1.0（覆盖相邻单位区间）──
    anchor = best_shift
    steps = np.arange(-1.0, 1.0 + precision * 0.5, precision)
    for d in steps:
        if abs(d) < precision * 0.5:
            continue                    # 跳过整数本身（粗搜已评估）
        s = anchor + d
        if abs(s) > max_shift + 1e-9:
            continue
        m = mse_for_shift(s, valid_idx, checker_yaw_full,
                          imu_unwrapped_full, total, n_info)
        if m < best_mse:
            best_mse, best_shift = m, s

    if math.isinf(best_mse):
        print("[shift-search] 未找到含有效解析帧的公共段，回退 shift = 0")
        return 0.0, best_mse
    return best_shift, best_mse


# ==================== 主分析流程 ====================

def analyze(record_dir, mode, precision=0.1, show=False, output=None):
    # ── 定位视频与信息文件 ──
    videos = sorted(glob.glob(os.path.join(record_dir, "video_*")))
    infos = sorted(glob.glob(os.path.join(record_dir, "frame_info_*")))
    if not videos or not infos:
        raise RuntimeError(f"{record_dir} 中未找到 video_* / frame_info_* 文件")
    video_path, info_path = videos[0], infos[0]
    print(f"[record] video: {video_path}")
    print(f"[record] info : {info_path}")

    # ── 相机参数 + 信息文件 ──
    K, dist = load_camera_params(mode)
    idx, dt, ts, accepted, imu_yaw = parse_frame_info(info_path)
    n_info = len(idx)
    print(f"[info] {n_info} 行数据 (frame {idx[0]}..{idx[-1]})")

    # ── 逐帧解析棋盘格位姿 ──
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] 共 {total} 帧")

    n_ok = 0
    yaw_raw = [None] * total          # 每帧 cam 系 yaw（失败为 None）
    for i in range(total):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if ret:
            # solvePnP（ITERSOLVEPNP_ITERATIVE 为默认）→ rvec/tvec
            ret_pnp, rvec, tvec = cv2.solvePnP(_objp, corners, K, dist)
            if ret_pnp:
                yaw, pitch, roll = pnp_rvec_to_euler(rvec)
                yaw_raw[i] = -yaw
                n_ok += 1
        if (i + 1) % 100 == 0 or i == total - 1:
            print(f"  [progress] {i + 1}/{total}")
    cap.release()
    print(f"[pnp] 成功解析 {n_ok}/{total} 帧 ({100.0 * n_ok / total:.1f}%)")

    # ── 棋盘格 yaw：整段视频有效值先解缠绕，再前向填充（失败帧用前一帧，首帧兜底 0）──
    yaw_valid_full = np.array([v for v in yaw_raw if v is not None])
    if len(yaw_valid_full) > 0:
        yaw_unwrapped_full = np.unwrap(yaw_valid_full)
        yaw_raw_unwrapped = [None] * total
        it = iter(yaw_unwrapped_full)
        for i in range(total):
            if yaw_raw[i] is not None:
                yaw_raw_unwrapped[i] = next(it)
        checker_yaw_full = forward_fill(yaw_raw_unwrapped, fill_zero=0.0)
    else:
        checker_yaw_full = np.zeros(total)
    valid_full = np.array([v is not None for v in yaw_raw])
    valid_idx = np.nonzero(valid_full)[0]

    # ── imu yaw：整段文本先解缠绕 ──
    imu_unwrapped_full = np.unwrap(imu_yaw)

    # ── 自动搜索最优帧平移（视频帧 k ↔ 文本行 k - shift）──
    best_shift, search_mse = search_best_shift(
        valid_idx, checker_yaw_full, imu_unwrapped_full, total, n_info,
        precision=precision)
    print(f"[shift-search] 最优 shift = {best_shift:.2f} 帧（该值下有效帧误差 MSE = "
          f"{search_mse:.6f}）")

    # ── 最终公共段（与搜索使用同一套范围/插值规则）──
    k_start, k_end = common_segment_range(best_shift, total, n_info)
    n_plot = k_end - k_start
    if n_plot <= 0:
        raise RuntimeError(f"最优 shift 下公共段为空 (total={total}, n_info={n_info}, "
                           f"shift={best_shift})")
    print(f"[common] 公共段 {n_plot} 帧（视频帧 {k_start}..{k_end - 1}，shift={best_shift:.2f}）")

    checker_yaw = checker_yaw_full[k_start:k_end]
    imu_yaw_n = sample_imu(imu_unwrapped_full, best_shift, k_start, k_end)
    valid_mask = valid_full[k_start:k_end]

    # ── 平均 dt：计算范围（最终公共段对应文本行）内所有帧 dt 的均值 ──
    p_first = k_start - best_shift        # 公共段首帧对应的文本位置
    p_last = k_end - 1 - best_shift       # 公共段末帧对应的文本位置
    r0 = int(math.floor(p_first))
    r1 = int(math.ceil(p_last))
    avg_dt = float(np.mean(dt[r0:r1 + 1]))
    print(f"[dt] 平均 dt（计算范围：公共段对应文本行 {r0}..{r1}，共 {r1 - r0 + 1} 行）= "
          f"{avg_dt:.6f} s")

    # ── imu 整体平移：使「视频成功解析的点」上两者平均值相等 ──
    if np.any(valid_mask):
        yaw_offset = float(np.mean(checker_yaw[valid_mask]) - np.mean(imu_yaw_n[valid_mask]))
    else:
        yaw_offset = 0.0
    imu_yaw_shifted = imu_yaw_n + yaw_offset
    print(f"[offset] imu_euler_yaw 平移量 = {yaw_offset:.6f} rad "
          f"({math.degrees(yaw_offset):.4f} deg)")

    # ── 误差曲线 ──
    error = imu_yaw_shifted - checker_yaw

    # ── 绘图（x 轴为绝对视频帧索引；视频帧 k ↔ 文本位置 k - best_shift）──
    shift_time = best_shift * avg_dt       # 帧平移量换算为时间（秒）
    x = np.arange(k_start, k_start + n_plot)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]})

    ax1.plot(x, checker_yaw, color="tab:blue", lw=1.5,
             label=f"checkerboard yaw (PnP, cam frame, valid {n_ok}/{total})")
    ax1.plot(x, imu_yaw_shifted, color="tab:orange", lw=1.5,
             label=f"imu_euler_yaw (unwrapped + offset {yaw_offset:.4f} rad)")
    if np.any(valid_mask):
        ax1.scatter(x[valid_mask], checker_yaw[valid_mask],
                    s=12, color="tab:blue", alpha=0.6, zorder=3,
                    label="successfully parsed frames")
    ax1.set_ylabel("yaw (rad)")
    ax1.set_title(
        f"Checkerboard PnP yaw vs IMU yaw (unwrapped)\n"
        f"best frame shift = {best_shift:.2f} frames = {shift_time:.4f} s "
        f"(avg dt = {avg_dt:.5f} s)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, error, color="tab:red", lw=1.2,
             label="error = shifted_imu_yaw - checkerboard_yaw")
    if np.any(valid_mask):
        ax2.scatter(x[valid_mask], error[valid_mask], s=8, color="tab:red", alpha=0.5)
    ax2.axhline(0.0, color="k", lw=0.6)
    ax2.set_xlabel("frame index")
    ax2.set_ylabel("error (rad)")
    ax2.set_title(f"Error curve (mean abs {np.mean(np.abs(error)):.5f} rad, "
                  f"std {np.std(error):.5f} rad)")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if output is None:
        output = os.path.join(record_dir, "checkerboard_imu_yaw_comparison.png")
    fig.savefig(output, dpi=150)
    print(f"[out] 结果图已保存: {output}")
    if show:
        plt.show()
    plt.close(fig)

    # ── 汇总 ──
    print("[summary]")
    print(f"  valid parse frames : {int(np.sum(valid_mask))}/{n_plot}")
    print(f"  avg dt             : {avg_dt:.6f} s")
    print(f"  best frame shift   : {best_shift:.2f} frames")
    print(f"  shift * avg dt     : {shift_time:.6f} s")
    print(f"  search MSE (valid) : {search_mse:.6f} rad^2")
    print(f"  imu offset         : {yaw_offset:.6f} rad")
    print(f"  error mean         : {np.mean(error):.6f} rad")
    print(f"  error std          : {np.std(error):.6f} rad")
    print(f"  error mean |.|     : {np.mean(np.abs(error)):.6f} rad")
    return output


def main():
    parser = argparse.ArgumentParser(description="解析 Record 模式录制数据，对比棋盘格 PnP yaw 与 IMU yaw")
    parser.add_argument("record_dir", help="Record 会话目录（record_YYYYmmdd_HHMMSS，含 video_* 与 frame_info_*）")
    parser.add_argument("--mode", choices=["camera", "video"], default="camera",
                        help="config.yaml 中使用的相机参数段（默认 camera_mode）")
    parser.add_argument("--precision", type=float, default=0.1,
                        help="自动搜索 shift 的小数精度（默认 0.1 帧）")
    parser.add_argument("--show", action="store_true", help="额外交互显示图像")
    parser.add_argument("--output", default=None, help="结果图保存路径")
    args = parser.parse_args()

    if not os.path.isdir(args.record_dir):
        print(f"[error] 目录不存在: {args.record_dir}", file=sys.stderr)
        sys.exit(1)
    if not (0.0 < args.precision <= 1.0):
        print(f"[error] --precision 需在 (0, 1] 范围内: {args.precision}", file=sys.stderr)
        sys.exit(1)

    try:
        analyze(args.record_dir, args.mode, precision=args.precision,
                show=args.show, output=args.output)
    except Exception as e:  # noqa: BLE001
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
