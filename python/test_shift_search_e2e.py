#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端测试：生成合成棋盘格视频 + frame_info txt，运行 analyze_record.py，
验证自动 shift 搜索（含小数）、平均 dt 与绘图输出。

用法: python3 python/test_shift_search_e2e.py
"""
import glob
import math
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_record as ar  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(ROOT, "recordings", "synth_e2e_test")

W, H = 1920, 1080
K = np.array([[650.0, 0.0, 960.0],
              [0.0, 650.0, 540.0],
              [0.0, 0.0, 1.0]], dtype=np.float64)
DIST = np.zeros((5, 1), dtype=np.float64)
N_FRAMES = 120
FPS = 30.0
DELAY = 1.5           # IMU 比视频晚 1.5 帧
YAW_OFFSET = 0.7      # 常数偏移（均值对齐应消除）

# 棋盘格：12x9 方格 → (11,8) 内角点；方格边长 60px
SQUARE_PX = 60
NCOLS, NROWS = 12, 9
BOARD_W, BOARD_H = NCOLS * SQUARE_PX, NROWS * SQUARE_PX
MARGIN = 80
IMG_W, IMG_H = BOARD_W + 2 * MARGIN, BOARD_H + 2 * MARGIN


def make_board():
    img = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
    for r in range(NROWS):
        for c in range(NCOLS):
            if (r + c) % 2 == 0:
                x0, y0 = MARGIN + c * SQUARE_PX, MARGIN + r * SQUARE_PX
                img[y0:y0 + SQUARE_PX, x0:x0 + SQUARE_PX] = 0
    return img


def rot_y(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def homography(yaw, d=2000.0):
    """棋盘绕自身中心（相机系 (0,0,d)）竖直轴旋转 yaw 的单应矩阵（棋盘像素系 → 相机像素系）。"""
    R = rot_y(yaw)
    Xc = np.array([0.0, 0.0, d])
    return K @ np.column_stack([R[:, :2], Xc])


def main():
    if os.path.isdir(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)

    board = make_board()
    k = np.arange(N_FRAMES)
    yaw_true = 0.18 * np.sin(2 * np.pi * 1.3 * k / N_FRAMES) \
        + 0.06 * np.sin(2 * np.pi * 4.7 * k / N_FRAMES + 1.0)

    video_path = os.path.join(WORK, "video_synth.mkv")
    vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for y in yaw_true:
        Hm = homography(y)
        frame = cv2.warpPerspective(board, Hm, (W, H))
        vw.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    vw.release()
    assert cv2.VideoCapture(video_path).isOpened()

    # ── 检测棋盘格 yaw（测试脚手架，逻辑与脚本相同）──
    cap = cv2.VideoCapture(video_path)
    yaw_raw = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, ar.CHESSBOARD_SIZE, None)
        if ret:
            ret_pnp, rvec, tvec = cv2.solvePnP(ar._objp, corners, K, DIST)
            if ret_pnp:
                yaw, _, _ = ar.pnp_rvec_to_euler(rvec)
                yaw_raw.append(-yaw)
            else:
                yaw_raw.append(None)
        else:
            yaw_raw.append(None)
    cap.release()
    assert len(yaw_raw) == N_FRAMES
    n_ok = sum(1 for v in yaw_raw if v is not None)
    print(f"[test] PnP 检测成功 {n_ok}/{N_FRAMES}")
    assert n_ok >= N_FRAMES * 0.9, "棋盘格检测成功率过低，无法进行有效测试"

    vis = ar.forward_fill(yaw_raw, fill_zero=0.0)

    # ── 构造 frame_info：imu 行 i = 视频帧 i 处的 yaw 延迟 DELAY 帧 + 常数偏移 ──
    #     imu[i] = interp(i - DELAY, arange(N), vis) + YAW_OFFSET
    imu = np.interp(np.arange(N_FRAMES) - DELAY, np.arange(N_FRAMES), vis) + YAW_OFFSET
    info_path = os.path.join(WORK, "frame_info_synth.txt")
    ts0 = 1700000000.0
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("# UnifiedAutoAim frame_info v2\n")
        f.write("# columns: frame_index dt_s timestamp_s accepted"
                " imu_euler_yaw imu_euler_pitch imu_euler_roll yaw_pos pitch_angle"
                " chassis_yaw chassis_pitch chassis_roll chassis_x chassis_y chassis_z\n")
        rng = np.random.default_rng(42)
        dt = np.full(N_FRAMES, 1.0 / FPS)
        dt[1:] += rng.uniform(-0.002, 0.002, N_FRAMES - 1)   # 抖动
        dt[0] = 0.0
        for i in range(N_FRAMES):
            f.write(f"{i} {dt[i]:.9f} {ts0 + i / FPS:.9f} 1 "
                    f"{imu[i]:.12f} 0 0 0 0 0 0 0 0 0 0\n")

    # ── 运行脚本 ──
    out_png = os.path.join(WORK, "result.png")
    env = dict(os.environ, MPLCONFIGDIR="/tmp/mplcfg")
    r = subprocess.run([sys.executable, os.path.join(HERE, "analyze_record.py"),
                        WORK, "--mode", "video", "--output", out_png],
                       capture_output=True, text=True, env=env)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit(1)

    # ── 校验 ──
    assert os.path.isfile(out_png), "未生成结果图"
    best_shift = None
    for line in r.stdout.splitlines():
        if "最优 shift" in line:
            best_shift = float(line.split("=")[1].split("帧")[0])
    assert best_shift is not None
    print(f"[test] 期望 shift ≈ {-DELAY}，脚本得到 {best_shift}")
    assert abs(best_shift - (-DELAY)) <= 0.11, f"shift 恢复失败: {best_shift}"

    avg_dt = None
    for line in r.stdout.splitlines():
        if line.strip().startswith("avg dt"):
            avg_dt = float(line.split(":")[1].split("s")[0])
    assert avg_dt is not None
    print(f"[test] avg_dt = {avg_dt:.6f}（期望 ≈ {1.0 / FPS:.6f}）")
    assert abs(avg_dt - 1.0 / FPS) < 0.01

    shift_time = best_shift * avg_dt
    print(f"[test] shift * avg_dt = {shift_time:.6f} s")
    print("E2E TEST PASSED")


if __name__ == "__main__":
    main()
