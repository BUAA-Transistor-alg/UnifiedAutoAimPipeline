#!/usr/bin/env python3
# launch_all.py — 一键启动：两个推理进程 + 主程序；脚本退出时全部关闭
#
# 用法：
#   python3 launch_all.py <unified_auto_aim 的全部参数...>
#   例如：
#     python3 launch_all.py --pipeline outpost --input video \
#       -v "/path/demo.mkv" --output visualize
#     python3 launch_all.py --pipeline outpost --input camera --output visualize+gimbal
#
# 行为：
#   - 启动前先清理本程序用到的共享内存段与具名信号量（key 取自 config 的
#     shm_key），避免上一次运行遗留的 IPC 影响本次运行；
#   - 先启动 outpost_infer_process / power_rune_infer_process（读 config 各自的
#     shm_key），等两者输出 "[InferShmServer] ready" 后再启动主程序；
#   - 主程序退出（按 'q' / 视频结束等）或收到 Ctrl+C 时，向所有子进程发 SIGINT，
#     等待退出，超时（5s）后 SIGKILL，然后再次清理共享内存/信号量，保证脚本
#     关闭后没有任何残留进程或 IPC 影响其他程序；
#   - 推理进程未就绪（编译失败/模型缺失）时报错退出，不会启动主程序。

import os
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(ROOT, "bin")
CONFIG = os.path.join(ROOT, "config", "config.yaml")

INFER_PROCS = [
    ("outpost_infer_process", "outpost"),
    ("power_rune_infer_process", "power_rune"),
]

READY_MARK = "InferShmServer] ready"
KILL_GRACE_SECONDS = 5.0

# Ctrl+C / SIGTERM：置位后主循环退出并关闭全部子进程
g_stop = False


def _on_signal(signum, frame):
    global g_stop
    print(f"\n[launch_all] 收到信号 {signum}，正在关闭全部进程 ...", flush=True)
    g_stop = True


def read_shm_keys():
    """从 config/config.yaml 读取所有 shm_key（outpost / power_rune 各一个）。"""
    keys = []
    try:
        with open(CONFIG, "r", encoding="utf-8") as f:
            text = f.read()
        keys = [int(k) for k in re.findall(r"^\s*shm_key:\s*(\d+)", text, re.MULTILINE)]
    except Exception as e:
        print(f"[launch_all] 读取 config 失败: {e}", file=sys.stderr)
    return keys


def cleanup_shm(keys):
    """删除本程序用到的共享内存段（ipcrm -M key）与具名信号量（/dev/shm 文件）。"""
    for key in keys:
        os.system(f"ipcrm -M {key} >/dev/null 2>&1")
        for sem in (f"/dev/shm/sem.uap_infer_req_{key}",
                    f"/dev/shm/sem.uap_infer_resp_{key}"):
            try:
                os.remove(sem)
            except FileNotFoundError:
                pass
    if keys:
        print(f"[launch_all] 已清理共享内存/信号量 (keys={keys})", flush=True)


def check_binaries():
    missing = [name for name, _ in INFER_PROCS] + ["unified_auto_aim"]
    for name in missing:
        path = os.path.join(BIN, name)
        if not os.path.isfile(path):
            print(f"[launch_all] 缺少可执行文件: {path}（请先运行 ./build.sh）", file=sys.stderr)
            return False
    return True


def forward_output(proc, name, ready_event):
    """逐行转发子进程输出；检测就绪标记。"""
    try:
        for line in proc.stdout:
            text = line.rstrip()
            print(f"[{name}] {text}", flush=True)
            if not ready_event.is_set() and READY_MARK in text:
                ready_event.set()
    except Exception:
        pass


def stop_process(proc, name):
    """优雅停止子进程：SIGINT → 等待 → SIGKILL。"""
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=KILL_GRACE_SECONDS)
        print(f"[launch_all] {name} 已退出", flush=True)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.kill()
        proc.wait(timeout=KILL_GRACE_SECONDS)
        print(f"[launch_all] {name} 超时，已 SIGKILL", flush=True)
    except Exception:
        pass


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    if not check_binaries():
        return 1

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # ── 0. 清理上一次遗留的共享内存/信号量（key 取自 config）──
    shm_keys = read_shm_keys()
    cleanup_shm(shm_keys)

    # ── 1. 启动两个推理进程 ──
    infer_procs = []
    ready_events = {}
    for name, tag in INFER_PROCS:
        path = os.path.join(BIN, name)
        print(f"[launch_all] 启动 {name} ...", flush=True)
        proc = subprocess.Popen(
            [path],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        ev = threading.Event()
        ready_events[name] = ev
        infer_procs.append(proc)
        threading.Thread(target=forward_output, args=(proc, tag, ev), daemon=True).start()

    # ── 2. 等待全部推理进程就绪（模型编译需数秒~十几秒）──
    deadline = time.time() + 120.0
    for name, _ in INFER_PROCS:
        ev = ready_events[name]
        while not ev.is_set():
            if time.time() > deadline:
                print(f"[launch_all] 等待 {name} 就绪超时（120s），中止。"
                      f"请检查模型路径/config 后重试。", file=sys.stderr)
                for p, n in zip(infer_procs, [x[0] for x in INFER_PROCS]):
                    stop_process(p, n)
                return 1
            # 进程提前退出（编译失败）则报错
            for p, n in zip(infer_procs, [x[0] for x in INFER_PROCS]):
                if p.poll() is not None and not ready_events[n].is_set():
                    print(f"[launch_all] {n} 异常退出（exit={p.returncode}），中止。", file=sys.stderr)
                    for q, m in zip(infer_procs, [x[0] for x in INFER_PROCS]):
                        stop_process(q, m)
                    return 1
            time.sleep(0.2)
    print("[launch_all] 两个推理进程已就绪。", flush=True)

    # ── 3. 启动主程序（参数透传）──
    app_args = sys.argv[1:]
    print(f"[launch_all] 启动 unified_auto_aim {' '.join(app_args)}", flush=True)
    app = subprocess.Popen(
        [os.path.join(BIN, "unified_auto_aim")] + app_args,
        cwd=ROOT,
    )

    # ── 4. 等待主程序退出（轮询，便于及时响应 Ctrl+C / SIGTERM）──
    try:
        while app.poll() is None and not g_stop:
            time.sleep(0.1)
    finally:
        stop_process(app, "unified_auto_aim")
        for p, n in zip(infer_procs, [x[0] for x in INFER_PROCS]):
            stop_process(p, n)
        # 关闭后再清理共享内存/信号量，避免遗留影响其他程序
        cleanup_shm(shm_keys)

    print("[launch_all] 全部进程已关闭。", flush=True)
    return app.returncode if app.returncode is not None else 0


if __name__ == "__main__":
    sys.exit(main())
