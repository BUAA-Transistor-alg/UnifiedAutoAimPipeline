#!/usr/bin/env python3
# launch_all.py — 一键启动：两个推理进程 + 主程序；脚本退出时全部关闭
#
# 用法：
#   python3 launch_all.py <unified_auto_aim 的全部参数...>
#   例如：
#     python3 launch_all.py --pipeline armor --input video \
#       -v "/path/demo.mkv" --output visualize
#     python3 launch_all.py --pipeline armor --input camera --output visualize+gimbal
#
# 行为：
#   - 启动前先清理本程序用到的共享内存段与具名信号量（key 取自 config 的
#     shm_key），避免上一次运行遗留的 IPC 影响本次运行；
#   - 推理进程启动策略由 config common.infer_process_lazy 决定：
#       false（默认）：先启动 armor_infer_process / power_rune_infer_process
#         （读 config 各自的 shm_key），等两者输出 "[InferShmServer] ready" 后再启动
#         主程序；无推理任务时推理进程在后台闲置；
#       true：不预启动推理进程，由主程序（unified_auto_aim）按需启动当前流水线
#         所需进程，并在切换流水线时立即关闭不再需要的进程（见 InferProcessManager）；
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

# config 读取规则 v2：机器配置选择器 config/selector.yaml → 机器配置文件
# config/robots/<active_config>.yaml（与 C++ RobotConfig::instance() 一致）
sys.path.insert(0, os.path.join(ROOT, "python"))
from path_resolver import resolve_machine_config_path  # noqa: E402

INFER_PROCS = [
    ("armor_infer_process", "armor"),
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


def get_config_path():
    """返回当前机器配置文件路径（config/selector.yaml → config/robots/<active_config>.yaml）；
    解析失败时打印错误并返回 None（与旧版读取失败行为一致）。"""
    try:
        return resolve_machine_config_path()
    except Exception as e:
        print(f"[launch_all] 解析机器配置失败: {e}", file=sys.stderr)
        return None


def read_shm_keys():
    """从机器配置文件读取所有 shm_key（armor / power_rune 各一个）。"""
    keys = []
    cfg = get_config_path()
    if not cfg:
        return keys
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            text = f.read()
        keys = [int(k) for k in re.findall(r"^\s*shm_key:\s*(\d+)", text, re.MULTILINE)]
    except Exception as e:
        print(f"[launch_all] 读取 config 失败: {e}", file=sys.stderr)
    return keys


def read_lazy_infer_mode():
    """读取机器配置文件 common.infer_process_lazy（true 时推理进程由主程序按需启停，
    本脚本不再预启动推理进程）。"""
    cfg = get_config_path()
    if not cfg:
        return False
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            text = f.read()
        m = re.search(r"^\s*infer_process_lazy:\s*(\w+)", text, re.MULTILINE)
        if m:
            return m.group(1).strip().lower() in ("true", "yes", "1", "on")
    except Exception as e:
        print(f"[launch_all] 读取 config 失败: {e}", file=sys.stderr)
    return False


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

    # 推理进程启动策略（config common.infer_process_lazy）：
    #   false（默认）：本脚本预启动全部推理进程并等待就绪；
    #   true：推理进程由主程序按需启停，本脚本只启动主程序。
    lazy_infer = read_lazy_infer_mode()

    # ── 1+2. 启动并等待推理进程就绪（非 lazy 模式）──
    infer_procs = []
    if lazy_infer:
        print("[launch_all] common.infer_process_lazy=true：推理进程由主程序按需"
              "启动/关闭（仅启动当前流水线所需进程），本脚本不再预启动推理进程。",
              flush=True)
    else:
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

        # 等待全部推理进程就绪（模型编译需数秒~十几秒）
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
