#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""auto_launch/watchdog.py — 统一自瞄主程序看门狗

职责（只做一件事：让主程序一直活着，且是「真的在干活」地活着）：
  1. 按 auto_launch.conf 的 [launch] 段执行运行主程序的命令行（命令行内容编码在配置文件里，
     本脚本不含任何针对具体可执行文件/参数的内容）；
  2. 进程级监控：子进程（及其后代：launch_all.py + 两个推理进程 + 主程序）退出后按策略重启；
  3. 心跳级监控：主程序经 Unix 域套接字周期喂狗（见 include/common/Debug/WatchdogFeed.h），
     超时未喂狗即判定卡死（死锁/相机断流/流水线停摆），杀掉整个进程组后重启；
  4. 退避重启：连续快速失败时指数退避（上限 restart_backoff_max_sec），避免崩溃风暴；
  5. 干净退出：SIGTERM/SIGINT → 向进程组发 SIGINT → 宽限后 SIGKILL，并清理套接字/锁文件；
     SIGHUP → 重读配置文件并重启子进程（systemd 里对应 systemctl reload）。

用法：
  ./watchdog.py                     # 看门狗模式（systemd 服务用的就是这条）
  ./watchdog.py --run               # 不用看门狗，只按配置前台跑一次命令行（等价旧版 launch 脚本）
  ./watchdog.py --check             # 只检查配置/环境/可执行文件，不启动
  ./watchdog.py --self-test         # 用内置假程序跑通「心跳→卡死→杀掉→重启」全链路（无需相机）
  ./watchdog.py --print-command     # 打印将要在项目根目录执行的 shell 脚本
  常用覆盖：--config PATH  --command "..."  --no-heartbeat  --log-level debug
            --max-restarts N  --feed-timeout SEC  --connect-timeout SEC

注意：
  - 本脚本只用 Python 标准库（systemd 下用 /usr/bin/python3 即可，不依赖 venv）；
  - 同一时刻只允许一个看门狗实例（对 socket_path + ".lock" 加 flock，进程崩溃后自动释放）；
  - 退出码：0 = 正常停止（收到信号 / restart_policy=unexpected 下子进程正常退出）；
    非 0 = 启动/配置失败，或达到 max_restarts。
"""

import argparse
import configparser
import fcntl
import os
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from shutil import which as shutil_which

DEFAULT_CONFIG_NAME = "auto_launch.conf"

# ── 配置默认值（与 auto_launch.conf 中的注释保持一致）──
DEFAULTS = {
    "launch": {
        "command": "",
        "preamble": "",
        "shell": "/bin/bash",
        "project_root": "..",
        "restart_policy": "always",
    },
    "watchdog": {
        "heartbeat_enabled": "true",
        "socket_path": "/tmp/unified_auto_aim_watchdog.sock",
        "feed_timeout_sec": "20",
        "connect_timeout_sec": "150",
        "shutdown_grace_sec": "5",
        "restart_delay_sec": "3",
        "restart_backoff_max_sec": "60",
        "healthy_run_sec": "60",
        "max_restarts": "0",
        "feed_period_ms": "500",
        "stall_timeout_ms": "8000",
        "boot_grace_ms": "120000",
        "log_file": "logs/watchdog.log",
        "log_max_bytes": str(5 * 1024 * 1024),
        "log_level": "info",
        "status_log_sec": "60",
    },
    "service": {
        "name": "unified_auto_aim.service",
        "description": "Unified Auto-Aim Pipeline (auto_launch watchdog)",
        "after": "network-online.target",
        "wants": "network-online.target",
        "environment": "PYTHONUNBUFFERED=1",
        "restart_sec": "3",
        "timeout_stop_sec": "20",
    },
}

LOG_LEVELS = {"debug": 10, "info": 20, "warn": 30, "warning": 30, "error": 40}
HEARTBEAT_RE = re.compile(r"^UAP-WD/1\s+FEED\s+seq=(\d+)\s+frames=(\d+)\s*$")

# shell 元字符：命令不含这些字符时用 exec 直接替换 shell（信号直达、退出码即命令退出码）
_SHELL_META = set("|&;<>()$`\\\"'*?[]{}~#\n")


class ConfigError(Exception):
    """配置文件错误（缺项/取值非法/路径不存在）"""


def log_stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Logger:
    """同时写 stdout（systemd 下进 journal）与可选日志文件；文件超过上限时轮转为 .1"""

    def __init__(self, level="info", path=None, max_bytes=0):
        self.level = LOG_LEVELS.get(str(level).lower(), 20)
        self.path = path
        self.max_bytes = int(max_bytes or 0)
        self.fh = None
        self._open()

    def _open(self):
        if not self.path:
            return
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if self.max_bytes > 0 and os.path.exists(self.path) \
                    and os.path.getsize(self.path) >= self.max_bytes:
                os.replace(self.path, self.path + ".1")
            self.fh = open(self.path, "a", encoding="utf-8", buffering=1)
        except OSError as e:
            self.fh = None
            sys.stderr.write(f"[watchdog] 无法写日志文件 {self.path}: {e}（改为只输出到 stdout）\n")

    def close(self):
        if self.fh:
            try:
                self.fh.close()
            finally:
                self.fh = None

    def reconfigure(self, level=None, path=None, max_bytes=None):
        """重载配置时更新级别/文件（路径变化则重新打开）"""
        if level is not None:
            self.level = LOG_LEVELS.get(str(level).lower(), self.level)
        if path is not None and path != self.path:
            self.close()
            self.path = path
            self._open()
        if max_bytes is not None:
            self.max_bytes = max_bytes

    def _write(self, tag, msg):
        line = f"{log_stamp()} [{tag}] {msg}"
        print(line, flush=True)
        if self.fh:
            try:
                self.fh.write(line + "\n")
            except OSError:
                pass

    def debug(self, msg, *a):
        if self.level <= 10:
            self._write("debug", msg % a if a else msg)

    def info(self, msg, *a):
        if self.level <= 20:
            self._write("info", msg % a if a else msg)

    def warn(self, msg, *a):
        if self.level <= 30:
            self._write("warn", msg % a if a else msg)

    def error(self, msg, *a):
        self._write("error", msg % a if a else msg)


class Config:
    """auto_launch.conf 的解析结果（含路径解析与合法性校验）"""

    def __init__(self, path, cli=None):
        self.path = os.path.abspath(path)
        if not os.path.isfile(self.path):
            raise ConfigError(f"配置文件不存在: {self.path}")
        self.base_dir = os.path.dirname(self.path)

        # 允许行内注释（" #..." / " ;..."，需前置空白）——配置文件里大量使用行内注释说明取值
        parser = configparser.ConfigParser(interpolation=None,
                                           inline_comment_prefixes=("#", ";"))
        # 保留 key 大小写不敏感（默认已是），并允许值里出现 '%'
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                parser.read_file(f)
        except (configparser.Error, OSError) as e:
            raise ConfigError(f"配置文件解析失败: {e}")

        self._parser = parser
        cli = cli or {}
        self.raw = {sec: dict(DEFAULTS[sec]) for sec in DEFAULTS}
        for sec in DEFAULTS:
            if parser.has_section(sec):
                for key, val in parser.items(sec):
                    self.raw[sec][key] = val.strip()

        # ── [launch] ──
        self.project_root = os.path.abspath(
            os.path.join(self.base_dir, self.raw["launch"]["project_root"] or ".."))
        self.command = (cli.get("command") or self.raw["launch"]["command"]).strip()
        self.preamble = self.raw["launch"]["preamble"].strip()
        self.shell = self.raw["launch"]["shell"].strip() or "/bin/bash"
        self.restart_policy = self.raw["launch"]["restart_policy"].strip().lower()

        # ── [watchdog] ──
        w = self.raw["watchdog"]
        self.heartbeat_enabled = parse_bool(w["heartbeat_enabled"], "watchdog.heartbeat_enabled")
        if cli.get("no_heartbeat"):
            self.heartbeat_enabled = False
        elif cli.get("heartbeat"):
            self.heartbeat_enabled = True
        self.socket_path = w["socket_path"].strip()
        self.feed_timeout_sec = parse_float(
            cli.get("feed_timeout") or w["feed_timeout_sec"], "watchdog.feed_timeout_sec")
        self.connect_timeout_sec = parse_float(
            cli.get("connect_timeout") or w["connect_timeout_sec"], "watchdog.connect_timeout_sec")
        self.shutdown_grace_sec = parse_float(w["shutdown_grace_sec"], "watchdog.shutdown_grace_sec")
        self.restart_delay_sec = parse_float(w["restart_delay_sec"], "watchdog.restart_delay_sec")
        self.restart_backoff_max_sec = parse_float(
            w["restart_backoff_max_sec"], "watchdog.restart_backoff_max_sec")
        self.healthy_run_sec = parse_float(w["healthy_run_sec"], "watchdog.healthy_run_sec")
        self.max_restarts = parse_int(
            cli.get("max_restarts") or w["max_restarts"], "watchdog.max_restarts")
        self.feed_period_ms = parse_int(w["feed_period_ms"], "watchdog.feed_period_ms")
        self.stall_timeout_ms = parse_int(w["stall_timeout_ms"], "watchdog.stall_timeout_ms")
        self.boot_grace_ms = parse_int(w["boot_grace_ms"], "watchdog.boot_grace_ms")
        self.log_level = (cli.get("log_level") or w["log_level"]).strip().lower()
        self.log_max_bytes = parse_int(w["log_max_bytes"], "watchdog.log_max_bytes")
        self.status_log_sec = parse_float(w["status_log_sec"], "watchdog.status_log_sec")
        if self.status_log_sec <= 0:
            self.status_log_sec = float("inf")   # <=0 表示不做周期性状态日志
        log_file = w["log_file"].strip()
        if cli.get("no_log"):
            log_file = ""
        elif cli.get("log_file"):
            log_file = cli["log_file"]
        self.log_file = os.path.abspath(os.path.join(self.project_root, log_file)) if log_file else ""

        # ── [service] ──
        s = self.raw["service"]
        self.service_name = s["name"].strip()
        self.service_description = s["description"].strip()
        self.service_after = split_lines(s["after"])
        self.service_wants = split_lines(s["wants"])
        self.service_environment = split_lines(s["environment"])
        self.service_restart_sec = parse_int(s["restart_sec"], "service.restart_sec")
        self.service_timeout_stop_sec = parse_int(s["timeout_stop_sec"], "service.timeout_stop_sec")

        self.validate()

    # ── 校验 ──
    def validate(self):
        if not self.command:
            raise ConfigError("launch.command 为空：请在 auto_launch.conf 里写好运行主程序的命令行")
        if not os.path.isdir(self.project_root):
            raise ConfigError(f"launch.project_root 不是目录: {self.project_root}")
        if not os.path.isfile(self.shell):
            raise ConfigError(f"launch.shell 不存在: {self.shell}")
        if self.restart_policy not in ("always", "unexpected"):
            raise ConfigError("launch.restart_policy 只能是 always 或 unexpected，"
                              f"当前为 {self.restart_policy!r}")
        if self.log_level not in LOG_LEVELS:
            raise ConfigError(f"watchdog.log_level 非法: {self.log_level!r}")
        if self.heartbeat_enabled:
            if not self.socket_path.startswith("/"):
                raise ConfigError(f"watchdog.socket_path 必须是绝对路径: {self.socket_path!r}")
            if len(self.socket_path.encode()) > 107:
                raise ConfigError(f"watchdog.socket_path 过长（Unix 域套接字上限 107 字节）: "
                                  f"{self.socket_path}")
            if self.feed_timeout_sec <= 0:
                raise ConfigError("watchdog.feed_timeout_sec 必须 > 0")
            if self.connect_timeout_sec <= 0:
                raise ConfigError("watchdog.connect_timeout_sec 必须 > 0")
            if self.feed_period_ms <= 0:
                raise ConfigError("watchdog.feed_period_ms 必须 > 0")
        for name, val in (("shutdown_grace_sec", self.shutdown_grace_sec),
                          ("restart_delay_sec", self.restart_delay_sec),
                          ("restart_backoff_max_sec", self.restart_backoff_max_sec)):
            if val < 0:
                raise ConfigError(f"watchdog.{name} 不能为负: {val}")
        if self.max_restarts < 0:
            raise ConfigError(f"watchdog.max_restarts 不能为负: {self.max_restarts}")
        if not self.service_name:
            raise ConfigError("service.name 不能为空")

    # ── 派生量 ──
    @property
    def lock_path(self):
        return self.socket_path + ".lock"

    def child_env(self):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        for key in ("WATCHDOG_SOCKET_PATH", "WATCHDOG_FEED_PERIOD_MS",
                    "WATCHDOG_STALL_TIMEOUT_MS", "WATCHDOG_BOOT_GRACE_MS"):
            env.pop(key, None)
        if self.heartbeat_enabled:
            env["WATCHDOG_SOCKET_PATH"] = self.socket_path
            env["WATCHDOG_FEED_PERIOD_MS"] = str(self.feed_period_ms)
            env["WATCHDOG_STALL_TIMEOUT_MS"] = str(self.stall_timeout_ms)
            env["WATCHDOG_BOOT_GRACE_MS"] = str(self.boot_grace_ms)
        return env

    def shell_script(self):
        """把 preamble + command 组装成一段 shell 脚本（命令行内容全部来自配置文件）

        命令是「简单可执行命令」时用 exec 直接替换 shell（信号直达、退出码即命令退出码）；
        否则原样执行（复合语句 / shell 内建命令如 cd、export 不能 exec，照样在同一个
        进程组里，看门狗 killpg 一样能整组清干净）。
        """
        lines = ["set -e"]
        if self.preamble:
            lines.extend(self.preamble.splitlines())
        lines.append(("exec " if use_exec(self.command, self.project_root) else "") + self.command)
        return "\n".join(lines) + "\n"

    def summary(self):
        return "\n".join([
            f"配置文件   : {self.path}",
            f"项目根目录 : {self.project_root}",
            f"启动命令   : {self.command}",
            f"shell      : {self.shell}"
            + (f"（含 {len(self.preamble.splitlines())} 行 preamble）" if self.preamble else ""),
            f"重启策略   : {self.restart_policy}",
            "喂狗通道   : "
            + (f"开启 → {self.socket_path}（超时 {self.feed_timeout_sec:g}s，"
               f"首次连接 {self.connect_timeout_sec:g}s，"
               f"帧门控 {self.stall_timeout_ms}ms，启动宽限 {self.boot_grace_ms}ms）"
               if self.heartbeat_enabled else "关闭（只按进程存活判定）"),
            f"退避重启   : 基础 {self.restart_delay_sec:g}s，上限 {self.restart_backoff_max_sec:g}s，"
            f"稳定判定 {self.healthy_run_sec:g}s"
            + ("，不限次数" if self.max_restarts == 0 else f"，最多 {self.max_restarts} 次"),
            f"日志       : {self.log_file or 'stdout（journal）'}"
            f"（级别 {self.log_level}）",
        ])


def parse_bool(text, what):
    val = str(text).strip().lower()
    if val in ("1", "true", "yes", "on", "y"):
        return True
    if val in ("0", "false", "no", "off", "n", ""):
        return False
    raise ConfigError(f"{what} 需要 true/false，当前为 {text!r}")


def parse_int(text, what):
    try:
        return int(str(text).strip())
    except ValueError:
        raise ConfigError(f"{what} 需要整数，当前为 {text!r}")


def parse_float(text, what):
    try:
        return float(str(text).strip())
    except ValueError:
        raise ConfigError(f"{what} 需要数值，当前为 {text!r}")


def split_lines(text):
    """多行配置值 → 非空行列表（去掉行内注释）"""
    out = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        out.append(line)
    return out


def is_simple_command(cmd):
    return not any(ch in _SHELL_META for ch in cmd)


def use_exec(cmd, project_root):
    """能否用 exec 替换 shell：命令无 shell 元字符且首词确实是个可执行文件

    为什么要判断：`exec cd x` / `exec export A=1` 这类 shell 内建命令会直接失败，
    而 `cd x && ./bin/y` 这种复合语句 exec 语义也不对——这些情况原样交给 shell 执行。
    """
    if not is_simple_command(cmd):
        return False
    parts = cmd.split()
    if not parts:
        return False
    first = parts[0]
    if "/" in first:
        path = first if os.path.isabs(first) else os.path.join(project_root, first)
        return os.path.isfile(path) and os.access(path, os.X_OK)
    return shutil_which(first) is not None


class Watchdog:
    """看门狗主体：起子进程 → 等首次心跳 → 监控（进程存活 + 心跳）→ 退出/超时则重启"""

    def __init__(self, cfg, log, args):
        self.cfg = cfg
        self.log = log
        self.args = args
        self.child = None            # subprocess.Popen
        self.sock = None             # 心跳用的 Unix 域数据报套接字
        self.lock_fd = None
        self.stop_requested = False
        self.reload_requested = False
        self.started_at = 0.0        # 本实例启动时刻（用于日志）
        self.spawn_time = 0.0        # 当前子进程启动时刻
        self.last_feed = None        # 最近一次收到心跳的时刻
        self.heartbeat_stat = ""     # 最近一次心跳内容（日志用）
        self.restarts = 0            # 已重启次数
        self.failures = 0            # 连续快速失败次数（退避用）

    # ────────────────────────── 信号 ──────────────────────────
    def install_signal_handlers(self):
        signal.signal(signal.SIGTERM, self._on_stop_signal)
        signal.signal(signal.SIGINT, self._on_stop_signal)
        signal.signal(signal.SIGHUP, self._on_reload_signal)

    def _on_stop_signal(self, signum, _frame):
        # 只置位，实际收尾在主循环里做（信号处理器里不做复杂操作）
        self.stop_requested = True

    def _on_reload_signal(self, signum, _frame):
        self.reload_requested = True

    # ────────────────────────── 资源 ──────────────────────────
    def acquire_lock(self):
        """flock 独占锁：防止两个看门狗同时拉起两套程序"""
        path = self.cfg.lock_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            self.lock_fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as e:
            # 典型场景：systemd 服务以 rm1 运行过，锁文件属主是 rm1 且权限 0644，
            # 现在以另一个用户手动跑就会 EACCES —— 这里给出可操作的提示
            raise ConfigError(
                f"无法打开锁文件 {path}: {e}。若已有以其它用户运行的看门狗/服务，请先停掉它"
                "（systemctl status <服务名>），或改用其它 socket_path（--config 指向不同配置）")
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.lock_fd)
            self.lock_fd = None
            raise ConfigError(
                f"已有看门狗实例在运行（锁文件 {path} 被占用）。"
                "如需确认：ps -ef | grep watchdog.py；如为残留：删除该锁文件")

    def open_socket(self):
        """建立心跳监听套接字（AF_UNIX + SOCK_DGRAM：无连接状态，主程序/看门狗各自重启都不影响对方）"""
        if not self.cfg.heartbeat_enabled:
            return
        path = self.cfg.socket_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            try:
                mode = os.stat(path).st_mode
                if not stat.S_ISSOCK(mode):
                    raise ConfigError(f"{path} 已存在且不是套接字文件，拒绝覆盖")
                os.unlink(path)
            except OSError as e:
                raise ConfigError(f"无法清理残留套接字 {path}: {e}")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            sock.bind(path)
        except OSError as e:
            sock.close()
            raise ConfigError(f"绑定心跳套接字失败 {path}: {e}")
        os.chmod(path, 0o666)   # 允许同机其它用户（如手动运行的调试进程）喂狗
        sock.settimeout(0.2)
        self.sock = sock
        self.log.info("心跳套接字已监听: %s", path)

    def release(self):
        """关闭套接字/锁文件（幂等）"""
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        if self.cfg.heartbeat_enabled and os.path.exists(self.cfg.socket_path):
            try:
                os.unlink(self.cfg.socket_path)
            except OSError as e:
                self.log.debug("删除套接字失败 %s: %s", self.cfg.socket_path, e)
        if self.lock_fd is not None:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(self.lock_fd)
            self.lock_fd = None

    # ────────────────────────── 子进程 ──────────────────────────
    def spawn(self):
        script = self.cfg.shell_script()
        self.log.info("启动: %s -c %s", self.cfg.shell, shlex.quote(script))
        self.log.debug("工作目录: %s", self.cfg.project_root)
        self.child = subprocess.Popen(
            [self.cfg.shell, "-c", script],
            cwd=self.cfg.project_root,
            env=self.cfg.child_env(),
            preexec_fn=os.setsid,   # 自成进程组：重启/退出时可整组 SIGINT→SIGKILL
        )
        self.spawn_time = time.time()
        self.last_feed = None
        self.heartbeat_stat = ""
        self.log.info("子进程已启动: pid=%d (pgid=%d)", self.child.pid, self.child.pid)

    def kill_child(self, reason, signum=signal.SIGINT):
        """整组停止子进程：SIGINT → 宽限 → SIGKILL。子进程已退出时也会清掉残留后代。"""
        if self.child is None:
            return
        pid = self.child.pid
        pgid = pid
        if self.child.poll() is None:
            self.log.warn("停止进程组 %d（%s）：先发 %s", pgid, reason, signal.Signals(signum).name)
            self._signal_group(pgid, signum)
            try:
                self.child.wait(timeout=self.cfg.shutdown_grace_sec)
                self.log.info("子进程 %d 已退出（退出码 %s）", pid, self.child.returncode)
                self._signal_group(pgid, signal.SIGKILL)   # 兜底清理未退出的后代
                self.child = None
                return
            except subprocess.TimeoutExpired:
                self.log.warn("宽限 %gs 内未退出，升级为 SIGKILL", self.cfg.shutdown_grace_sec)
        self._signal_group(pgid, signal.SIGKILL)
        try:
            self.child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.log.error("子进程 %d 无法回收（僵尸残留）", pid)
        else:
            self.log.info("进程组 %d 已强制终止", pgid)
        self.child = None

    def _signal_group(self, pgid, signum):
        try:
            os.killpg(pgid, signum)
        except ProcessLookupError:
            pass
        except OSError as e:
            self.log.debug("killpg(%d, %d) 失败: %s", pgid, signum, e)

    # ────────────────────────── 心跳 ──────────────────────────
    def drain_heartbeats(self):
        """收干套接字缓冲区里的心跳；返回是否收到过数据"""
        got = False
        if not self.sock:
            return got
        while True:
            try:
                data, _addr = self.sock.recvfrom(4096)
            except socket.timeout:
                break
            except OSError as e:
                self.log.debug("recvfrom 失败: %s", e)
                break
            if not data:
                continue
            self.last_feed = time.time()
            got = True
            text = data.decode("utf-8", "replace").strip()
            m = HEARTBEAT_RE.match(text)
            self.heartbeat_stat = f"frames={m.group(2)}" if m else text[:64]
        return got

    # ────────────────────────── 监控 ──────────────────────────
    def monitor(self):
        """监控当前子进程，返回结束原因：
        ('stopped', None)      —— 收到停止信号
        ('reload', None)       —— 收到 SIGHUP
        ('exited', returncode) —— 子进程退出
        ('no_heartbeat', None) —— 起不来：connect_timeout 内没有首次喂狗
        ('feed_timeout', None) —— 卡死：feed_timeout 内没有喂狗
        """
        next_status_log = time.time() + self.cfg.status_log_sec
        while True:
            if self.stop_requested:
                return ("stopped", None)
            if self.reload_requested:
                return ("reload", None)

            rc = self.child.poll()
            if rc is not None:
                return ("exited", rc)

            if self.cfg.heartbeat_enabled:
                self.drain_heartbeats()
                now = time.time()
                if self.last_feed is None:
                    if now - self.spawn_time > self.cfg.connect_timeout_sec:
                        return ("no_heartbeat", None)
                elif now - self.last_feed > self.cfg.feed_timeout_sec:
                    self.log.warn("距上次喂狗 %.1fs（> %.1fs）：判定主程序卡死",
                                  now - self.last_feed, self.cfg.feed_timeout_sec)
                    return ("feed_timeout", None)

            now = time.time()
            if now >= next_status_log:
                next_status_log = now + self.cfg.status_log_sec
                self.log_status(now)

            time.sleep(0.1)

    def log_status(self, now):
        run = now - self.spawn_time
        if self.cfg.heartbeat_enabled:
            feed = "尚未收到" if self.last_feed is None else f"{now - self.last_feed:.1f}s 前"
            self.log.info("运行中: pid=%s 已运行 %.0fs，最近心跳 %s %s",
                          self.child.pid if self.child else "-", run, feed, self.heartbeat_stat)
        else:
            self.log.info("运行中: pid=%s 已运行 %.0fs（未启用心跳）",
                          self.child.pid if self.child else "-", run)

    # ────────────────────────── 主循环 ──────────────────────────
    def run(self):
        self.started_at = time.time()
        self.acquire_lock()
        self.open_socket()

        while not self.stop_requested:
            if self.reload_requested:
                self.reload_requested = False
                self.reload_config()

            self.spawn()
            reason, rc = self.monitor()

            if reason == "stopped":
                self.kill_child("收到停止信号")
                break

            if reason == "reload":
                self.reload_requested = False
                self.kill_child("配置重载")
                self.reload_config()
                continue

            # ── 子进程结束：先整组清干净，再决定是否重启 ──
            if reason == "exited":
                self.log.warn("子进程退出: 退出码 %s", format_returncode(rc))
                self.kill_child("子进程已退出，清理残留后代")
            else:
                self.kill_child("未喂狗（%s）" % ("启动超时" if reason == "no_heartbeat" else "心跳超时"))

            ran = time.time() - self.spawn_time
            if ran >= self.cfg.healthy_run_sec:
                self.failures = 0

            if self._should_stop_after(reason, rc):
                break

            if self.cfg.max_restarts and self.restarts >= self.cfg.max_restarts:
                self.log.error("已达 max_restarts=%d，停止重启", self.cfg.max_restarts)
                return 2

            delay = min(self.cfg.restart_delay_sec * (2 ** self.failures),
                        self.cfg.restart_backoff_max_sec)
            self.failures += 1
            self.restarts += 1
            self.log.info("第 %d 次重启，%.1fs 后重试", self.restarts, delay)
            if self._sleep_interruptible(delay):
                self.kill_child("收到停止信号")
                break

        self.log.info("看门狗退出（累计重启 %d 次，运行 %.0fs）",
                      self.restarts, time.time() - self.started_at)
        return 0

    def _should_stop_after(self, reason, rc):
        """restart_policy=unexpected 时，子进程正常退出（退出码 0）不再重启"""
        if self.cfg.restart_policy != "unexpected":
            return False
        if reason == "exited" and rc == 0:
            self.log.info("restart_policy=unexpected 且子进程正常退出，不再重启")
            return True
        return False

    def _sleep_interruptible(self, seconds):
        """可被停止信号打断的睡眠；返回 True 表示期间收到停止请求"""
        end = time.time() + seconds
        while time.time() < end:
            if self.stop_requested:
                return True
            time.sleep(min(0.1, max(0.0, end - time.time())))
        return False

    def reload_config(self):
        """重读配置文件（SIGHUP / systemctl reload）：套接字相关参数变化时重建监听"""
        self.log.info("重载配置: %s", self.cfg.path)
        try:
            new_cfg = Config(self.cfg.path, self._cli_overrides())
            new_cfg.validate()
        except ConfigError as e:
            self.log.error("配置重载失败，继续使用旧配置: %s", e)
            return
        old_socket = (self.cfg.heartbeat_enabled, self.cfg.socket_path)
        old_log = (self.cfg.log_file, self.cfg.log_max_bytes)
        self.cfg = new_cfg
        if (new_cfg.heartbeat_enabled, new_cfg.socket_path) != old_socket:
            if self.sock:
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None
            self.open_socket()
        if (new_cfg.log_file, new_cfg.log_max_bytes) != old_log:
            self.log.reconfigure(path=new_cfg.log_file, max_bytes=new_cfg.log_max_bytes)
        self.log.reconfigure(level=new_cfg.log_level)
        self.log.info("配置已重载")

    def _cli_overrides(self):
        return {
            "command": self.args.command,
            "no_heartbeat": self.args.no_heartbeat,
            "heartbeat": self.args.heartbeat,
            "feed_timeout": self.args.feed_timeout,
            "connect_timeout": self.args.connect_timeout,
            "max_restarts": self.args.max_restarts,
            "log_file": self.args.log_file,
            "no_log": self.args.no_log,
            "log_level": self.args.log_level,
        }


def format_returncode(rc):
    if rc is None:
        return "unknown"
    if rc < 0:
        try:
            return f"{rc} ({signal.Signals(-rc).name})"
        except ValueError:
            return str(rc)
    return str(rc)


def run_foreground(cfg, log, args):
    """--run：不看门狗，只在项目根目录前台执行配置里的命令行（等价旧版 launch 脚本）"""
    script = cfg.shell_script()
    if args.print_command:
        print(f"# 项目根目录: {cfg.project_root}")
        print(f"# shell: {cfg.shell}")
        print(script, end="")
        return 0
    log.info("前台运行（无看门狗）: %s -c %s", cfg.shell, shlex.quote(script))
    log.debug("工作目录: %s", cfg.project_root)
    proc = subprocess.Popen([cfg.shell, "-c", script], cwd=cfg.project_root,
                            env=cfg.child_env(), preexec_fn=os.setsid)

    def _fwd(signum, _frame):
        # 前台模式：Ctrl+C / kill 直接转发给整个进程组，不做重启
        try:
            os.killpg(proc.pid, signum)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, _fwd)
    signal.signal(signal.SIGINT, _fwd)
    rc = proc.wait()
    log.info("运行结束: 退出码 %s", format_returncode(rc))
    return 0 if rc == 0 else 1


def check(cfg, log, args):
    """--check：不启动程序，只做静态检查（配置文件/路径/命令行/日志目录）"""
    ok = True
    print("配置检查:")
    print(indent(cfg.summary()))

    print("\n检查项:")
    checks = []

    shell_ok = os.path.isfile(cfg.shell) and os.access(cfg.shell, os.X_OK)
    checks.append(("shell 可执行", shell_ok, cfg.shell))

    first = cfg.command.split()[0] if cfg.command.split() else ""
    if is_simple_command(cfg.command) and first:
        path = os.path.join(cfg.project_root, first) if "/" in first else None
        found = (path and os.access(path, os.X_OK)) or bool(shutil_which(first))
        checks.append((f"命令首词可执行（{first}）", bool(found),
                       path if path else "在 PATH 中查找"))
    else:
        checks.append(("命令为复合 shell 语句（跳过首词检查）", True, cfg.command[:48] + "…"
                       if len(cfg.command) > 48 else cfg.command))

    sock_dir = os.path.dirname(cfg.socket_path) if cfg.heartbeat_enabled else None
    if sock_dir:
        writable = os.path.isdir(sock_dir) and os.access(sock_dir, os.W_OK)
        checks.append(("心跳套接字目录可写", writable, sock_dir))

    if cfg.log_file:
        log_dir = os.path.dirname(cfg.log_file)
        writable = os.access(log_dir, os.W_OK) if os.path.isdir(log_dir) else \
            os.access(os.path.dirname(log_dir), os.W_OK)
        checks.append(("日志目录可写", writable, log_dir))

    bin_dir = os.path.join(cfg.project_root, "bin")
    checks.append(("bin/ 目录存在", os.path.isdir(bin_dir), bin_dir))

    for name, passed, detail in checks:
        print(f"  [{'OK' if passed else '!!'}] {name}: {detail}")
        ok = ok and passed

    if not cfg.heartbeat_enabled:
        print("\n提示: heartbeat_enabled=false，看门狗只按进程存活重启，"
              "无法发现主程序卡死。")
    else:
        print("\n提示: 心跳需要主程序本次构建包含 common/Debug/WatchdogFeed"
              "（否则首次喂狗会超时，看门狗将反复重启）。")
    print("\n结论:", "全部通过" if ok else "存在问题（见上面 [!!] 项）")
    return 0 if ok else 1


def indent(text, prefix="  "):
    return "\n".join(prefix + line for line in text.splitlines())


SELF_TEST_FEEDER = '''#!/usr/bin/env python3
# watchdog.py --self-test 使用的假“主程序”：喂狗 N 秒后停喂（模拟卡死），进程继续活着
import os, signal, socket, sys, time
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))   # 被看门狗 SIGINT 时安静退出
path = os.environ["WATCHDOG_SOCKET_PATH"]
fed = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
s.connect(path)
print("[selftest-feeder] 已连接看门狗，开始喂狗 %.0fs" % fed, flush=True)
t0, i = time.time(), 0
while True:
    if time.time() - t0 < fed:
        i += 1
        try:
            s.send(("UAP-WD/1 FEED seq=%d frames=%d\\n" % (i, i)).encode())
        except OSError:
            time.sleep(0.3)
            try:
                s.connect(path)
            except OSError:
                pass
    time.sleep(0.2)
'''


def self_test(cfg, log, args):
    """--self-test：不需要相机/模型，用内置假程序验证整条链路（约 10s）"""
    fd, feeder = tempfile.mkstemp(prefix="uap-watchdog-selftest-", suffix=".py")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(SELF_TEST_FEEDER)
    try:
        child_cmd = " ".join([shlex.quote(sys.executable), shlex.quote(feeder), "2"])
        argv = [sys.executable, os.path.abspath(__file__),
                "--config", cfg.path,
                "--command", child_cmd,
                "--heartbeat",                 # 自检必须验心跳，无视配置里的开关
                "--feed-timeout", "2",
                "--connect-timeout", "6",
                "--max-restarts", "1",
                "--no-log",
                "--log-level", "info"]
        print("[watchdog] self-test：用假程序验证 心跳 → 卡死判定 → 杀进程组 → 重启")
        print("[watchdog] 执行: " + " ".join(shlex.quote(a) for a in argv[1:]))
        print("-" * 72)
        env = dict(os.environ, UAP_WATCHDOG_SELFTEST_CHILD="1")
        proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, env=env)
        out = proc.stdout or ""
        print(out, end="")
        print("-" * 72)

        checks = [
            ("心跳套接字建立并收到喂狗", "[selftest-feeder] 已连接看门狗" in out),
            ("喂狗超时判定为卡死", "判定主程序卡死" in out),
            ("整组 SIGINT/SIGKILL 清理", ("已退出" in out) or ("已强制终止" in out)),
            ("退避后重启子进程", "第 1 次重启" in out),
            ("max_restarts 生效并干净退出", proc.returncode == 2),
            ("无残留：套接字已清理", not os.path.exists(cfg.socket_path)),
        ]
        ok = True
        for name, passed in checks:
            print(f"  [{'OK' if passed else '!!'}] {name}")
            ok = ok and passed
        print("\nself-test 结论:", "全部通过" if ok else "存在失败项")
        if not ok:
            print("提示: 若第一项就失败，检查 socket_path 所在目录权限；"
                  "若日志出现“已有看门狗实例在运行”，先停掉在跑的看门狗（或 systemd 服务）再自检。")
        return 0 if ok else 1
    finally:
        try:
            os.unlink(feeder)
        except OSError:
            pass


def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="watchdog.py",
        description="统一自瞄主程序看门狗（配置见同目录 auto_launch.conf）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help=f"配置文件路径（默认 {DEFAULT_CONFIG_NAME}，与脚本同目录）")
    p.add_argument("--run", action="store_true",
                   help="不看门狗，只在项目根目录前台运行配置中的命令行")
    p.add_argument("--check", action="store_true", help="只检查配置与环境，不启动")
    p.add_argument("--self-test", action="store_true",
                   help="用内置假程序验证心跳/超时/杀进程组/重启链路（约 10s，不需要相机）")
    p.add_argument("--print-command", action="store_true",
                   help="打印将执行的 shell 脚本（配合 --run/--check 使用，或单独使用）")
    p.add_argument("--command", default=None, help="覆盖配置中的运行命令行（调试用）")
    p.add_argument("--no-heartbeat", action="store_true", help="关闭心跳判定，只看进程存活")
    p.add_argument("--heartbeat", action="store_true",
                   help="强制开启心跳判定（覆盖配置，调试/自检用）")
    p.add_argument("--feed-timeout", default=None, help="覆盖 feed_timeout_sec")
    p.add_argument("--connect-timeout", default=None, help="覆盖 connect_timeout_sec")
    p.add_argument("--max-restarts", default=None, help="覆盖 max_restarts（调试用）")
    p.add_argument("--log-file", default=None, help="覆盖日志文件路径（相对项目根）")
    p.add_argument("--no-log", action="store_true", help="不写日志文件，只输出到 stdout")
    p.add_argument("--log-level", default=None, help="覆盖日志级别 debug/info/warn/error")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    conf_path = args.config or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           DEFAULT_CONFIG_NAME)
    cli = {
        "command": args.command,
        "no_heartbeat": args.no_heartbeat,
        "heartbeat": args.heartbeat,
        "feed_timeout": args.feed_timeout,
        "connect_timeout": args.connect_timeout,
        "max_restarts": args.max_restarts,
        "log_file": args.log_file,
        "no_log": args.no_log,
        "log_level": args.log_level,
    }

    try:
        cfg = Config(conf_path, cli)
    except ConfigError as e:
        sys.stderr.write(f"[watchdog] 配置错误: {e}\n")
        return 2

    # 日志级别：--check/--self-test/--print-command 时也照常输出到 stdout，但不建日志文件
    log = Logger(cfg.log_level, cfg.log_file if not (args.check or args.self_test) else None,
                 cfg.log_max_bytes)
    try:
        if args.print_command and not args.run:
            print(f"# 项目根目录: {cfg.project_root}")
            print(f"# shell: {cfg.shell}")
            print(cfg.shell_script(), end="")
            return 0
        if args.check:
            return check(cfg, log, args)
        if args.self_test:
            return self_test(cfg, log, args)
        if args.run:
            return run_foreground(cfg, log, args)

        log.info("=" * 72)
        log.info("统一自瞄看门狗启动")
        log.info("\n%s", cfg.summary())
        wd = Watchdog(cfg, log, args)
        wd.install_signal_handlers()
        try:
            return wd.run()
        finally:
            wd.release()
    except KeyboardInterrupt:
        log.warn("收到中断信号，退出")
        return 0
    except ConfigError as e:
        log.error("%s", e)
        return 2
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
