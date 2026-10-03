#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""auto_launch/install_service.py — 把看门狗装成 systemd 系统服务（开机自启）

做的事情很少，但每一条都为了「机器人上电即工作、且出问题能查」：
  1. 从同目录 auto_launch.conf 读取 [launch]/[watchdog]/[service]，生成 unit 文件
     （ExecStart 固定为 /usr/bin/python3 <本项目>/auto_launch/watchdog.py --config ...）；
  2. 安装前先跑一次 watchdog.py --check 做静态自检（配置/路径/权限），失败则中止；
  3. sudo install 到 /etc/systemd/system/，daemon-reload，enable --now；
  4. 支持 status / restart / stop / start / logs / print / uninstall。

用法：
  ./install_service.py                      # 安装并启动（默认动作，需要 sudo）
  ./install_service.py --dry-run            # 只打印 unit 文件与将要执行的命令，不动系统
  ./install_service.py --user rm1 install   # 指定服务运行用户（默认：调用者的用户）
  ./install_service.py print                # 打印 unit 文件
  ./install_service.py status               # systemctl status + is-enabled
  ./install_service.py logs                 # journalctl 最近 200 行
  ./install_service.py uninstall            # 停用并删除 unit（保留项目文件与日志）

说明：
  - 服务是**系统级**的（/etc/systemd/system），开箱即用、不依赖用户登录；卸载用 uninstall；
  - 服务运行用户默认是调用本脚本的用户（sudo 下取 SUDO_USER），因为它需要访问相机/串口；
  - 生成 unit 的 Restart 策略跟随配置的 launch.restart_policy：
      always     → Restart=always（子进程正常退出也重启，相机自启动用）
      unexpected → Restart=on-failure
"""

import argparse
import os
import pwd
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from watchdog import Config, ConfigError, DEFAULT_CONFIG_NAME  # noqa: E402

UNIT_DIR = "/etc/systemd/system"
DEFAULT_PYTHON = "/usr/bin/python3"


def log(msg):
    print(f"[install_service] {msg}", flush=True)


def is_root():
    return os.geteuid() == 0


def sudo_prefix(user=None):
    """需要提权时返回 ['sudo']（已是 root 则为空）"""
    return [] if is_root() else ["sudo"]


def detect_user(explicit=None):
    """服务运行用户：显式指定 > sudo 调用者 > 当前用户"""
    if explicit:
        return explicit
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user and sudo_user != "root":
        return sudo_user
    if is_root():
        return None   # root 直接安装但没说是哪个用户：交给调用方报错提示
    return pwd.getpwuid(os.getuid()).pw_name


def pick_python(explicit=None):
    if explicit:
        return explicit
    if os.path.exists(DEFAULT_PYTHON):
        return DEFAULT_PYTHON
    return sys.executable


def render_unit(cfg, user, python, watchdog_path):
    """生成 unit 文本。所有路径写绝对路径，服务不依赖登录环境。"""
    restart = "always" if cfg.restart_policy == "always" else "on-failure"
    env_lines = "".join(f"Environment={kv}\n" for kv in cfg.service_environment)
    after = " ".join(cfg.service_after) if cfg.service_after else "-"
    wants = " ".join(cfg.service_wants) if cfg.service_wants else "-"
    gid = primary_gid(user)
    group_line = f"Group={gid}\n" if gid is not None else ""

    return f"""# 由 auto_launch/install_service.py 生成，请勿手工编辑（改 auto_launch.conf 后重新安装）
# 配置文件: {cfg.path}
# 运行用户: {user}
[Unit]
Description={cfg.service_description}
Documentation=file:{os.path.join(HERE, 'README.md')}
After={after}
Wants={wants}
# 不做启动频率限制：看门狗自身退出后由 systemd 继续拉起（重启风暴由看门狗内部退避兜住）
StartLimitIntervalSec=0

[Service]
Type=simple
User={user}
{group_line}WorkingDirectory={cfg.project_root}
{env_lines}ExecStart={python} {watchdog_path} --config {cfg.path}
ExecReload=/bin/kill -HUP $MAINPID
Restart={restart}
RestartSec={cfg.service_restart_sec}
KillMode=control-group
KillSignal=SIGTERM
TimeoutStopSec={cfg.service_timeout_stop_sec}
# 需要更高调度优先级时自行打开（负值需要 CAP_SYS_NICE，可能因权限启动失败）
#Nice=-5

[Install]
WantedBy=multi-user.target
"""


def primary_gid(user):
    """用户主组 gid（不存在则 None）"""
    try:
        return pwd.getpwnam(user).pw_gid
    except KeyError:
        return None


def safe_user_exists(user):
    try:
        pwd.getpwnam(user)
        return True
    except KeyError:
        return False


def unit_path(name):
    return os.path.join(UNIT_DIR, name)


def run(cmd, dry=False, check=True):
    log("$ " + " ".join(cmd))
    if dry:
        return 0
    try:
        return subprocess.call(cmd)
    except FileNotFoundError as e:
        if check:
            log(f"命令不存在: {e}")
        return 127


def require(rc, what):
    if rc != 0:
        raise SystemExit(f"[install_service] {what} 失败（退出码 {rc}）")


def precheck(cfg, python, watchdog_path, dry):
    """安装前自检：配置文件 + 脚本存在 + --check 通过"""
    if not os.path.isfile(watchdog_path):
        raise SystemExit(f"[install_service] 找不到看门狗脚本: {watchdog_path}")
    if not os.path.isfile(cfg.path):
        raise SystemExit(f"[install_service] 找不到配置文件: {cfg.path}")
    if dry:
        log("dry-run：跳过 watchdog.py --check")
        return
    log("安装前自检: watchdog.py --check")
    rc = subprocess.call([python, watchdog_path, "--config", cfg.path, "--check"])
    if rc != 0:
        raise SystemExit("[install_service] 自检未通过，已中止安装"
                         "（确认无误可用 --force 跳过）")


def cmd_print(cfg, user, python, watchdog_path, args):
    text = render_unit(cfg, user, python, watchdog_path)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        log(f"unit 已写入 {args.output}")
    else:
        print(text, end="")
    return 0


def cmd_install(cfg, user, python, watchdog_path, args):
    unit = render_unit(cfg, user, python, watchdog_path)
    dry = args.dry_run

    if not dry and not args.force:
        precheck(cfg, python, watchdog_path, dry)

    log(f"服务名   : {cfg.service_name}")
    log(f"运行用户 : {user}")
    log(f"解释器   : {python}")
    log(f"项目根   : {cfg.project_root}")
    log(f"看门狗   : {watchdog_path} --config {cfg.path}")

    if args.output:
        return cmd_print(cfg, user, python, watchdog_path, args)

    fd, tmp = tempfile.mkstemp(prefix="uap-service-", suffix=".service")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(unit)
    try:
        run(sudo_prefix() + ["install", "-m", "0644", "-o", "root", "-g", "root",
                             tmp, unit_path(cfg.service_name)], dry=dry)
        run(sudo_prefix() + ["systemctl", "daemon-reload"], dry=dry)
        if args.no_enable:
            log("已安装（未 enable/start，按 --no-enable）")
        else:
            run(sudo_prefix() + ["systemctl", "enable", "--now", cfg.service_name], dry=dry)
            log("已安装并启动。查看状态: ./install_service.py status；查看日志: ./install_service.py logs")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return 0


def cmd_uninstall(cfg, user, python, watchdog_path, args):
    dry = args.dry_run
    run(sudo_prefix() + ["systemctl", "disable", "--now", cfg.service_name], dry=dry)
    run(sudo_prefix() + ["rm", "-f", unit_path(cfg.service_name)], dry=dry)
    run(sudo_prefix() + ["systemctl", "daemon-reload"], dry=dry)
    run(sudo_prefix() + ["systemctl", "reset-failed", cfg.service_name], dry=dry)
    log(f"已卸载 {cfg.service_name}（项目文件与日志保留；日志文件: {cfg.log_file or 'journal'})")
    return 0


def cmd_status(cfg, user, python, watchdog_path, args):
    run(sudo_prefix() + ["systemctl", "status", "--no-pager", "--full", cfg.service_name],
        dry=args.dry_run)
    run(sudo_prefix() + ["systemctl", "is-enabled", cfg.service_name], dry=args.dry_run)
    if cfg.log_file and os.path.exists(cfg.log_file):
        log(f"看门狗日志文件: {cfg.log_file}")
    return 0


def cmd_logs(cfg, user, python, watchdog_path, args):
    cmd = sudo_prefix() + ["journalctl", "-u", cfg.service_name, "-n",
                           str(args.lines), "--no-pager", "-o", "short-iso"]
    return run(cmd, dry=args.dry_run)


def cmd_simple(action):
    def _run(cfg, user, python, watchdog_path, args):
        return run(sudo_prefix() + ["systemctl", action, cfg.service_name], dry=args.dry_run)
    return _run


def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="install_service.py",
        description="把 auto_launch 看门狗安装为 systemd 系统服务（开机自启）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", nargs="?", default="install",
                   choices=["install", "uninstall", "status", "logs", "start", "stop",
                            "restart", "print"],
                   help="默认 install")
    p.add_argument("--config", default=os.path.join(HERE, DEFAULT_CONFIG_NAME),
                   help="auto_launch.conf 路径")
    p.add_argument("--user", default=None,
                   help="服务运行用户（默认 sudo 调用者/当前用户；需能访问相机与串口）")
    p.add_argument("--python", default=None, help=f"解释器（默认 {DEFAULT_PYTHON}）")
    p.add_argument("--output", default=None, help="只把 unit 写到该路径，不安装")
    p.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令，不做任何修改")
    p.add_argument("--no-enable", action="store_true", help="install 时只安装、不 enable/start")
    p.add_argument("--force", action="store_true", help="跳过安装前的 watchdog.py --check 自检")
    p.add_argument("--lines", type=int, default=200, help="logs 动作显示的行数")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    try:
        cfg = Config(args.config)
    except ConfigError as e:
        raise SystemExit(f"[install_service] 配置错误: {e}")

    user = detect_user(args.user)
    if not user:
        raise SystemExit("[install_service] 当前以 root 运行且未指定运行用户，"
                         "请加 --user <用户名>（例如 --user rm1）")
    if not safe_user_exists(user):
        raise SystemExit(f"[install_service] 用户不存在: {user}")

    python = pick_python(args.python)
    if not os.path.isfile(python):
        raise SystemExit(f"[install_service] 解释器不存在: {python}")
    watchdog_path = os.path.join(HERE, "watchdog.py")

    # 项目目录对服务用户可读可执行（否则服务起不来，提前提示）
    if not os.access(cfg.project_root, os.R_OK | os.X_OK) and not args.dry_run:
        log(f"警告: {cfg.project_root} 对当前用户不可读/不可进入，服务可能启动失败")

    if not is_root() and not args.dry_run and not shutil.which("sudo"):
        raise SystemExit("[install_service] 需要 root 权限但没有 sudo，"
                         "请用 root 直接运行并在 install 时加 --user <用户名>")

    if args.action == "install":
        os.chmod(watchdog_path, 0o755)   # 允许直接 ./watchdog.py 手动运行
        return cmd_install(cfg, user, python, watchdog_path, args)
    if args.action == "print":
        return cmd_print(cfg, user, python, watchdog_path, args)
    if args.action == "uninstall":
        return cmd_uninstall(cfg, user, python, watchdog_path, args)
    if args.action == "status":
        return cmd_status(cfg, user, python, watchdog_path, args)
    if args.action == "logs":
        return cmd_logs(cfg, user, python, watchdog_path, args)
    return cmd_simple(args.action)(cfg, user, python, watchdog_path, args)


if __name__ == "__main__":
    sys.exit(main())
