# auto_launch — 主程序看门狗 + 开机自启

让 `unified_auto_aim` 在机器人上「上电即跑、崩了自动重启、卡死也能自动重启」。

```
auto_launch/
├── auto_launch.conf     # 唯一配置：运行主程序的命令行 + 看门狗参数 + 服务参数
├── watchdog.py          # 看门狗本体（只用标准库，systemd 下用 /usr/bin/python3 运行）
├── install_service.py   # 安装/卸载/查看 systemd 系统服务（开机自启）
└── README.md            # 本文件
```

配套的 C++ 侧改动（喂狗通道，默认关闭、不影响平时手动运行）：

| 文件 | 作用 |
| --- | --- |
| `include/common/Debug/WatchdogFeed.h` | 喂狗通道接口与说明 |
| `src/common/Debug/WatchdogFeed.cpp` | 上报线程 + Unix 域数据报发送实现 |
| `src/main.cpp` | `start()` / 处理线程 `tick()` / 退出时 `stop()`；并补上 `SIGTERM` 处理 |
| `CMakeLists.txt` | 把 `WatchdogFeed.cpp` 加进 `COMMON_SOURCES` |

---

## 1. 快速上手

```bash
cd <项目根>

# ① 只检查配置/路径/权限（不启动任何程序）
auto_launch/watchdog.py --check

# ①' 动态自检：用内置假程序跑通「心跳 → 卡死判定 → 杀进程组 → 重启」（约 10s，不需要相机）
auto_launch/watchdog.py --self-test

# ② 不用看门狗，直接按配置前台跑一次（等价于旧的 launch 脚本，Ctrl+C 关闭）
auto_launch/watchdog.py --run

# ③ 看门狗模式（前台观察；systemd 服务跑的就是这条）
auto_launch/watchdog.py

# ④ 安装成 systemd 系统服务并立即启动（需要 sudo）
auto_launch/install_service.py

# ⑤ 查看状态 / 日志 / 重启 / 卸载
auto_launch/install_service.py status
auto_launch/install_service.py logs
auto_launch/install_service.py restart
auto_launch/install_service.py uninstall
```

> 装机流程建议：先 `--check`，再 `--self-test`，最后 `install_service.py`（它安装前会再跑一次 `--check`）。
> `--self-test` 会临时用同一个 `socket_path`，所以**先停掉在跑的看门狗/服务**，否则会报「已有看门狗实例在运行」。

安装服务后，改配置不必重装：

```bash
sudo systemctl reload unified_auto_aim.service   # 重读 auto_launch.conf 并重启子进程
sudo systemctl restart unified_auto_aim.service
```

---

## 2. 运行主程序的命令行写在哪

只写在一个地方：`auto_launch/auto_launch.conf` 的 `[launch] command`。
`watchdog.py` / `install_service.py` 里都没有任何针对具体可执行文件、参数的内容。

```ini
[launch]
command = python3 launch_all.py --pipeline armor --input camera --output gimbal
shell = /bin/bash
preamble =
project_root = ..
restart_policy = always
```

- `command`：在 `project_root` 下、用 `shell -c` 执行的完整命令行（`launch_all.py` 会先起两个推理
  进程并等就绪，再起主程序；与 config 里 `infer_process_lazy: false` 的机器一致）。
- `preamble`：命令前要先跑的 shell 片段（多行，续行需缩进），例如 `source /opt/ros/humble/setup.bash`。
- `project_root`：相对本文件所在目录，默认 `..` 即项目根；也是子进程的工作目录。
- `restart_policy`：`always`（任何退出都重启，相机自启动用）/ `unexpected`（仅异常退出重启）。

想临时换个命令、不动配置：

```bash
auto_launch/watchdog.py --command "./bin/unified_auto_aim --pipeline power_rune --input camera --output gimbal"
auto_launch/watchdog.py --print-command    # 只看将要执行的 shell 脚本
```

---

## 3. 看门狗怎么判活

两层，缺一不可：

1. **进程级**：子进程（`bash → launch_all.py → 两个推理进程 + 主程序`，同一进程组）退出后按
   `restart_policy` 重启；重启前先 `SIGINT` 整组、宽限 `shutdown_grace_sec` 后 `SIGKILL`，
   保证共享内存/信号量/串口不留残余。
2. **心跳级**：主程序通过 `common/Debug/WatchdogFeed` 向 `socket_path`（Unix 域**数据报**套接字）
   周期上报 `UAP-WD/1 FEED seq=<n> frames=<n>`。
   - 启动后 `connect_timeout_sec` 内收不到第一帧心跳 → 判定起不来，重启；
   - 之后超过 `feed_timeout_sec` 收不到心跳 → 判定卡死，重启。

主程序只在**流水线真正产出有效帧**时 `tick()`，所以下面这些情况都会被识别为卡死并重启：
相机断流、处理线程死锁、各阶段队列停摆、启动阶段卡在不产出第一帧的地方（超过 `boot_grace_ms`）。
关闭帧门控（`stall_timeout_ms <= 0`）则退化为「只要进程活着就一直喂」。

喂狗是**可选**的：未设置 `WATCHDOG_SOCKET_PATH` 时 `WatchdogFeed` 完全不启动线程，
平时 `launch_all.py` / 直接跑 `bin/unified_auto_aim` 的行为和加此功能之前一样。

主程序侧可用的环境变量（看门狗会按配置自动注入）：

| 变量 | 含义 | 配置项 |
| --- | --- | --- |
| `WATCHDOG_SOCKET_PATH` | 看门狗套接字；不设置=关闭喂狗 | `watchdog.socket_path` |
| `WATCHDOG_FEED_PERIOD_MS` | 上报周期 | `watchdog.feed_period_ms` |
| `WATCHDOG_STALL_TIMEOUT_MS` | 多久没新帧就停止上报；`<=0` 关闭帧门控 | `watchdog.stall_timeout_ms` |
| `WATCHDOG_BOOT_GRACE_MS` | 多久没第一帧就停止上报；`<=0` 不限制 | `watchdog.boot_grace_ms` |
| `WATCHDOG_FEED_VERBOSE=1` | 打印连接/断开细节 | 手动设置 |

看门狗不会因为子进程退出码是 0 就放松（`restart_policy=always` 时视频播完也会重启）；
连续快速失败按 `restart_delay_sec × 2ⁿ` 退避，上限 `restart_backoff_max_sec`，
连续运行满 `healthy_run_sec` 后退避计数清零（避免「崩溃风暴」把 CPU 打满）。

---

## 4. 配置项速查（`auto_launch.conf`）

| 段 | 键 | 说明 |
| --- | --- | --- |
| launch | `command` | **运行主程序的命令行**（唯一来源） |
| launch | `preamble` / `shell` / `project_root` | 启动前 shell 片段 / 解释器 / 工作目录 |
| launch | `restart_policy` | `always` \| `unexpected` |
| watchdog | `heartbeat_enabled` | 是否启用心跳判定（旧二进制没有喂狗通道时设 false） |
| watchdog | `socket_path` | 心跳套接字（绝对路径，≤107 字节） |
| watchdog | `feed_timeout_sec` | 喂狗超时（默认 20s） |
| watchdog | `connect_timeout_sec` | 等待首次喂狗上限（默认 150s，要覆盖推理进程编译模型的时间） |
| watchdog | `shutdown_grace_sec` | SIGINT → SIGKILL 的宽限 |
| watchdog | `restart_delay_sec` / `restart_backoff_max_sec` / `healthy_run_sec` | 退避重启参数 |
| watchdog | `max_restarts` | 0=无限（调试可设小值） |
| watchdog | `feed_period_ms` / `stall_timeout_ms` / `boot_grace_ms` | 注入主程序的喂狗参数 |
| watchdog | `log_file` / `log_max_bytes` / `log_level` / `status_log_sec` | 日志（默认写 `logs/watchdog.log`，超过上限轮转为 `.1`） |
| service | `name` / `description` | 服务名与描述 |
| service | `after` / `wants` / `environment` | unit 依赖与环境变量 |
| service | `restart_sec` / `timeout_stop_sec` | systemd 重启间隔 / 停止宽限 |

---

## 5. systemd 服务

`install_service.py` 生成的 `/etc/systemd/system/unified_auto_aim.service` 形如：

```ini
[Unit]
Description=Unified Auto-Aim Pipeline (auto_launch watchdog)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0            # 不做启动频率限制，重启交给看门狗退避

[Service]
Type=simple
User=<安装时的用户>                  # 需要访问相机/串口，默认 sudo 调用者
WorkingDirectory=<项目根>
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 <项目根>/auto_launch/watchdog.py --config <项目根>/auto_launch/auto_launch.conf
ExecReload=/bin/kill -HUP $MAINPID  # systemctl reload = 重读配置并重启子进程
Restart=always                      # 跟随 launch.restart_policy
RestartSec=3
KillMode=control-group              # 停服务时整组清理，避免推理进程残留
KillSignal=SIGTERM
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
```

要点：

- **系统级**服务，开机自启，不依赖用户登录；
- `User=` 默认取 `sudo` 的调用者（也可 `--user <用户名>` 指定），必须是平时能读相机/串口的那个用户；
- 安装前会自动跑一次 `watchdog.py --check` 自检，失败即中止（`--force` 可跳过）；
- `--dry-run` 只打印将要执行的命令；`print` 只打印 unit；`--output <路径>` 只写文件不安装。

---

## 6. 常见问题

**Q: 日志在哪？**
`logs/watchdog.log`（配置文件项 `watchdog.log_file`，留空则只进 journal）。
systemd 下也可以 `install_service.py logs`，或 `journalctl -u unified_auto_aim -f`。
主程序自身的输出被看门狗原样转发到同一处。

**Q: 启动后反复「未喂狗（启动超时）」？**
说明主程序没在 `connect_timeout_sec` 内喂狗。依次检查：
① 主程序是不是**本次改动之前编译的旧二进制**（没有 `WatchdogFeed`）——重新 `./build.sh`，
或先把 `heartbeat_enabled` 设为 `false`；
② 推理进程编译模型太慢（把 `connect_timeout_sec` 调大，或确认模型路径正确）；
③ 用 `--log-level debug` 看子进程输出。

**Q: `systemctl status` 显示 `masked`（服务明明装过）？**
两种成因，`install_service.py status` 会直接指出是哪一种：
① 目标是**指向 `/dev/null` 的符号链接**（`systemctl mask` 的产物，可能在 `/etc/systemd/system/`
   或 `/run/systemd/system/`）；
② 目标是 **0 字节文件** —— systemd 对空 unit 文件同样按 `masked` 处理。最常见的来源是
   **装完/改完立刻断电**：ext4 的 delayed allocation 还没把数据块写盘，只落了 inode 元数据
   （权限、属主、mtime 都在），日志恢复后文件内容为空。真机踩过一次：14:19:17 装完、
   14:19:21 掉电，重启后 unit 就是 0 字节。

修复（新版安装器会自动做这三步，并在装完后 `sync` + 回读校验内容）：
```bash
sudo systemctl unmask unified_auto_aim.service
sudo rm -f /etc/systemd/system/unified_auto_aim.service /run/systemd/system/unified_auto_aim.service
sudo systemctl daemon-reload
auto_launch/install_service.py            # 重新安装
```
**通用教训：改完配置 / 装完服务、构建完二进制，断电前敲一次 `sync`。**
机器人经常直接断电，这一条能省掉很多"文件莫名变空/内容丢失"的排查。

**Q: 报告「已有看门狗实例在运行」？**
防止两套程序同时抢相机/串口。用 `ps -ef | grep watchdog.py` 确认；确认没有残留后可删除
`<socket_path>.lock` 再启动。

**Q: 手动调试时不想被看门狗杀？**
平时手动跑 `launch_all.py` 本来就不看门狗（没有 `WATCHDOG_SOCKET_PATH`）。
若确实要用看门狗跑交互模式，加 `--no-heartbeat`，或把 `stall_timeout_ms` 设为 `0`。

**Q: `--input camera` 但相机没插，会怎样？**
主程序启动失败退出 → 看门狗按退避重启（间隔逐步涨到 `restart_backoff_max_sec`），
日志里能看到每次退出码，不会打满 CPU。

**Q: 停止/卸载**
```bash
auto_launch/install_service.py stop        # 停服务（整组清理，含推理进程）
auto_launch/install_service.py uninstall   # 停用并删除 unit（保留项目文件与日志）
```

---

## 7. 相对上一版（rm2026 `auto_launch/`）的改动

| 项 | rm2026 版 | 本版 |
| --- | --- | --- |
| 运行命令行 | 硬编码在 `auto_aim_launch.py` 里 | 抽到 `auto_launch.conf` 的 `launch.command`，看门狗/安装脚本都不含具体参数 |
| 判活 | 只靠心跳超时（主程序须自己实现喂狗） | 进程级 + 心跳级两层；心跳由 `WatchdogFeed` 提供，按「帧是否推进」喂狗，能发现相机断流/死锁 |
| 心跳套接字 | `SOCK_STREAM` + accept，重启后需重连握手 | `SOCK_DGRAM`，无连接状态，双方各自重启互不影响 |
| 单实例 | 无保护（两个看门狗会拉起两套程序） | `flock` 独占锁 |
| 重启策略 | 固定等待 5s 无限重启 | `always/unexpected` + 指数退避 + 稳定计数清零 + `max_restarts` |
| 退出 | 只处理 SIGINT/SIGTERM，靠 `killpg` | 同样的进程组语义，另加 `SIGHUP` 重载配置、整组 `SIGINT→SIGKILL` 宽限 |
| 日志 | 只有 stdout | stdout(journal) + 按大小轮转的日志文件 + 周期状态行（含心跳帧数） |
| 安装 | 生成 unit 后 `sudo mv`，路径/用户名靠字符串替换 | 脚本自动解析项目路径/用户，安装前自检，支持 `install/uninstall/status/logs/print/--dry-run/--no-enable` |
| SIGTERM | 主程序只处理 SIGINT（systemd 停止会直接杀） | `src/main.cpp` 补上 `SIGTERM`，走正常收尾 |
