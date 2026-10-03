#include "common/Debug/WatchdogFeed.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cstring>

// 文件内日志去重标志（只有上报线程访问）
namespace {
bool g_connect_failed_logged = false;   // 首次连接失败已打印
bool g_connected_logged      = false;   // 连接成功已打印
bool g_path_warned           = false;   // 套接字路径过长已打印
}  // namespace

WatchdogFeed& WatchdogFeed::instance() {
    static WatchdogFeed inst;   // C++11 起函数局部静态初始化线程安全
    return inst;
}

WatchdogFeed::~WatchdogFeed() {
    stop();
}

long long WatchdogFeed::nowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

long long WatchdogFeed::envMs(const char* name, long long fallback_ms) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback_ms;
    errno = 0;
    char* end = nullptr;
    const long long x = std::strtoll(v, &end, 10);
    if (errno != 0 || end == v) {
        std::cerr << "[WatchdogFeed] 环境变量 " << name << "=\"" << v
                  << "\" 非法，按默认值 " << fallback_ms << " 处理" << std::endl;
        return fallback_ms;
    }
    return x;
}

void WatchdogFeed::start() {
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) return;   // 幂等

    const char* path = std::getenv("WATCHDOG_SOCKET_PATH");
    if (!path || !*path) {
        // 平时手动运行（launch_all.py / 直接跑 bin/unified_auto_aim）时走这里：
        // 完全不启动线程，行为与加本功能之前一致
        std::cout << "[WatchdogFeed] 未设置 WATCHDOG_SOCKET_PATH，喂狗通道关闭"
                     "（由看门狗启动时才会开启）" << std::endl;
        return;
    }

    socket_path_ = path;
    feed_period_ms_ = std::max<long long>(50, envMs("WATCHDOG_FEED_PERIOD_MS", feed_period_ms_));
    stall_timeout_ms_ = envMs("WATCHDOG_STALL_TIMEOUT_MS", stall_timeout_ms_);
    boot_grace_ms_ = envMs("WATCHDOG_BOOT_GRACE_MS", boot_grace_ms_);
    const char* verbose = std::getenv("WATCHDOG_FEED_VERBOSE");
    verbose_ = verbose && *verbose && std::strcmp(verbose, "0") != 0;

    start_ms_ = nowMs();
    last_tick_ms_.store(start_ms_, std::memory_order_relaxed);
    enabled_.store(true, std::memory_order_relaxed);
    running_.store(true, std::memory_order_relaxed);
    thread_ = std::thread(&WatchdogFeed::loop, this);

    std::cout << "[WatchdogFeed] 喂狗通道已开启 → " << socket_path_
              << " (period=" << feed_period_ms_ << "ms, stall=" << stall_timeout_ms_
              << "ms, boot_grace=" << boot_grace_ms_ << "ms)" << std::endl;
}

void WatchdogFeed::stop() {
    if (!started_.load(std::memory_order_relaxed)) return;
    running_.store(false, std::memory_order_relaxed);
    if (thread_.joinable()) thread_.join();   // 先停线程再关套接字，避免并发使用 fd_
    closeSocket();
    enabled_.store(false, std::memory_order_relaxed);
}

void WatchdogFeed::tick() {
    if (!enabled_.load(std::memory_order_relaxed)) return;
    frames_.fetch_add(1, std::memory_order_relaxed);
    last_tick_ms_.store(nowMs(), std::memory_order_relaxed);
}

bool WatchdogFeed::tryConnect() {
    if (socket_path_.size() >= sizeof(sockaddr_un::sun_path)) {
        if (!g_path_warned) {
            g_path_warned = true;
            std::cerr << "[WatchdogFeed] 套接字路径过长（上限 "
                      << sizeof(sockaddr_un::sun_path) - 1 << " 字节）：" << socket_path_
                      << std::endl;
        }
        return false;
    }

    const int fd = ::socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd < 0) return false;

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        const int err = errno;
        ::close(fd);
        if (!g_connect_failed_logged) {
            g_connect_failed_logged = true;
            std::cerr << "[WatchdogFeed] 连接看门狗失败（errno=" << err << ": "
                      << std::strerror(err) << "）：" << socket_path_ << "，稍后重试"
                      << std::endl;
        }
        return false;
    }

    fd_ = fd;
    g_connect_failed_logged = false;
    if (!g_connected_logged || verbose_) {
        g_connected_logged = true;
        std::cout << "[WatchdogFeed] 已连接看门狗 " << socket_path_ << std::endl;
    }
    return true;
}

void WatchdogFeed::sendFeed(std::uint64_t seq, std::uint64_t frames) {
    char buf[128];
    const int n = std::snprintf(buf, sizeof(buf), "UAP-WD/1 FEED seq=%llu frames=%llu\n",
                                static_cast<unsigned long long>(seq),
                                static_cast<unsigned long long>(frames));
    if (n <= 0 || static_cast<std::size_t>(n) >= sizeof(buf)) return;

    const ssize_t sent = ::send(fd_, buf, static_cast<std::size_t>(n), MSG_DONTWAIT | MSG_NOSIGNAL);
    if (sent < 0) {
        // 看门狗退出 / 套接字被删：断开，下个周期重连（不影响主程序）
        if (verbose_) {
            std::cerr << "[WatchdogFeed] 上报失败（errno=" << errno << ": "
                      << std::strerror(errno) << "），稍后重连" << std::endl;
        }
        closeSocket();
    }
}

void WatchdogFeed::closeSocket() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

void WatchdogFeed::loop() {
    std::uint64_t seq = 0;
    while (running_.load(std::memory_order_relaxed)) {
        // 分片睡眠：stop() 最多等 50ms 即可生效，不受上报周期影响
        for (long long slept = 0;
             slept < feed_period_ms_ && running_.load(std::memory_order_relaxed);
             slept += 50) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (!running_.load(std::memory_order_relaxed)) break;

        const long long now = nowMs();
        const std::uint64_t frames = frames_.load(std::memory_order_relaxed);

        // ── 健康判定：帧是否在推进 ──
        //   frames == 0：尚未产出第一帧，只受启动宽限约束（<= 0 表示不限制）；
        //   frames > 0 ：帧门控开启时要求 tick 距现在不超过 stall_timeout
        //               （<= 0 表示关闭帧门控，只要进程活着就一直喂）。
        bool alive = true;
        const char* reason = nullptr;
        if (frames == 0) {
            if (boot_grace_ms_ > 0 && now - start_ms_ > boot_grace_ms_) {
                alive = false;
                reason = "启动超时，未产出第一帧";
            }
        } else if (stall_timeout_ms_ > 0 &&
                   now - last_tick_ms_.load(std::memory_order_relaxed) > stall_timeout_ms_) {
            alive = false;
            reason = "帧停滞，处理线程长时间无新帧";
        }

        if (!alive) {
            // 故意停止上报：看门狗会在 feed_timeout 后重启整个进程组
            if (!stalled_) {
                stalled_ = true;
                std::cout << "[WatchdogFeed] 停止喂狗：" << reason << "（等待看门狗重启）"
                          << std::endl;
            }
            continue;
        }
        if (stalled_) {
            stalled_ = false;
            std::cout << "[WatchdogFeed] 帧恢复推进，重新开始喂狗" << std::endl;
        }

        if (fd_ < 0 && !tryConnect()) continue;
        sendFeed(++seq, frames);
    }
    closeSocket();
}
