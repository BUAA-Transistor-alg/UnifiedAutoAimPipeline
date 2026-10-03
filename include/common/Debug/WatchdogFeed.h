#ifndef WATCHDOG_FEED_H
#define WATCHDOG_FEED_H

// WatchdogFeed.h — 看门狗喂狗通道（可选功能，默认关闭，不影响平时手动运行）
//
// 用途：auto_launch/watchdog.py 以「心跳超时」判断主程序是否卡死（死锁 / 相机断流 /
//       流水线停摆），本类负责把主程序的存活状态周期性上报给它。
//
// 启用条件（未设置环境变量时本类完全空转，不建线程、不建套接字）：
//   WATCHDOG_SOCKET_PATH      看门狗监听的 Unix 域套接字路径（DGRAM）。设置后启用。
//   WATCHDOG_FEED_PERIOD_MS   上报周期，默认 500 ms。
//   WATCHDOG_STALL_TIMEOUT_MS 帧门控超时：连续该时长没有新帧（tick）就停止上报，
//                             让看门狗超时重启；<= 0 表示关闭帧门控（只要进程活着就上报）。
//   WATCHDOG_BOOT_GRACE_MS    启动宽限：从上电到产出第一帧的最长时间，超过则停止上报
//                             （抓「启动阶段就卡死」）；<= 0 表示不限制。
//   WATCHDOG_FEED_VERBOSE=1   打印每次连接/断开的详细日志。
//
// 上报内容：一行 UTF-8 文本 "UAP-WD/1 FEED seq=<n> frames=<n>\n"。
//   看门狗只要求「收得到数据」，序号/帧数仅用于日志与排查。
//
// 设计要点：
//   - tick() 由处理线程每处理完一帧调用一次（两次原子写，无锁、无系统调用）；
//   - 上报线程独立，负责连接与发送，连不上（看门狗未启动 / 路径不存在）时周期性
//     重连，不影响主程序；
//   - 只有「帧在推进」才算健康：相机断流或流水线内部卡死都会让 tick 停止，
//     从而触发看门狗重启（这一层是纯进程存活监控做不到的）；
//   - 退出时 stop() 会 join 上报线程，避免线程在静态对象析构后访问已释放资源。

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

class WatchdogFeed {
public:
    /// 进程内唯一实例（函数局部静态，C++11 起线程安全）
    static WatchdogFeed& instance();

    /// 读取环境变量并启动上报线程；未设置 WATCHDOG_SOCKET_PATH 时为空操作。
    /// 幂等：重复调用只有第一次生效。
    void start();

    /// 停止上报线程并关闭套接字（幂等）。正常退出前调用；未 start() 时为空操作。
    void stop();

    /// 每处理完一帧调用一次（廉价、线程安全）。
    void tick();

    /// 已累计的帧数（仅用于日志/排查）
    std::uint64_t frames() const { return frames_.load(std::memory_order_relaxed); }

    /// 喂狗通道是否已启用（环境变量有效）
    bool enabled() const { return enabled_.load(std::memory_order_relaxed); }

    /// 看门狗套接字路径（未启用时为空串）
    const std::string& socketPath() const { return socket_path_; }

private:
    WatchdogFeed() = default;
    ~WatchdogFeed();
    WatchdogFeed(const WatchdogFeed&) = delete;
    WatchdogFeed& operator=(const WatchdogFeed&) = delete;

    /// 上报线程主循环
    void loop();

    /// 连接看门狗套接字（失败返回 false，下个周期重试）
    bool tryConnect();

    /// 关闭并重置套接字描述符
    void closeSocket();

    /// 发送一次心跳（失败时自动断开，下个周期重连）
    void sendFeed(std::uint64_t seq, std::uint64_t frames);

    /// steady_clock 毫秒时间戳（单调，不受系统时间调整影响）
    static long long nowMs();

    /// 读取环境变量整数值（缺省 / 非法时返回默认值）
    static long long envMs(const char* name, long long fallback_ms);

    std::string socket_path_;
    long long   feed_period_ms_  = 500;     // 上报周期
    long long   stall_timeout_ms_ = 5000;   // 帧门控：多久没新帧就停喂
    long long   boot_grace_ms_   = 120000;  // 启动宽限：多久没第一帧就停喂
    bool        verbose_         = false;   // 连接/断开详细日志

    std::atomic<bool>          enabled_{false};    // 环境变量有效
    std::atomic<bool>          started_{false};    // start() 已执行
    std::atomic<bool>          running_{false};    // 上报线程运行标志
    std::atomic<std::uint64_t> frames_{0};         // tick() 累计帧数
    std::atomic<long long>     last_tick_ms_{0};   // 最近一次 tick() 的时刻

    long long   start_ms_ = 0;   // start() 时刻（计算启动宽限）
    int         fd_       = -1;  // 已连接的 Unix 域套接字（-1 = 未连接）
    bool        stalled_  = false;  // 上一周期是否处于「停喂」状态（仅用于日志去重）
    std::thread thread_;
};

#endif  // WATCHDOG_FEED_H
