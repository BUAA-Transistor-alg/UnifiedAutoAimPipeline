#ifndef INFER_PROCESS_MANAGER_H
#define INFER_PROCESS_MANAGER_H

// InferProcessManager.h — 按需启动/停止/强制重启推理进程（主程序始终构造；
// 仅在 common.infer_process_lazy=true 时于切换流水线时按需启停推理进程）
//
// 默认模式（infer_process_lazy=false）：推理进程由 launch_all.py 在启动时全部启动并
// 常驻（无推理任务时后台闲置），主程序不主动启停；本类仅在被推理客户端调用
// forceRestart（推理进程疑似挂死）时按需接管对应进程的重启。
//
// lazy 模式（infer_process_lazy=true）：本类从主程序 fork/exec 推理进程：
//   - 启动时：startAndWait(kind) 同步启动当前流水线所需推理进程并等待其输出
//     "[InferShmServer] ready"（模型编译需数秒~十几秒），就绪后回调 on_ready(kind)
//     让主程序重连该流水线的 InferShmClient（客户端先于服务端附加，信号量需重开）；
//   - 切换时：switchTo(kind) 在后台线程中立即停止所有不再需要的推理进程（SIGINT →
//     等待 → SIGKILL，释放其 GPU 显存），再启动新流水线所需推理进程并等待就绪；
//     可连续调用，以后一次请求为准；就绪前该流水线推理经 InferShmClient 的
//     2s 响应超时返回空结果，界面不阻塞；
//   - 强制重启：forceRestart(kind) 线程安全，可从任意线程（如流水线推理阶段线程）
//     调用：立即 SIGKILL 终止该 kind 的推理进程（含非 lazy 模式下由 launch_all.py
//     启动的外部进程，按进程名在 /proc 中查找并终止），重新启动并等待就绪，就绪后
//     回调 on_ready(kind) 让主程序重连客户端。推理客户端在响应超时且距上一次正常
//     返回超过配置时限时调用（见 InferShmClient）；
//   - 退出时：shutdown() 停止全部推理进程并结束后台线程；推理进程设置
//     PR_SET_PDEATHSIG，主程序异常退出时子进程自动终止，不留残留。
//
// 线程模型：procs_ 映射只允许单线程访问——启动阶段由主线程（startAndWait，worker
// 尚未启动），运行阶段仅由后台 worker 线程访问。switchTo / forceRestart 均只把请求
// 排入队列（mtx_ 保护）后立即返回，由 worker 串行处理，因此二者可从任意线程安全
// 调用；startAndWait 仍须在 worker 启动之前完成（推理客户端在推理进程就绪前不会
// 触发 forceRestart——其"上次正常返回"尚未建立，见 InferShmClient::runInference）。

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace Infer {

class InferProcessManager {
public:
    enum class Kind { ARMOR, POWER_RUNE };

    /// @param on_ready 某推理进程启动并就绪后的回调（主程序用来重连该流水线的
    ///                 InferShmClient 信号量）；可能从后台线程调用
    explicit InferProcessManager(std::function<void(Kind)> on_ready);
    ~InferProcessManager();

    InferProcessManager(const InferProcessManager&) = delete;
    InferProcessManager& operator=(const InferProcessManager&) = delete;

    /// 同步启动 kind 对应推理进程并等待就绪（启动路径；此时 worker 线程尚未
    /// 启动，不得与 switchTo / forceRestart 并发调用）。就绪后回调 on_ready(kind)。
    /// @return true 成功；false 启动失败/超时（进程已清理）
    bool startAndWait(Kind kind, int timeout_sec = 120);

    /// 运行时切换（异步，立即返回）：后台线程立即停止所有非 kind 的推理进程，
    /// 启动 kind 的推理进程并等待就绪，随后回调 on_ready(kind)。
    /// 线程安全：可从任意线程调用；连续调用时以后一次请求为准。
    void switchTo(Kind kind);

    /// 强制重启 kind 对应推理进程（异步，立即返回；线程安全，可从任意线程调用）：
    /// 后台线程立即 SIGKILL 终止该进程（含非 lazy 模式下由 launch_all.py 启动的
    /// 外部进程，按进程名查找），重新启动并等待就绪，随后回调 on_ready(kind)。
    /// 同一 kind 同时只接受一次强制重启（进行中/排队时重复调用被忽略）。
    void forceRestart(Kind kind);

    /// 停止全部推理进程并等待后台线程结束（幂等，程序退出时调用）
    void shutdown();

private:
    /// 单个推理进程的运行时状态（reader 线程持有其引用，见 readerLoop）
    struct Proc {
        pid_t pid = -1;              // 子进程 PID（-1 = 未运行）
        int   read_fd = -1;          // 子进程 stdout/stderr 管道读端
        std::thread reader;          // 转发输出 + 检测就绪/退出
        std::atomic<bool> ready{false};   // 已输出 "[InferShmServer] ready"
        std::atomic<bool> exited{false};  // 子进程已退出（管道 EOF）
    };

    /// 后台请求类型（worker 串行处理）
    enum class RequestType { SWITCH, FORCE_RESTART };
    struct Request {
        RequestType type;
        Kind        kind;
        uint64_t    switch_count_seen = 0;  // 排队时的 switch_count_（强制重启丢弃判定）
    };

    static constexpr int STOP_GRACE_MS = 5000;   // SIGINT/SIGKILL 后等待退出上限

    std::string processPath(Kind kind) const;
    static std::string processName(Kind kind);
    static std::string readyMarker();

    /// 启动（或复用已就绪的）kind 推理进程并等待就绪；失败时清理并返回 false。
    /// 不调用 on_ready（由 startAndWait / applySwitch / applyForceRestart 统一调用）。
    /// 仅单线程访问。
    bool spawnAndWait(Kind kind, int timeout_sec);
    /// 停止并清理 kind 推理进程（SIGINT → 等待 → SIGKILL + 收割 + join reader；
    /// force=true 时跳过优雅等待，立即 SIGKILL）。仅单线程访问。
    void stopProcess(Kind kind, bool force = false);
    void stopAllProcesses();
    /// 终止由外部（launch_all.py）启动的同名推理进程（非 lazy 模式强制重启用），
    /// 按进程名在 /proc 中查找，SIGKILL 后等待其退出。仅单线程访问。
    void stopExternalProcess(Kind kind);
    /// 在 /proc 中查找进程名恰为 processName(kind) 的运行中进程 PID（找不到返回 -1）
    static pid_t findExternalPid(Kind kind);
    void readerLoop(Proc& proc, Kind kind);   // 转发输出，置 ready/exited
    void workerLoop();
    void applySwitch(Kind kind);
    void applyForceRestart(Kind kind);

    std::function<void(Kind)> on_ready_;

    std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<Request> queue_;               // 待处理请求（mtx_ 保护）
    std::array<std::atomic<bool>, 2> restart_active_{false, false};  // 强制重启去重（mtx_ 保护）
    std::array<std::atomic<uint64_t>, 2> switch_count_{0, 0};  // 各 kind 完成切换次数
    bool  worker_running_ = false;            // worker 线程已启动（mtx_ 保护）
    std::atomic<bool> shutdown_{false};

    std::unordered_map<Kind, Proc> procs_;   // 仅单线程访问（见文件头注释）
    std::thread worker_;
};

} // namespace Infer

#endif // INFER_PROCESS_MANAGER_H
