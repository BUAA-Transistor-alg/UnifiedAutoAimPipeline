#ifndef INFER_PROCESS_MANAGER_H
#define INFER_PROCESS_MANAGER_H

// InferProcessManager.h — 按需启动/停止推理进程（common.infer_process_lazy=true 时
// 由主程序使用）
//
// 默认模式（infer_process_lazy=false）：推理进程由 launch_all.py 在启动时全部启动并
// 常驻（无推理任务时后台闲置），主程序不管理进程，本类不被使用。
//
// lazy 模式（infer_process_lazy=true）：本类从主程序 fork/exec 推理进程：
//   - 启动时：startAndWait(kind) 同步启动当前流水线所需推理进程并等待其输出
//     "[InferShmServer] ready"（模型编译需数秒~十几秒），就绪后回调 on_ready(kind)
//     让主程序重连该流水线的 InferShmClient（客户端先于服务端附加，信号量需重开）；
//   - 切换时：switchTo(kind) 在后台线程中立即停止所有不再需要的推理进程（SIGINT →
//     等待 → SIGKILL，释放其 GPU 显存），再启动新流水线所需推理进程并等待就绪；
//     可连续调用，以后一次请求为准；就绪前该流水线推理经 InferShmClient 的
//     2s 响应超时返回空结果，界面不阻塞；
//   - 退出时：shutdown() 停止全部推理进程并结束后台线程；推理进程设置
//     PR_SET_PDEATHSIG，主程序异常退出时子进程自动终止，不留残留。
//
// 线程模型：procs_ 映射只允许单线程访问——启动阶段由主线程（startAndWait，
// worker 尚未启动），运行阶段仅由 worker 线程（switchTo 的请求）访问；
// shutdown 先 join worker 再由主线程清理。startAndWait 与 switchTo 不得并发调用。

#include <atomic>
#include <condition_variable>
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
    /// 启动，不得与 switchTo 并发调用）。就绪后回调 on_ready(kind)。
    /// @return true 成功；false 启动失败/超时（进程已清理）
    bool startAndWait(Kind kind, int timeout_sec = 120);

    /// 运行时切换（异步，立即返回）：后台线程立即停止所有非 kind 的推理进程，
    /// 启动 kind 的推理进程并等待就绪，随后回调 on_ready(kind)。
    void switchTo(Kind kind);

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

    static constexpr int STOP_GRACE_MS = 5000;   // SIGINT 后等待优雅退出上限

    std::string processPath(Kind kind) const;
    static std::string processName(Kind kind);
    static std::string readyMarker();

    /// 启动（或复用已就绪的）kind 推理进程并等待就绪；失败时清理并返回 false。
    /// 不调用 on_ready（由 startAndWait / applySwitch 统一调用）。仅单线程访问。
    bool spawnAndWait(Kind kind, int timeout_sec);
    /// 停止并清理 kind 推理进程（SIGINT → 等待 → SIGKILL + 收割 + join reader）。
    /// 仅单线程访问。
    void stopProcess(Kind kind);
    void stopAllProcesses();
    void readerLoop(Proc& proc, Kind kind);   // 转发输出，置 ready/exited
    void workerLoop();
    void applySwitch(Kind kind);

    std::function<void(Kind)> on_ready_;

    std::mutex mtx_;
    std::condition_variable cv_;
    bool  request_pending_ = false;   // 有待处理的 switchTo 请求（mtx_ 保护）
    Kind  desired_ = Kind::ARMOR;   // 最近一次目标流水线（mtx_ 保护）
    bool  worker_running_ = false;    // worker 线程已启动（mtx_ 保护）
    std::atomic<bool> shutdown_{false};

    std::unordered_map<Kind, Proc> procs_;   // 仅单线程访问（见文件头注释）
    std::thread worker_;
};

} // namespace Infer

#endif // INFER_PROCESS_MANAGER_H
