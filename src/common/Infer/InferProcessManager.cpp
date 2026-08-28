// InferProcessManager.cpp — 按需启动/停止推理进程的实现（见头文件注释）
#include "common/Infer/InferProcessManager.h"
#include "common/PathResolver.h"

#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

namespace Infer {

namespace {
constexpr const char* kReadyMark = "[InferShmServer] ready";
} // namespace

InferProcessManager::InferProcessManager(std::function<void(Kind)> on_ready)
    : on_ready_(std::move(on_ready)) {}

InferProcessManager::~InferProcessManager() {
    shutdown();
}

bool InferProcessManager::startAndWait(Kind kind, int timeout_sec) {
    if (!spawnAndWait(kind, timeout_sec)) {
        return false;
    }
    if (on_ready_) on_ready_(kind);
    return true;
}

void InferProcessManager::switchTo(Kind kind) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        desired_ = kind;
        request_pending_ = true;
        if (!worker_running_) {
            worker_running_ = true;
            worker_ = std::thread(&InferProcessManager::workerLoop, this);
        }
    }
    cv_.notify_one();
}

void InferProcessManager::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        shutdown_.store(true);
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    stopAllProcesses();
}

std::string InferProcessManager::processPath(Kind kind) const {
    const char* name = (kind == Kind::ARMOR)
        ? "armor_infer_process" : "power_rune_infer_process";
    return PathResolver::resolvePath(std::string("bin/") + name);
}

std::string InferProcessManager::processName(Kind kind) {
    return (kind == Kind::ARMOR) ? "armor_infer_process"
                                   : "power_rune_infer_process";
}

std::string InferProcessManager::readyMarker() {
    return kReadyMark;
}

// ── reader 线程：转发子进程输出到 stdout（带标签），检测就绪标记与进程退出 ──
void InferProcessManager::readerLoop(Proc& proc, Kind kind) {
    char buf[4096];
    std::string pending;
    ssize_t n;
    while ((n = read(proc.read_fd, buf, sizeof(buf))) > 0) {
        pending.append(buf, static_cast<size_t>(n));
        size_t pos;
        while ((pos = pending.find('\n')) != std::string::npos) {
            std::string line = pending.substr(0, pos);
            pending.erase(0, pos + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::cout << "[infer_process/" << processName(kind) << "] "
                      << line << std::endl;
            if (!proc.ready.load() &&
                line.find(readyMarker()) != std::string::npos) {
                proc.ready.store(true);
            }
        }
    }
    if (!pending.empty()) {   // 末尾无换行的残片
        std::cout << "[infer_process/" << processName(kind) << "] "
                  << pending << std::endl;
        if (!proc.ready.load() &&
            pending.find(readyMarker()) != std::string::npos) {
            proc.ready.store(true);
        }
    }
    proc.exited.store(true);
    if (proc.read_fd >= 0) {
        close(proc.read_fd);
        proc.read_fd = -1;
    }
}

// ── 启动 + 等待就绪（仅单线程访问）──
bool InferProcessManager::spawnAndWait(Kind kind, int timeout_sec) {
    // 已在运行且就绪：直接复用
    auto it = procs_.find(kind);
    if (it != procs_.end() && it->second.ready.load()) {
        return true;
    }
    // 清理残留（上次启动失败/未清理的旧进程）
    stopProcess(kind);

    const std::string path = processPath(kind);
    if (access(path.c_str(), X_OK) != 0) {
        std::cerr << "[InferProcessManager] 推理进程不存在或不可执行: "
                  << path << "（请先运行 ./build.sh）" << std::endl;
        return false;
    }

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        std::cerr << "[InferProcessManager] pipe 失败: " << strerror(errno) << std::endl;
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "[InferProcessManager] fork 失败: " << strerror(errno) << std::endl;
        close(pipefd[0]);
        close(pipefd[1]);
        return false;
    }
    if (pid == 0) {
        // ── 子进程 ──
        // 父进程（主程序）退出时自动终止，避免主程序异常退出后残留推理进程；
        // prctl 后复查父进程，处理 prctl 生效前父进程已退出的竞态
        const pid_t parent = getppid();
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        if (getppid() != parent) {
            _exit(1);
        }
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[0]);
        close(pipefd[1]);
        execl(path.c_str(), path.c_str(), static_cast<char*>(nullptr));
        std::cerr << "[InferProcessManager] exec 失败: " << strerror(errno) << std::endl;
        _exit(127);
    }
    close(pipefd[1]);

    // ── 父进程 ──
    // Proc 含 std::atomic 成员（不可移动/拷贝），就地默认构造后逐字段赋值
    Proc& slot = procs_[kind];
    if (slot.reader.joinable()) {
        // 理论不可达（spawnAndWait 开头已 stopProcess 清理槽位）；防御：避免覆盖
        // 仍在运行的 reader 线程导致 std::terminate
        std::cerr << "[InferProcessManager] 内部错误：进程槽位已有运行中的 reader，"
                  << "先清理。" << std::endl;
        stopProcess(kind);
        return false;
    }
    slot.pid = pid;
    slot.read_fd = pipefd[0];
    slot.reader = std::thread(&InferProcessManager::readerLoop, this,
                              std::ref(slot), kind);

    std::cout << "[InferProcessManager] 已启动 " << processName(kind)
              << " (pid=" << pid << ")，等待就绪 ..." << std::endl;

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(timeout_sec);
    while (std::chrono::steady_clock::now() < deadline) {
        if (shutdown_.load()) {
            stopProcess(kind);
            return false;
        }
        if (slot.ready.load()) {
            std::cout << "[InferProcessManager] " << processName(kind)
                      << " 就绪。" << std::endl;
            return true;
        }
        if (slot.exited.load()) {
            std::cerr << "[InferProcessManager] " << processName(kind)
                      << " 提前退出（请检查模型路径/config）" << std::endl;
            stopProcess(kind);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cerr << "[InferProcessManager] 等待 " << processName(kind)
              << " 就绪超时（" << timeout_sec << "s）" << std::endl;
    stopProcess(kind);
    return false;
}

// ── 停止并清理推理进程（仅单线程访问）──
void InferProcessManager::stopProcess(Kind kind) {
    auto it = procs_.find(kind);
    if (it == procs_.end()) return;
    Proc& proc = it->second;

    if (proc.pid > 0 && !proc.exited.load()) {
        if (kill(proc.pid, SIGINT) != 0 && errno != ESRCH) {
            std::cerr << "[InferProcessManager] kill(SIGINT) " << processName(kind)
                      << " 失败: " << strerror(errno) << std::endl;
        }
        // 等待优雅退出（推理进程空闲时 sem_timedwait 100ms 内返回，很快退出）
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(STOP_GRACE_MS);
        while (std::chrono::steady_clock::now() < deadline && !proc.exited.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (!proc.exited.load()) {
            std::cerr << "[InferProcessManager] " << processName(kind)
                      << " 未在 " << (STOP_GRACE_MS / 1000) << "s 内退出，SIGKILL"
                      << std::endl;
            kill(proc.pid, SIGKILL);
        }
    }

    // 收割僵尸进程
    if (proc.pid > 0) {
        int status = 0;
        waitpid(proc.pid, &status, 0);
        proc.pid = -1;
    }
    if (proc.reader.joinable()) proc.reader.join();
    if (proc.read_fd >= 0) {
        close(proc.read_fd);
        proc.read_fd = -1;
    }
    std::cout << "[InferProcessManager] " << processName(kind) << " 已停止。"
              << std::endl;
    procs_.erase(it);
}

void InferProcessManager::stopAllProcesses() {
    // 先收集再逐个停止（erase 会使迭代器失效）
    std::vector<Kind> kinds;
    kinds.reserve(procs_.size());
    for (const auto& kv : procs_) kinds.push_back(kv.first);
    for (Kind k : kinds) stopProcess(k);
}

// ── worker 线程：串行处理切换请求 ──
void InferProcessManager::workerLoop() {
    while (true) {
        Kind k;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this]() {
                return shutdown_.load() || request_pending_;
            });
            if (shutdown_.load()) break;
            request_pending_ = false;
            k = desired_;
        }
        applySwitch(k);
    }
}

void InferProcessManager::applySwitch(Kind kind) {
    // 1. 立即停止所有不再需要的推理进程（释放其 GPU 显存）
    std::vector<Kind> to_stop;
    for (const auto& kv : procs_) {
        if (kv.first != kind) to_stop.push_back(kv.first);
    }
    for (Kind k : to_stop) stopProcess(k);
    if (shutdown_.load()) return;

    // 2. 确保需要的推理进程已启动并就绪
    auto it = procs_.find(kind);
    if (it == procs_.end() || !it->second.ready.load()) {
        if (!spawnAndWait(kind, /*timeout_sec=*/120)) {
            std::cerr << "[InferProcessManager] 启动 " << processName(kind)
                      << " 失败，流水线推理暂不可用。" << std::endl;
            return;
        }
    }

    // 3. 就绪后通知主程序重连该流水线的 InferShmClient
    if (on_ready_) on_ready_(kind);
}

} // namespace Infer
