#ifndef TASK_POOL_H
#define TASK_POOL_H

#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>

/**
 * @brief 固定大小的持久线程池，构造时分配线程，析构时回收。
 *
 * 用法：
 *   TaskPool pool;                          // 自动选择合适的线程数
 *   pool.run_parallel(N, [&](int i) {
 *       // 处理第 i 个任务 (i = 0 .. N-1)，由池中某一线程调用
 *   });
 *   // run_parallel 阻塞直到所有任务完成
 */
class TaskPool
{
public:
    /// @param num_threads  线程数，0 表示自动选择 (≤ hardware_concurrency/2, ≤4, ≥1)
    explicit TaskPool(size_t num_threads = 0);

    ~TaskPool();

    // 不可拷贝 / 移动
    TaskPool(const TaskPool&)            = delete;
    TaskPool& operator=(const TaskPool&) = delete;

    /// 提交任务（不阻塞），立即返回。调用 wait() 等待所有任务完成。
    void submit(int total_tasks, std::function<void(int)> fn);

    /// 等待所有已提交的任务完成。
    void wait();

    /// 提交任务并阻塞直到全部完成（等价于 submit + wait）。
    void run_parallel(int total_tasks, std::function<void(int)> fn);

    /// 工作线程数
    size_t size() const { return workers_.size(); }

private:
    void workerLoop();

    std::vector<std::thread> workers_;
    std::atomic<bool>        exit_{false};

    std::mutex               mtx_;
    std::condition_variable  cv_;
    std::condition_variable  done_cv_;     // 任务全部完成的等待条件变量
    bool                     work_ready_{false};

    std::function<void(int)> task_fn_;
    std::atomic<int>         task_idx_{0};
    int                      total_tasks_{0};
    std::atomic<int>         active_workers_{0};
};

#endif // TASK_POOL_H
