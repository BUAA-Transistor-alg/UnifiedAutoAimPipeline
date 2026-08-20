#include "common/TaskPool.h"
#include <algorithm>

TaskPool::TaskPool(size_t num_threads)
{
    if (num_threads == 0) {
        unsigned int hw = std::thread::hardware_concurrency();
        num_threads = std::max<size_t>(1,
            std::min<size_t>(hw > 0 ? hw / 2 : 2, 4));
    }
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&TaskPool::workerLoop, this);
    }
}

TaskPool::~TaskPool()
{
    exit_.store(true, std::memory_order_relaxed);
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

void TaskPool::submit(int total_tasks, std::function<void(int)> fn)
{
    task_fn_      = std::move(fn);
    total_tasks_  = total_tasks;
    task_idx_     = 0;

    {
        std::lock_guard<std::mutex> lk(mtx_);
        work_ready_ = true;
    }
    cv_.notify_all();
}

void TaskPool::wait()
{
    std::unique_lock<std::mutex> lk(mtx_);
    done_cv_.wait(lk, [this]() {
        return task_idx_.load(std::memory_order_acquire) >= total_tasks_ &&
               active_workers_.load(std::memory_order_acquire) == 0;
    });
    work_ready_ = false;
}

void TaskPool::run_parallel(int total_tasks, std::function<void(int)> fn)
{
    submit(total_tasks, std::move(fn));
    wait();
}

void TaskPool::workerLoop()
{
    while (!exit_.load(std::memory_order_relaxed)) {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this]() {
                return work_ready_ || exit_.load(std::memory_order_relaxed);
            });
            if (exit_.load(std::memory_order_relaxed)) return;
            active_workers_.fetch_add(1, std::memory_order_relaxed);
        }

        for (int idx = task_idx_.fetch_add(1, std::memory_order_relaxed);
             idx < total_tasks_;
             idx = task_idx_.fetch_add(1, std::memory_order_relaxed))
        {
            task_fn_(idx);
        }

        // 最后一个完成任务的 worker 通知 done_cv_
        // 需要在锁内修改状态以确保条件变量的正确同步语义
        bool is_last;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            is_last = (active_workers_.fetch_sub(1, std::memory_order_relaxed) == 1);
        }
        if (is_last) {
            done_cv_.notify_one();
        }
    }
}
