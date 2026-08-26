#ifndef PIPELINE_STAGE_H
#define PIPELINE_STAGE_H

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <cstddef>

/**
 * @brief 流水线处理阶段模板
 *
 * 封装了单阶段工作线程的完整生命周期：idle 状态管理、mutex/cv 同步、
 * 输入输出队列的自主维护（内建 pop/push 逻辑）。
 * 处理逻辑由外部通过 ProcessFunc 注入，队列与锁通过 Config 传入。
 *
 * @tparam Container  数据容器类型 (如 std::deque<std::unique_ptr<PipelineData>>)
 */
template <typename Container>
class PipelineStage {
public:
    /// 实际处理函数：对 stage_in_flight_ 中的数据进行处理（修改原位数据）
    using ProcessFunc  = std::function<void(Container&)>;

    /// 阶段完成回调（idle 置 true 后调用），用于唤醒事件驱动调度器
    using DoneCallback = std::function<void()>;

    /// 构造时传入的配置
    struct Config {
        Container*  input_queue{nullptr};     // 输入队列指针
        Container*  output_queue{nullptr};    // 输出队列指针
        std::mutex* input_mtx{nullptr};       // 输入队列锁，nullptr = 无锁
        std::mutex* output_mtx{nullptr};      // 输出队列锁，nullptr = 无锁
        std::condition_variable* input_cv{nullptr}; // 输入队列非空通知，nullptr = 无需通知
        size_t      max_batch{1};             // 最大处理批量
        size_t      output_max{0};            // 输出队列容量上限，0 = 不限制
        ProcessFunc process_fn;               // 处理逻辑
        DoneCallback on_done;                 // 完成回调
    };

    PipelineStage() = default;

    ~PipelineStage() {
        shutdown();
    }

    // 禁止拷贝和移动（持有线程）
    PipelineStage(const PipelineStage&) = delete;
    PipelineStage& operator=(const PipelineStage&) = delete;
    PipelineStage(PipelineStage&&) = delete;
    PipelineStage& operator=(PipelineStage&&) = delete;

    /**
     * @brief 启动工作线程，传入完整配置
     */
    void launch(const Config& cfg) {
        input_queue_  = cfg.input_queue;
        output_queue_ = cfg.output_queue;
        input_mtx_    = cfg.input_mtx;
        output_mtx_   = cfg.output_mtx;
        input_cv_     = cfg.input_cv;
        max_batch_    = cfg.max_batch;
        output_max_   = cfg.output_max;
        process_fn_   = cfg.process_fn;
        on_done_      = cfg.on_done;
        worker_       = std::thread(&PipelineStage::run, this);
    }

    /**
     * @brief 阶段是否空闲
     */
    bool isIdle() const {
        return idle_.load(std::memory_order_acquire);
    }

    /**
     * @brief 尝试推进本阶段：输出已完成数据 → 检查容量 → 输入新数据 → 启动处理
     * @return true 表示启动了新的处理（有进展）
     */
    bool tryAdvance() {
        if (!isIdle()) return false;

        // ---- 1. 将上一轮处理完的数据推入输出队列 ----
        if (output_mtx_) {
            std::lock_guard<std::mutex> lk(*output_mtx_);
            for (auto& d : stage_in_flight_)
                output_queue_->push_back(std::move(d));
            stage_in_flight_.clear();
            // 容量检查在锁内完成
            if (output_max_ > 0 && output_queue_->size() >= output_max_)
                return false;
        } else {
            for (auto& d : stage_in_flight_)
                output_queue_->push_back(std::move(d));
            stage_in_flight_.clear();
            if (output_max_ > 0 && output_queue_->size() >= output_max_)
                return false;
        }

        // ---- 2. 从输入队列拉取新数据 ----
        size_t count = 0;
        if (input_mtx_) {
            std::unique_lock<std::mutex> lk(*input_mtx_);
            if (input_queue_->empty()) return false;
            count = std::min(max_batch_, input_queue_->size());
            for (size_t i = 0; i < count; ++i) {
                stage_in_flight_.push_back(std::move(input_queue_->front()));
                input_queue_->pop_front();
            }
        } else {
            if (input_queue_->empty()) return false;
            count = std::min(max_batch_, input_queue_->size());
            for (size_t i = 0; i < count; ++i) {
                stage_in_flight_.push_back(std::move(input_queue_->front()));
                input_queue_->pop_front();
            }
        }

        if (count > 0) {
            if (input_cv_) input_cv_->notify_one();
            start();
            return true;
        }
        return false;
    }

    /**
     * @brief 阻塞等待本阶段空闲（worker 完成当前批处理）。
     *
     * 前提：调用方必须已暂停调度器（例如持有流水线的 advance_mtx_ 或等价互斥），
     * 保证不会有新的 tryAdvance 启动新批，否则本等待可能永不结束。
     */
    void waitIdle() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]() {
            return idle_.load(std::memory_order_acquire);
        });
    }

    /**
     * @brief 丢弃当前在途批数据。
     *
     * 前提：本阶段已空闲（worker 已结束 process_fn_，见 waitIdle）且调度器已暂停，
     * 此时 stage_in_flight_ 不再被任何线程访问，可直接清空。
     */
    void discardInFlight() {
        stage_in_flight_.clear();
    }

    /**
     * @brief 关闭工作线程（阻塞直到线程退出）
     */
    void shutdown() {
        if (shutdown_done_) return;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            exit_flag_ = true;
        }
        cv_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
        shutdown_done_ = true;
    }

private:
    /// 唤醒工作线程
    void start() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            idle_.store(false, std::memory_order_release);
        }
        cv_.notify_one();
    }

    /// 工作线程主循环
    void run() {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [this]() {
                    return !idle_.load(std::memory_order_acquire) || exit_flag_;
                });
                if (exit_flag_) return;
            }

            process_fn_(stage_in_flight_);

            // idle 置位与唤醒必须在 mtx_ 下完成：waitIdle 在持锁时检查/等待
            // idle_，避免"置位-通知"落在其检查与睡眠之间造成通知丢失。
            {
                std::lock_guard<std::mutex> lock(mtx_);
                idle_.store(true, std::memory_order_release);
            }
            cv_.notify_all();

            if (on_done_) {
                on_done_();
            }
        }
    }

    // --- 队列与锁 ---
    Container*  input_queue_{nullptr};
    Container*  output_queue_{nullptr};
    std::mutex* input_mtx_{nullptr};
    std::mutex* output_mtx_{nullptr};
    std::condition_variable* input_cv_{nullptr};
    size_t      max_batch_{1};
    size_t      output_max_{0};

    // --- 当前处理中数据 ---
    Container   stage_in_flight_;

    // --- 线程管理 ---
    std::thread       worker_;
    std::atomic<bool> idle_{true};
    std::mutex        mtx_;
    std::condition_variable cv_;
    bool              exit_flag_{false};
    bool              shutdown_done_{false};

    // --- 处理逻辑 ---
    ProcessFunc  process_fn_;
    DoneCallback on_done_;
};

#endif // PIPELINE_STAGE_H
