// LatestSlot.h — 单缓冲位"最新覆盖"邮箱（Latest-wins mailbox）
//
// 语义（供多级循环线程间解耦，见 main.cpp 弹道/可视化线程的用法）：
//  - 单缓冲位：任意时刻至多保存一个数据；
//  - publish()：写入最新数据；若槽中旧数据尚未被取走则直接覆盖（丢弃中间帧，
//    消费线程跟不上时不积压）；
//  - take()：取走并清空槽位；槽空时阻塞等待（条件变量唤醒，无轮询延迟——
//    消费循环不设帧率上限，有数据即处理）；
//  - stop()：停止——唤醒所有阻塞的 take() 并使其后续返回 false（线程退出）。
//
// 典型用法（流水线式数据流）：
//   上一级处理完 → slot.publish(结果)；
//   本级线程     → while (slot.take(x)) { 处理 x; next_slot.publish(y); }
#ifndef LATEST_SLOT_H
#define LATEST_SLOT_H

#include <condition_variable>
#include <mutex>
#include <optional>
#include <utility>

template <typename T>
class LatestSlot {
public:
    // 写入最新数据；槽中旧数据未被取走时直接覆盖
    void publish(T value) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            slot_ = std::move(value);
        }
        cv_.notify_one();
    }

    // 阻塞直到有数据；取走后清空槽位。返回 false 表示已停止且无数据。
    bool take(T& out) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [&] { return slot_.has_value() || stopped_; });
        if (!slot_.has_value()) return false;   // 已停止
        out = std::move(*slot_);
        slot_.reset();
        return true;
    }

    // 非阻塞取走（无数据立即返回 false，不等待）。供需要同时泵其他工作的
    // 消费线程使用（如可视化线程：无请求时仍需持续刷新窗口/读取按键）。
    bool tryTake(T& out) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!slot_.has_value()) return false;
        out = std::move(*slot_);
        slot_.reset();
        return true;
    }

    // 停止：唤醒所有阻塞的 take() 并使其后续返回 false
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stopped_ = true;
        }
        cv_.notify_all();
    }

private:
    std::mutex mtx_;
    std::condition_variable cv_;
    std::optional<T> slot_;   // 单缓冲位（无数据时为空）
    bool stopped_ = false;
};

#endif // LATEST_SLOT_H
