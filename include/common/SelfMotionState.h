// SelfMotionState.h — 自身运动状态（位置 / 速度 / 加速度按时间查询）
//
// 用途：给弹道解算提供"某个时间点的自身位置 / 自身速度"，对应未来位置轨迹与未来
// 速度轨迹。main 的取帧线程每帧调用 update() 写入自身坐标；弹道线程按预测点对应的
// 发射时刻调用 positionAt() / velocityAt() 查询。
//
// ★ 当前为占位实现：
//   - positionAt()：忽略传入时间戳，返回最近一次 update() 设置的位置（最新值）；
//   - velocityAt() / accelerationAt()：恒返回 0 向量。
//   后续实现按时间预测/插值时只改这三个方法，调用方无需改动。
//
// 线程安全：update() 单写者（取帧线程），三个查询多读者（弹道线程等），内部用互斥量保护。
#ifndef SELF_MOTION_STATE_H
#define SELF_MOTION_STATE_H

#include <chrono>
#include <mutex>

#include <opencv2/core.hpp>

// 更新用数据帧：当前只承载自身坐标（world 系，米）；后续扩展在此结构体里加字段。
struct SelfPositionSample {
    cv::Vec3f position = cv::Vec3f(0.0f, 0.0f, 0.0f);  // 自身坐标（world 系，米）
};

class SelfMotionState {
public:
    // 写入一帧自身坐标；timestamp 为该数据的对应时间戳（当前实现记录备用）。
    void update(const SelfPositionSample& sample,
                std::chrono::steady_clock::time_point timestamp);

    // 查询某时间点的自身位置；当前实现忽略 timestamp，返回最新被设置的位置。
    cv::Vec3f positionAt(std::chrono::steady_clock::time_point timestamp) const;

    // 查询某时间点的自身速度；当前实现为占位，恒返回 0。
    cv::Vec3f velocityAt(std::chrono::steady_clock::time_point timestamp) const;

    // 查询某时间点的自身加速度；当前实现为占位，恒返回 0。
    cv::Vec3f accelerationAt(std::chrono::steady_clock::time_point timestamp) const;

private:
    mutable std::mutex mutex_;                            // 保护 position_ / timestamp_
    cv::Vec3f position_ = cv::Vec3f(0.0f, 0.0f, 0.0f);    // 最近一次写入的位置
    std::chrono::steady_clock::time_point timestamp_{};   // 最近一次写入的时间戳（备用）
};

#endif // SELF_MOTION_STATE_H
