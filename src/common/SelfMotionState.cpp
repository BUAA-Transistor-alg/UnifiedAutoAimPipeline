// SelfMotionState.cpp — 自身运动状态（占位实现）
#include "common/SelfMotionState.h"

void SelfMotionState::update(const SelfPositionSample& sample,
                             std::chrono::steady_clock::time_point timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    position_ = sample.position;
    timestamp_ = timestamp;
}

cv::Vec3f SelfMotionState::positionAt(std::chrono::steady_clock::time_point timestamp) const {
    (void)timestamp;  // 占位：不看查询时间，返回最新设置的位置
    std::lock_guard<std::mutex> lock(mutex_);
    return position_;
}

cv::Vec3f SelfMotionState::velocityAt(std::chrono::steady_clock::time_point timestamp) const {
    (void)timestamp;  // 占位：返回 0
    return cv::Vec3f(0.0f, 0.0f, 0.0f);
}

cv::Vec3f SelfMotionState::accelerationAt(std::chrono::steady_clock::time_point timestamp) const {
    (void)timestamp;  // 占位：返回 0
    return cv::Vec3f(0.0f, 0.0f, 0.0f);
}
