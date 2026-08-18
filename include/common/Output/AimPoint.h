// AimPoint.h — 预测瞄准点共享槽（GimbalOutput 写，VisualizeOutput 读）
//
// 弹道解算（GimbalOutput）每帧算出实际瞄准的预测目标点（world 系）与预测时间，
// 写入本共享槽；可视化（VisualizeOutput）读取并在画面上绘制，保证两个输出模式
// 独立运行也能显示一致的最新瞄准点。两种流水线模式（Outpost / PowerRune）共用。
#ifndef AIM_POINT_H
#define AIM_POINT_H

#include <mutex>

#include <opencv2/opencv.hpp>

class AimPoint {
public:
    // 写入最新瞄准点（线程安全）
    void set(const cv::Vec3f& predicted_point, double predict_time) {
        std::lock_guard<std::mutex> lock(mtx_);
        valid_ = true;
        predicted_point_ = predicted_point;
        predict_time_ = predict_time;
    }

    // 失效（预测器不可用/保持模式时调用）
    void invalidate() {
        std::lock_guard<std::mutex> lock(mtx_);
        valid_ = false;
    }

    // 读取最近瞄准点；无有效瞄准点时返回 false
    bool get(cv::Vec3f& predicted_point, double& predict_time) const {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!valid_) return false;
        predicted_point = predicted_point_;
        predict_time = predict_time_;
        return true;
    }

private:
    mutable std::mutex mtx_;
    bool valid_ = false;
    cv::Vec3f predicted_point_ = cv::Vec3f(0, 0, 0);
    double predict_time_ = 0.0;
};

#endif // AIM_POINT_H
