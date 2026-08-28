#include "Armor/EKF/FirstObjectTracker.h"

#include <algorithm>

#include "common/TransformTree/CoordinateTransform.h"

namespace sp_ekf {

FirstObjectTracker::FirstObjectTracker(const Params& params) : params_(params) {}

bool FirstObjectTracker::processFrame(const std::vector<cv::Vec3f>& world_pos,
                                      const std::vector<cv::Vec3f>& world_euler,
                                      const TimePoint& t) {
    // 取本帧第一个有效物体（PnP 失败填充的 (0,0,0) 视为无效，跳过）。
    const cv::Vec3f* obs_pos = nullptr;
    const cv::Vec3f* obs_euler = nullptr;
    const std::size_t count = std::min(world_pos.size(), world_euler.size());
    for (std::size_t k = 0; k < count; ++k) {
        const cv::Vec3f& wp = world_pos[k];
        if (wp[0] != 0.0f || wp[1] != 0.0f || wp[2] != 0.0f) {
            obs_pos = &wp;
            obs_euler = &world_euler[k];
            break;
        }
    }

    if (obs_pos) {
        // 保存第一个有效物体的位姿（世界坐标米 + 欧拉角构造的旋转矩阵 CV_64F）
        position_ = cv::Vec3d(static_cast<double>((*obs_pos)[0]),
                              static_cast<double>((*obs_pos)[1]),
                              static_cast<double>((*obs_pos)[2]));
        cv::Mat R32 = CoordinateTransform::eulerToRotationMatrix(
            (*obs_euler)[0], (*obs_euler)[1], (*obs_euler)[2]);
        R32.convertTo(R_, CV_64F);
        ++detect_count_;
        valid_ = detect_count_ >= params_.minDetectCount;
        last_obs_ts_ = t;
        return valid_;
    }

    // 本帧无观测：连续超过重置时间 → 重置（下次观测重新累计识别帧数）
    if (valid_ || detect_count_ > 0) {
        const double since =
            std::chrono::duration<double>(t - last_obs_ts_).count();
        if (since > params_.resetTimeoutSec) {
            reset();
            return false;
        }
    }
    return valid_;
}

void FirstObjectTracker::reset() {
    detect_count_ = 0;
    valid_ = false;
    position_ = cv::Vec3d(0, 0, 0);
    R_ = cv::Mat();
    last_obs_ts_ = TimePoint{};
}

std::unique_ptr<std::function<std::vector<cv::Point3f>(double)>>
FirstObjectTracker::capturePosePredictor() const {
    if (!valid_) return nullptr;

    // 完全捕捉当前保存的位姿（值拷贝，独立于后续变化）；无运动模型，
    // 始终返回该物体最新位置，仅 1 个物体。
    const cv::Vec3d pos = position_;
    return std::make_unique<std::function<std::vector<cv::Point3f>(double)>>(
        [pos](double) -> std::vector<cv::Point3f> {
            return {cv::Point3f(static_cast<float>(pos[0]),
                                static_cast<float>(pos[1]),
                                static_cast<float>(pos[2]))};
        });
}

}  // namespace sp_ekf
