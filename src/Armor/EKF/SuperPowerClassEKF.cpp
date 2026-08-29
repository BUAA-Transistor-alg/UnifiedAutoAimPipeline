#include "Armor/EKF/SuperPowerClassEKF.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

#include "common/TransformTree/CoordinateTransform.h"

namespace sp_ekf {

namespace {
// 原接口（SuperPowerPredictor）的项目层位置单位约定为毫米。
constexpr double kMillimetersPerMeter = 1000.0;
constexpr double kPi = 3.141592653589793238462643383279502884;
}  // namespace

std::shared_ptr<YAML::Node> ClassEKF::buildConfig(const Params& params) {
    // 经原接口（SuperPowerPredictor）的 YAML 配置机制下发全部可调参数：
    // 段内字段与 config armor.super_power_ekf 一一对应；YAML 块只覆盖这些
    // SP 常量，集中管理配置来源，且不读取旧估计器配置。
    std::shared_ptr<YAML::Node> node =
        std::make_shared<YAML::Node>(YAML::Load(R"(
superpower_ekf:
  min_detect_count: 5
  max_temp_lost_count: 15
  max_dt_s: 0.1
  initial_radius_m: 0.2
  armor_num: 4
  angular_velocity_fit:
    window_s: 0.20
    min_samples: 4
  joint_update:
    enabled: false
    max_joint_nis: 20.09
    max_secondary_position_error_m: 0.45
    max_secondary_angle_error_rad: 0.80
    measurement_variance_scale: 1.5
    angle_variance_scale: 4.0
)"));
    YAML::Node sp = (*node)["superpower_ekf"];
    sp["min_detect_count"] = params.minDetectCount;
    sp["max_temp_lost_count"] = params.maxTempLostCount;
    sp["max_dt_s"] = params.maxDtSec;
    sp["initial_radius_m"] = params.initialRadiusM;
    sp["armor_num"] = params.armorNum;
    sp["angular_velocity_fit"]["window_s"] = params.angularVelocityFitWindowSec;
    sp["angular_velocity_fit"]["min_samples"] = params.angularVelocityFitMinSamples;
    YAML::Node joint = sp["joint_update"];
    joint["enabled"] = params.jointUpdateEnabled;
    joint["max_joint_nis"] = params.jointMaxNis;
    joint["max_secondary_position_error_m"] = params.jointMaxSecondaryPositionErrorM;
    joint["max_secondary_angle_error_rad"] = params.jointMaxSecondaryAngleErrorRad;
    joint["measurement_variance_scale"] = params.jointMeasurementVarianceScale;
    joint["angle_variance_scale"] = params.jointAngleVarianceScale;
    return node;
}

ClassEKF::ClassEKF(const Params& params)
    : params_(params),
      config_(buildConfig(params_)) {}

bool ClassEKF::processFrame(const std::vector<cv::Vec3f>& world_pos,
                            const std::vector<cv::Vec3f>& world_euler,
                            const TimePoint& t) {
    // 统一换算成原接口的项目层观测：位置单位毫米（米×1000），yaw 为项目 yaw
    // （弧度，原接口内部再转 SP 外法线角），t 为秒（steady_clock 相对时间）。
    const double t_sec =
        std::chrono::duration<double>(t.time_since_epoch()).count();

    // 收集本帧有效观测（跳过 PnP 失败填充的 (0,0,0)；world_pos 与 world_euler
    // 一一对应，对应 stage4 CategoryData），最多取前两个作为主/副板。
    std::vector<EKFTargetObservation> observations;
    observations.reserve(2);
    const std::size_t count = std::min(world_pos.size(), world_euler.size());
    for (std::size_t k = 0; k < count; ++k) {
        const cv::Vec3f& wp = world_pos[k];
        if (wp[0] != 0.0f || wp[1] != 0.0f || wp[2] != 0.0f) {
            EKFTargetObservation obs;
            obs.x = static_cast<double>(wp[0]) * kMillimetersPerMeter;
            obs.y = static_cast<double>(wp[1]) * kMillimetersPerMeter;
            obs.z = static_cast<double>(wp[2]) * kMillimetersPerMeter;
            obs.yaw = static_cast<double>(world_euler[k][0]);  // 项目 yaw（弧度）
            obs.t = t_sec;
            observations.push_back(obs);
            if (observations.size() == 2) break;
        }
    }

    if (!observations.empty()) {
        const EKFTargetObservation& primary = observations.front();
        if (!predictor_) {
            // 初始化：首条观测构造 SuperPowerPredictor（内部建立 Target，dt=0）
            predictor_ = std::make_unique<SuperPowerPredictor>(
                primary,
                params_.initialRadiusM * kMillimetersPerMeter,
                config_);
        } else if (observations.size() >= 2) {
            // 双板（联合）观测更新：主/副板为同一帧（同一时间戳），门控失败时
            // 由原接口内部自动回退为主板单板更新
            predictor_->updatePair(primary, observations[1]);
        } else {
            // 单板观测更新：由原接口内部按“距上一次更新”的 dt 完成 predict+update
            predictor_->update(primary);
        }
        last_obs_ts_ = t;
    } else if (predictor_) {
        // 本帧无观测（失检）：连续超过超时阈值 → 经原接口 clear() 销毁内部目标
        // （下次观测自动重建）；未超时 → missUpdate 仅做时间预测推进运动模型
        const double since =
            std::chrono::duration<double>(t - last_obs_ts_).count();
        if (since > params_.observationLostTimeoutSec) {
            predictor_->clear();
        } else {
            predictor_->missUpdate(t_sec);
        }
    }

    // 帧末统一刷新 state 快照：仅当原接口 ready()（TRACKING 且目标存在）时
    // 用 state() 保存后验状态，并返回 state 是否可用；同步导出车体中心坐标与
    // 旋转矩阵（与 OutpostESEKF 的 getPosition/getRotationMatrix 形式一致）。
    if (predictor_ && predictor_->ready()) {
        state_ = predictor_->state();
        state_available_ = true;
        position_ = cv::Vec3d(state_.center_x / kMillimetersPerMeter,
                              state_.center_y / kMillimetersPerMeter,
                              state_.center_z / kMillimetersPerMeter);
        // 车体仅跟踪绕世界系 z 轴的 yaw（pitch/roll 无观测），旋转矩阵 = Rz(yaw)
        cv::Mat R32 = CoordinateTransform::eulerToRotationMatrix(
            static_cast<float>(state_.yaw), 0.0f, 0.0f);
        R32.convertTo(R_, CV_64F);
    } else {
        state_available_ = false;
    }
    return state_available_;
}

std::unique_ptr<std::function<std::vector<cv::Point3f>(double)>>
ClassEKF::capturePosePredictor() const {
    if (!state_available_) return nullptr;

    // 完全捕捉当前后验状态（值拷贝，独立于后续滤波变化）。
    const EKFTargetState s = state_;

    // 匀速平移 + 匀角速旋转外推，重建 4 块装甲中心位置（项目几何
    // p = c + r*[sin(yaw), -cos(yaw)]；1/3 号装甲使用 r2 与高度偏置 h）。
    return std::make_unique<std::function<std::vector<cv::Point3f>(double)>>(
        [s](double dt) -> std::vector<cv::Point3f> {
            constexpr int kArmorNum = 4;
            const double center_x = s.center_x + s.center_vx * dt;
            const double center_y = s.center_y + s.center_vy * dt;
            const double center_z = s.center_z + s.center_vz * dt;
            const double phase = s.yaw + s.w * dt;

            std::vector<cv::Point3f> armors;
            armors.reserve(kArmorNum);
            for (int id = 0; id < kArmorNum; ++id) {
                const double yaw_i = phase + id * 2.0 * kPi / kArmorNum;
                const bool use_l_h = (id == 1 || id == 3);
                const double r = use_l_h ? s.r2 : s.r1;
                const double px = center_x + r * std::sin(yaw_i);
                const double py = center_y - r * std::cos(yaw_i);
                const double pz = use_l_h ? center_z + s.h : center_z;
                armors.emplace_back(
                    static_cast<float>(px / kMillimetersPerMeter),
                    static_cast<float>(py / kMillimetersPerMeter),
                    static_cast<float>(pz / kMillimetersPerMeter));
            }
            return armors;
        });
}

void ClassEKF::reset() {
    if (predictor_) predictor_->clear();
    last_obs_ts_ = TimePoint{};
    state_available_ = false;
    position_ = cv::Vec3d(0, 0, 0);
    R_ = cv::Mat();
}

}  // namespace sp_ekf
