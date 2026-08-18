// InputController.cpp — 云台输入控制器（序列输入模式）实现
#include "InputController.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "RobotConfig.h"

namespace {

// 截取前 skip 个元素，保证结果至少有一个元素：
// - 原始序列非空时，截取后为空则至少保留最后一个元素（截取只丢弃开头，末尾元素天然保留）；
// - 原始序列为空时，补一个 fallback 值（yaw/pitch 用 0.0，fire 用 false）。
template <typename T>
std::vector<T> truncateKeepLast(const std::vector<T>& seq, int skip, const T& fallback) {
    if (seq.empty()) {
        return std::vector<T>{fallback};
    }
    const size_t s = std::min((size_t)std::max(0, skip), seq.size());
    if (s >= seq.size()) {
        return std::vector<T>{seq.back()};   // 截取会清空：至少保留最后一个元素
    }
    return std::vector<T>(seq.begin() + (long)s, seq.end());
}

} // namespace

InputController::InputController(RobotController& rc, std::shared_ptr<GimbalSolver> gimbal)
    : rc_(rc),
      gimbal_(gimbal),
      pred_ballistic_(gimbal),
      extra_predict_time_(RobotConfig::instance().predictedBallistic.extraPredictTime),
      dt_control_(RobotConfig::instance().robotController.dtControl),
      prediction_seq_len_(RobotConfig::instance().inputController.predictionSeqLen),
      pitch_seq_lead_(RobotConfig::instance().inputController.pitchSeqLead),
      fire_seq_lead_(RobotConfig::instance().inputController.fireSeqLead),
      pitch_bias_(RobotConfig::instance().inputController.pitchBias),
      fire_angle_lower_limit_(RobotConfig::instance().inputController.fireAngleLowerLimit),
      fire_angle_length_(RobotConfig::instance().inputController.fireAngleLength) {
    if (prediction_seq_len_ < 1) {
        throw std::invalid_argument("InputController: prediction_seq_len 必须 >= 1");
    }
    if (pitch_seq_lead_ < 0 || pitch_seq_lead_ >= prediction_seq_len_) {
        throw std::invalid_argument("InputController: pitch_seq_lead 必须小于 prediction_seq_len");
    }
    if (fire_seq_lead_ < 0) {
        throw std::invalid_argument("InputController: fire_seq_lead 必须 >= 0");
    }
}

bool InputController::computeFire(double ref, double pred, double threshold) {
    // 角度差先解缠绕到 (-π, π]
    const double diff = std::remainder(ref - pred, 2.0 * M_PI);
    return std::fabs(diff) < threshold;
}

void InputController::update(const RobotController::State& st, const Predictor& predictor) {
    const double chassis_yaw = st.strict.imu_euler_yaw - st.strict.yaw_pos;  // 底盘 yaw 修正

    // ── 预测序列：i = 1..n，以 extra + i*dt 作为额外预测时间逐个完整解算 ──
    std::vector<double> yaw_seq, pitch_seq;
    yaw_seq.reserve((size_t)prediction_seq_len_);
    pitch_seq.reserve((size_t)prediction_seq_len_);
    bool   first_success = false;
    cv::Vec3f first_point;
    double first_time = 0.0;
    for (int i = 1; i <= prediction_seq_len_; ++i) {
        const double extra_i = extra_predict_time_ + i * dt_control_;
        const PredictedBallisticSolver::Result res = pred_ballistic_.solve(predictor, extra_i);
        yaw_seq.push_back(res.gimbal.yaw + chassis_yaw);              // yaw 序列原样（叠加底盘修正）
        pitch_seq.push_back(res.gimbal.pitch + pitch_bias_);          // pitch 叠加偏置
        if (i == 1) {   // 首个序列元素：供绘制 / fire 距离
            first_success = res.success;
            first_point  = res.predicted_point;
            first_time   = res.predict_time;
        }
    }

    // ── fire 序列：ref/pred 每一对按当前方法计算 ──
    std::vector<bool> fire_seq;
    last_.mpc_available = false;
    const size_t ns = std::min(st.mpc.ref_sequence.size(), st.mpc.pred_sequence.size());
    if (ns > 0) {
        // 动态阈值：基于首个序列元素瞄准目标与 yaw 系原点在 world xy 平面的投影距离
        double threshold = fire_angle_lower_limit_;
        if (gimbal_) {
            const cv::Vec3f yaw_origin = gimbal_->yawWorldOrigin();
            const cv::Vec3f target     = first_point;
            const double dist_xy = std::hypot((double)target[0] - (double)yaw_origin[0],
                                              (double)target[1] - (double)yaw_origin[1]);
            if (dist_xy > 1e-6) {
                threshold = std::max(threshold, fire_angle_length_ / dist_xy);
            }
        }
        last_.mpc_available  = true;
        last_.fire_threshold = threshold;
        fire_seq.reserve(ns);
        for (size_t k = 0; k < ns; ++k) {
            fire_seq.push_back(computeFire(st.mpc.ref_sequence[k], st.mpc.pred_sequence[k],
                                           threshold));
        }
    }

    // ── 截取并保证每个传出序列至少有一个元素 ──
    // yaw 原样返回（skip=0）；pitch 截取第 m 个之后；fire 截取第 o 个之后。
    // 截取前序列为空时补一个 fallback（yaw/pitch 补 0.0，fire 补 false）。
    const std::vector<double> yaw_out   = truncateKeepLast(yaw_seq, 0, 0.0);
    const std::vector<double> pitch_out = truncateKeepLast(pitch_seq, pitch_seq_lead_, 0.0);
    const std::vector<bool>   fire_out  = truncateKeepLast(fire_seq, fire_seq_lead_, false);

    // ── 记录并发送 ──
    last_.auto_aim_enable = first_success;
    last_.predicted_point = first_point;
    last_.predict_time    = first_time;
    last_.yaw_seq         = yaw_out;
    last_.pitch_seq       = pitch_out;
    last_.fire_seq        = fire_out;
    rc_.set(first_success, /*yaw_torque_only_mode=*/false,
            yaw_out, pitch_out, fire_out);
}
