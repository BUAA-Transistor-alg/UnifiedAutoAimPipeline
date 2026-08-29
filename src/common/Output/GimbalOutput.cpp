// GimbalOutput.cpp — 云台控制输出模式实现
#include "common/Output/GimbalOutput.h"
#include "common/RobotConfig.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

// 截取前 skip 个元素，保证结果至少有一个元素：
// - 原始序列非空时，截取后为空则至少保留最后一个元素；
// - 原始序列为空时，补一个 fallback 值（yaw/pitch 用 0.0，fire 用 false）。
template <typename T>
std::vector<T> truncateKeepLast(const std::vector<T>& seq, int skip, const T& fallback) {
    if (seq.empty()) {
        return std::vector<T>{fallback};
    }
    const size_t s = std::min((size_t)std::max(0, skip), seq.size());
    if (s >= seq.size()) {
        return std::vector<T>{seq.back()};
    }
    return std::vector<T>(seq.begin() + (long)s, seq.end());
}

} // namespace

GimbalOutput::GimbalOutput(SequencePredictor& aim, RobotController& rc)
    : aim_(aim),
      rc_(rc),
      yaw_torque_only_mode_(RobotConfig::instance().common.robotController.yawTorqueOnlyMode),
      pitch_seq_lead_(RobotConfig::instance().common.predictSequence.pitchSeqLead),
      fire_seq_lead_(RobotConfig::instance().common.predictSequence.fireSeqLead),
      fire_angle_lower_limit_(RobotConfig::instance().common.predictSequence.fireAngleLowerLimit),
      fire_angle_length_(RobotConfig::instance().common.predictSequence.fireAngleLength) {}

bool GimbalOutput::computeFire(double ref, double pred, double threshold) {
    // 角度差先解缠绕到 (-π, π]
    const double diff = std::remainder(ref - pred, 2.0 * M_PI);
    return std::fabs(diff) < threshold;
}

void GimbalOutput::update(const PipelineResult& result, RobotController*)
{
    // 无新帧时不重发序列，让 McuMpcController 后台 100Hz 线程正常消费已发送序列
    if (!result.valid) return;

    // ── 直接读取串口/MPC 状态（不经流水线）──
    const RobotController::State st = rc_.getState();

    // ── 取 SequencePredictor 最新预测（main 每帧已调用，含预测云台控制序列 + 瞄准点）──
    const SequencePredictor::Result seq = aim_.latest();

    if (seq.valid && !seq.items.empty()) {
        // ── fire 序列：ref/pred 每一对按当前方法计算 ──
        std::vector<bool> fire_seq;
        bool mpc_available = false;
        const size_t ns = std::min(st.mpc.ref_sequence.size(), st.mpc.pred_sequence.size());
        if (ns > 0) {
            // 动态阈值：基于首个序列元素瞄准目标与 yaw 系原点在 world xy 平面的投影距离
            double threshold = fire_angle_lower_limit_;
            const cv::Vec3f yaw_origin = aim_.yawWorldOrigin();   // 线程安全缓存（弹道线程写入）
            const cv::Vec3f target = seq.first_point;
            const double dist_xy = std::hypot((double)target[0] - (double)yaw_origin[0],
                                              (double)target[1] - (double)yaw_origin[1]);
            if (dist_xy > 1e-6) {
                threshold = std::max(threshold, fire_angle_length_ / dist_xy);
            }
            mpc_available = true;
            last_.fire_threshold = threshold;
            fire_seq.reserve(ns);
            for (size_t k = 0; k < ns; ++k) {
                fire_seq.push_back(computeFire(st.mpc.ref_sequence[k], st.mpc.pred_sequence[k],
                                               threshold));
            }
        }

        // ── 序列组装与截取：yaw 原样；pitch 截第 m 个之后；fire 截第 o 个之后 ──
        std::vector<double> yaw_seq, pitch_seq;
        yaw_seq.reserve(seq.items.size());
        pitch_seq.reserve(seq.items.size());
        for (const auto& item : seq.items) {
            yaw_seq.push_back(item.yaw);
            pitch_seq.push_back(item.pitch);
        }
        const std::vector<double> yaw_out   = truncateKeepLast(yaw_seq, 0, 0.0);
        const std::vector<double> pitch_out = truncateKeepLast(pitch_seq, pitch_seq_lead_, 0.0);
        const std::vector<bool>   fire_out  = truncateKeepLast(fire_seq, fire_seq_lead_, false);

        last_.auto_aim_enable = seq.valid;
        last_.predicted_point = seq.first_point;
        last_.predict_time    = seq.first_predict_time;
        last_.yaw_seq         = yaw_out;
        last_.pitch_seq       = pitch_out;
        last_.fire_seq        = fire_out;
        last_.mpc_available   = mpc_available;
        rc_.set(/*auto_aim_enable=*/true, /*yaw_torque_only_mode=*/yaw_torque_only_mode_,
                yaw_out, pitch_out, fire_out,
                /*integral_enable=*/seq.integral_enable);
    } else {
        // 预测不可用：保持模式（自瞄关闭，目标保持当前严格反解位置）。
        // 序列模式下用单元素序列调用序列 set。
        last_ = LastOutput{};
        const double hold_yaw   = st.strict.yaw_pos;
        const double hold_pitch = st.strict.pitch_angle;
        rc_.set(/*auto_aim_enable=*/false, /*yaw_torque_only_mode=*/yaw_torque_only_mode_,
                std::vector<double>{hold_yaw},
                std::vector<double>{hold_pitch},
                std::vector<bool>{false},
                /*integral_enable=*/false);
    }
}
