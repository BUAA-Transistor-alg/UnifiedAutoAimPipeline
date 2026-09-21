// GimbalOutput.cpp — 云台控制输出模式实现
#include "common/Output/GimbalOutput.h"
#include "common/Ballistic/SequencePredictor.h"
#include "common/RobotConfig.h"
#include "common/Debug/AimSwitchLog.h"

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

// 选靶切换日志（AimSwitchLog）辅助：预测器来源名
const char* aimLogSourceName(const SequencePredictor::PredictorSource& s) {
    switch (s.kind) {
        case SequencePredictor::PredictorSource::Kind::ARMOR:      return "Armor";
        case SequencePredictor::PredictorSource::Kind::POWER_RUNE: return "PowerRune";
        default:                                                   return "None";
    }
}

}  // namespace

GimbalOutput::GimbalOutput(tcs::RobotController& rc)
    : rc_(rc),
      yaw_torque_only_mode_(
          RobotConfig::instance().common.singleYawRobotController().yawTorqueOnlyMode),
      pitch_seq_lead_(RobotConfig::instance().common.predictSequence.pitchSeqLead),
      fire_seq_lead_(RobotConfig::instance().common.predictSequence.fireSeqLead),
      fire_angle_lower_limit_(RobotConfig::instance().common.predictSequence.fireAngleLowerLimit),
      fire_angle_length_(RobotConfig::instance().common.predictSequence.fireAngleLength),
      extra_predict_time_(RobotConfig::instance().common.predictedBallistic.extraPredictTime),
      dt_control_(RobotConfig::instance().common.dtControl()),
      sentry_(RobotConfig::instance().common.sentryController),
      scan_seq_points_((RobotConfig::instance().common.predictSequence.predictionPoints - 1)
                           * RobotConfig::instance().common.predictSequence.interpolationRefine
                       + 1
                       + RobotConfig::instance().common.predictSequence.exactLeadPoints) {}

bool GimbalOutput::computeFire(double ref, double pred, double threshold) {
    // 角度差先解缠绕到 (-π, π]
    const double diff = std::remainder(ref - pred, 2.0 * M_PI);
    return std::fabs(diff) < threshold;
}

void GimbalOutput::update(const PipelineResult& result, tcs::RobotController*,
                          OutputContext& ctx)
{
    // 无新帧时不重发序列，让 McuMpcController 后台 100Hz 线程正常消费已发送序列
    if (!result.valid) return;

    // ── 直接读取串口/MPC 状态（不经流水线）──
    const tcs::RobotController::State st = rc_.getState();

    // ── 取当帧预测（main 弹道线程经 SequencePredictor::predict 写入 ctx：
    //    含预测云台控制序列 + 瞄准点 + yaw 系原点）──
    const SequencePredictor::Result& seq = ctx.predict_result;

    if (seq.valid && !seq.items.empty()) {
        // ── fire 序列：ref/pred 每一对按当前方法计算 ──
        std::vector<bool> fire_seq;
        bool mpc_available = false;
        const size_t ns = std::min(st.mpc.ref_sequence.size(), st.mpc.pred_sequence.size());
        if (ns > 0) {
            // 动态阈值：基于首个序列元素瞄准目标与 yaw 系原点在 world xy 平面的投影距离
            double threshold = fire_angle_lower_limit_;
            const cv::Vec3f yaw_origin = seq.yaw_world_origin;   // 随当帧预测结果传入
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
                // 条件1：MPC 预测轨迹与目标轨迹（参考）误差在动态角度阈值内
                const bool track_ok = computeFire(st.mpc.ref_sequence[k], st.mpc.pred_sequence[k],
                                                  threshold);
                // 条件2（需求5，仅 fast_target 帧）：该火控点命中时刻有目标（装甲板）在
                // 枪线上（匀速旋转模型；非 fast_target 时恒 true，保持原行为）
                const bool line_ok = SequencePredictor::fastGunLineOk(
                    seq, (int)k, extra_predict_time_, dt_control_);
                fire_seq.push_back(track_ok && line_ok);   // 两个条件都满足才开火
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

        // ── 哨兵扫描控制器：本帧有有效预测 → 复位“无目标”计时并退出扫描；
        //    同时记录本帧下发序列首值，供进入扫描前的保持段填充整条序列 ──
        sentry_.update(/*valid=*/true, result.frame_timestamp);
        last_valid_yaw_   = yaw_out.front();
        last_valid_pitch_ = pitch_out.front();
        have_last_valid_  = true;

        last_.auto_aim_enable = seq.valid;
        last_.predicted_point = seq.first_point;
        last_.predict_time    = seq.first_predict_time;
        last_.yaw_seq         = yaw_out;
        last_.pitch_seq       = pitch_out;
        last_.fire_seq        = fire_out;
        last_.mpc_available   = mpc_available;
        ctx.fire_out = fire_out;   // 回写 fire 序列（可视化取首元素控制井形叉丝颜色）
        rc_.set(/*auto_aim_enable=*/true, /*yaw_torque_only_mode=*/yaw_torque_only_mode_,
                yaw_out, pitch_out, fire_out,
                /*integral_enable=*/seq.integral_enable);

        // ── 选靶切换日志（common/Debug/AimSwitchLog）：本帧的预瞄点计划 ──
        // 行 = 控制器节拍 tick（= yaw 序列下标）：该节拍的预瞄点（目标索引 / world 坐标）、
        // fire 判定与**实际下发**的 fire 位（含 fire_seq_lead 偏移）、以及控制器真正
        // 执行到了哪一拍（ticks_since_set 为上一帧序列已消费的节拍数，减 1 即当前执行点）。
        // yaw 序列不截取、pitch/fire 截取 —— 故第 k 拍实际下发的是「第 k 点 yaw +
        // 第 k+pitch_lead 点 pitch + 第 k+fire_lead 点 fire」，两部分都记在 CSV 里。
        {
            AimSwitchLog::Plan plan;
            plan.frame_timestamp = result.frame_timestamp;
            plan.source          = aimLogSourceName(result.predictor.source);
            plan.sequence_sent   = true;
            plan.yaw_lead        = 0;
            plan.pitch_lead      = (int)pitch_seq_lead_;
            plan.fire_lead       = (int)fire_seq_lead_;
            plan.ticks_since_set = (unsigned long long)st.mpc.ticks_since_set;
            plan.exec_tick       = (st.mpc.ticks_since_set > 0)
                                       ? (int)(st.mpc.ticks_since_set - 1) : -1;
            plan.actual_yaw      = st.fused.yaw_pos;
            plan.actual_pitch    = st.mcu.pitch_angle;
            const size_t n_ticks = std::max(yaw_out.size(), seq.items.size());
            plan.ticks.resize(n_ticks);
            for (size_t k = 0; k < n_ticks; ++k) {
                AimSwitchLog::PlanPoint& tp = plan.ticks[k];
                tp.tick = (int)k;
                // 该节拍的预瞄点（仅 yaw 序列范围内有点；items 通常与 yaw_out 等长）
                if (k < seq.items.size()) {
                    const SequencePredictor::Item& it = seq.items[k];
                    tp.has_point    = true;
                    tp.target_index = it.target_index;
                    tp.success      = it.success;
                    tp.x            = it.predicted_point[0];
                    tp.y            = it.predicted_point[1];
                    tp.z            = it.predicted_point[2];
                    tp.gimbal_yaw   = it.gimbal_yaw;
                    tp.gimbal_pitch = it.gimbal_pitch;
                    tp.flight_time  = it.flight_time;
                }
                // 该节拍实际下发的 yaw / pitch（pitch 序列被截掉前 pitch_seq_lead_ 个：
                // 控制器的第 k 拍拿到的是 pitch_out[k]，即预瞄点 (k + pitch_seq_lead_) 的
                // pitch；pitch 序列先耗尽时控制器保持上一次的 pitch 值）
                if (k < yaw_out.size()) {
                    tp.sent       = true;
                    tp.yaw_sent   = yaw_out[k];
                }
                if (k < pitch_out.size()) tp.pitch_sent = pitch_out[k];
                // 该节拍预瞄点自身的 fire 判定（MPC ref vs pred，未截取）
                if (k < fire_seq.size()) tp.fire_plan = fire_seq[k] ? 1 : 0;
                // 该节拍实际下发的 fire 位（= fire_seq[k + fire_seq_lead_]）
                if (k < fire_out.size()) tp.fire_sent = fire_out[k] ? 1 : 0;
            }
            AimSwitchLog::instance().plan(plan);
        }
    } else {
        // 预测不可用：自瞄关闭。
        //  - 哨兵扫描控制器关闭（enabled = false）：**原行为完全不变**——用单元素
        //    序列保持当前严格反解位置；
        //  - 开启且已进入扫描模式：按配置生成扫描序列（yaw 匀速旋转 + pitch 锯齿波，
        //    见 common/SentryController.h），auto_aim_enable = true、fire 全 false；
        //  - 开启但尚未超过 idle_timeout_sec：保持段——用上一个有效输出序列的首值
        //    填充整条序列（从未有过有效输出时退化为本帧严格反解位置），自瞄关闭。
        sentry_.update(/*valid=*/false, result.frame_timestamp);

        last_ = LastOutput{};
        std::vector<double> yaw_out, pitch_out;
        if (sentry_.enabled()) {
            const int n = std::max(1, scan_seq_points_);
            bool auto_aim = false;
            if (sentry_.scanning()) {
                sentry_.buildYawSequence(n, dt_control_, st.strict.yaw_pos,
                                         result.frame_timestamp, yaw_out);
                sentry_.buildPitchSequence(n, dt_control_, result.frame_timestamp, pitch_out);
                auto_aim = true;
            } else {
                const double hold_yaw   = have_last_valid_ ? last_valid_yaw_ : st.strict.yaw_pos;
                const double hold_pitch = have_last_valid_ ? last_valid_pitch_ : st.strict.pitch_angle;
                yaw_out.assign((size_t)n, hold_yaw);
                pitch_out.assign((size_t)n, hold_pitch);
            }
            last_.auto_aim_enable = auto_aim;
            last_.yaw_seq         = yaw_out;
            last_.pitch_seq       = pitch_out;
            last_.fire_seq.assign((size_t)n, false);
            ctx.fire_out = last_.fire_seq;   // 预测不可用：无有效 fire（首元素 false）
            rc_.set(auto_aim, /*yaw_torque_only_mode=*/yaw_torque_only_mode_,
                    yaw_out, pitch_out, last_.fire_seq,
                    /*integral_enable=*/false);
        } else {
            // 原行为（未开启哨兵控制器时与引入本功能前完全一致）
            const double hold_yaw   = st.strict.yaw_pos;
            const double hold_pitch = st.strict.pitch_angle;
            ctx.fire_out = std::vector<bool>{false};   // 预测不可用：无有效 fire（首元素为 false）
            rc_.set(/*auto_aim_enable=*/false, /*yaw_torque_only_mode=*/yaw_torque_only_mode_,
                    std::vector<double>{hold_yaw},
                    std::vector<double>{hold_pitch},
                    std::vector<bool>{false},
                    /*integral_enable=*/false);
        }
    }
}
