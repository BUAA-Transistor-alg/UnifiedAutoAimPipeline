// GimbalOutputForBigSmallYaw.cpp — 新构型（大/小双 yaw）云台控制输出模式实现
#include "common/BigSmallYaw/GimbalOutputForBigSmallYaw.h"

#include <algorithm>
#include <cmath>

#include "common/Ballistic/SequencePredictor.h"
#include "common/RobotConfig.h"

namespace {

// 截取前 skip 个元素，保证结果至少有一个元素（与 GimbalOutput 同一语义）：
// - 原始序列非空时，截取后为空则至少保留最后一个元素；
// - 原始序列为空时，补一个 fallback 值（pitch 用 0.0，fire 用 false）。
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

namespace bsy {

GimbalOutputForBigSmallYaw::GimbalOutputForBigSmallYaw(RobotControllerAdapter& ctrl)
    : ctrl_(ctrl),
      splitter_(BigSmallYawSplitterConfig{
          RobotConfig::instance().common.bigSmallYaw.bigSmall.joints.smallMinAngle,
          RobotConfig::instance().common.bigSmallYaw.bigSmall.joints.smallMaxAngle,
          RobotConfig::instance().common.bigSmallYaw.bigSmall.joints.smallCenterAngle,
          RobotConfig::instance().common.bigSmallYaw.bigSmall.robotController.mpc.smallLimitSoftRatio,
          RobotConfig::instance().common.bigSmallYaw.bigSmall.splitter.plannerMaxVelocity,
          RobotConfig::instance().common.bigSmallYaw.bigSmall.splitter.plannerMaxAcceleration,
          RobotConfig::instance().common.bigSmallYaw.bigSmall.splitter.plannerMaxJerk,
          RobotConfig::instance().common.bigSmallYaw.bigSmall.splitter.plannerSubsteps}),
      big_torque_only_(RobotConfig::instance().common.bigSmallYaw.bigSmall.robotController.controller.bigTorqueOnly),
      small_torque_only_(RobotConfig::instance().common.bigSmallYaw.bigSmall.robotController.controller.smallTorqueOnly),
      pitch_seq_lead_(RobotConfig::instance().common.predictSequence.pitchSeqLead),
      fire_seq_lead_(RobotConfig::instance().common.predictSequence.fireSeqLead),
      fire_angle_lower_limit_(RobotConfig::instance().common.predictSequence.fireAngleLowerLimit),
      fire_angle_length_(RobotConfig::instance().common.predictSequence.fireAngleLength),
      dt_control_(RobotConfig::instance().common.dtControl()),
      extra_predict_time_(RobotConfig::instance().common.predictedBallistic.extraPredictTime),
      sentry_(RobotConfig::instance().common.sentryController),
      scan_seq_points_((RobotConfig::instance().common.predictSequence.predictionPoints - 1)
                           * RobotConfig::instance().common.predictSequence.interpolationRefine
                       + 1
                       + RobotConfig::instance().common.predictSequence.exactLeadPoints) {}

bool GimbalOutputForBigSmallYaw::computeFire(double ref, double pred, double threshold) {
    // 角度差先解缠绕到 (-π, π]
    const double diff = std::remainder(ref - pred, 2.0 * M_PI);
    return std::fabs(diff) < threshold;
}

void GimbalOutputForBigSmallYaw::update(const PipelineResult& result, tcs::RobotController*,
                                        OutputContext& ctx) {
    // 无新帧时不重发序列，让 McuMpcController 后台 100Hz 线程正常消费已发送序列
    if (!result.valid) return;

    // ── 直接读取新构型控制器状态（不经流水线）──
    const RobotState st = ctrl_.state();

    // ── 当帧预测（main 弹道线程经 SequencePredictor::predict 写入 ctx）──
    const SequencePredictor::Result& seq = ctx.predict_result;

    // 记录"本帧实际下发给子模组的内容"（供可视化覆盖层；序列类字段取首元素 + 长度）：
    // 每次调用 ctrl_.controller().set(...) 之前调用，保证覆盖层显示的就是真实下发量。
    auto recordSent = [&](const char* kind, bool auto_aim_enable,
                          const std::vector<double>& big_seq,
                          const std::vector<double>& small_seq,
                          const std::vector<double>& pitch_seq,
                          const std::vector<bool>& fire_seq,
                          bool integral_enable) {
        ctx.bsy_sent = OutputContext::BigSmallSent{};
        ctx.bsy_sent.valid             = true;
        ctx.bsy_sent.kind              = kind;
        ctx.bsy_sent.auto_aim_enable   = auto_aim_enable;
        ctx.bsy_sent.big_torque_only   = big_torque_only_;
        ctx.bsy_sent.small_torque_only = small_torque_only_;
        ctx.bsy_sent.integral_enable   = integral_enable;
        ctx.bsy_sent.big_yaw_front     = big_seq.empty()   ? 0.0   : big_seq.front();
        ctx.bsy_sent.small_yaw_front   = small_seq.empty() ? 0.0   : small_seq.front();
        ctx.bsy_sent.pitch_front       = pitch_seq.empty() ? 0.0   : pitch_seq.front();
        ctx.bsy_sent.fire_front        = fire_seq.empty()  ? false : fire_seq.front();
        ctx.bsy_sent.big_len           = (int)big_seq.size();
        ctx.bsy_sent.small_len         = (int)small_seq.size();
        ctx.bsy_sent.pitch_len         = (int)pitch_seq.size();
        ctx.bsy_sent.fire_len          = (int)fire_seq.size();
    };

    if (seq.valid && !seq.items.empty()) {
        holding_ = false;

        // ── fire 序列：用 MPC 的**小 yaw** 参考/预测序列逐对判定（与旧版 yaw 通道一致）──
        std::vector<bool> fire_seq;
        bool mpc_available = false;
        const size_t ns = std::min(st.ref_small_azimuth_seq.size(),
                                   st.pred_small_azimuth_seq.size());
        if (ns > 0) {
            // 动态阈值：基于首个序列元素瞄准目标与 yaw 系原点（小 yaw 轴中心）在
            // world xy 平面的投影距离
            double threshold = fire_angle_lower_limit_;
            const cv::Vec3f yaw_origin = seq.yaw_world_origin;
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
                const bool track_ok = computeFire(st.ref_small_azimuth_seq[k],
                                                  st.pred_small_azimuth_seq[k], threshold);
                // 条件2（需求5，仅 fast_target 帧）：该火控点命中时刻有目标（装甲板）在
                // 枪线上（匀速旋转模型；非 fast_target 时恒 true，保持原行为）
                const bool line_ok = SequencePredictor::fastGunLineOk(
                    seq, (int)k, extra_predict_time_, dt_control_);
                fire_seq.push_back(track_ok && line_ok);   // 两个条件都满足才开火
            }
        }

        // ── 瞄准方位角序列（= 每个返回点的小 yaw 输出目标世界方位角）──
        std::vector<double> aim_azimuth;
        aim_azimuth.reserve(seq.items.size());
        for (const auto& item : seq.items) aim_azimuth.push_back((double)item.yaw);

        // ── 大小 yaw 拆分：平滑大 yaw 轨迹 + 越软限位时的无限幅跳变 ──
        // 初值取上一轮计划（按该帧 shared_frame_timestamp 与上一轮的时间差插值）
        const BigSmallYawSplitter::Output sp =
            splitter_.split(aim_azimuth, dt_control_, st.yaw_small_azimuth, result.frame_timestamp);

        // ── pitch / fire 序列截取（yaw 两路序列不截取：拆分器输出已与预测序列等长）──
        std::vector<double> pitch_seq;
        pitch_seq.reserve(seq.items.size());
        for (const auto& item : seq.items) pitch_seq.push_back((double)item.pitch);
        const std::vector<double> pitch_out = truncateKeepLast(pitch_seq, pitch_seq_lead_, 0.0);
        const std::vector<bool>   fire_out  = truncateKeepLast(fire_seq, fire_seq_lead_, false);

        // ── 哨兵扫描控制器：本帧有有效预测 → 复位“无目标”计时并退出扫描；
        //    同时记录本帧下发序列首值，供进入扫描前的保持段填充整条序列 ──
        sentry_.update(/*valid=*/true, result.frame_timestamp);
        last_valid_big_   = sp.big_azimuth.empty() ? st.yaw_big_azimuth : sp.big_azimuth.front();
        last_valid_small_ = sp.small_azimuth.empty() ? st.yaw_small_azimuth : sp.small_azimuth.front();
        last_valid_pitch_ = pitch_out.front();
        have_last_valid_  = true;

        last_.auto_aim_enable = true;
        last_.predicted_point = seq.first_point;
        last_.predict_time    = seq.first_predict_time;
        last_.big_yaw_seq     = sp.big_azimuth;
        last_.small_yaw_seq   = sp.small_azimuth;
        last_.pitch_seq       = pitch_out;
        last_.fire_seq        = fire_out;
        last_.mpc_available   = mpc_available;
        last_.jump_count      = sp.jump_count;
        last_.unlimited_episodes = sp.unlimited_episodes;
        last_.over_limit      = sp.over_limit;
        last_.theta_small_max_abs = sp.theta_small_max_abs;
        last_.soft_min = sp.soft_min;
        last_.soft_max = sp.soft_max;

        ctx.fire_out = fire_out;   // 回写 fire 序列（可视化取首元素控制井形叉丝颜色）
        // 回写拆分器诊断（可视化叠加“大小 yaw 角 / 参考 / 越限标志”）
        ctx.split_diag.valid            = true;
        ctx.split_diag.jump_count       = sp.jump_count;
        ctx.split_diag.unlimited_episodes = sp.unlimited_episodes;
        ctx.split_diag.over_limit       = sp.over_limit;
        ctx.split_diag.theta_small_max_abs = sp.theta_small_max_abs;
        ctx.split_diag.soft_min         = sp.soft_min;
        ctx.split_diag.soft_max         = sp.soft_max;
        ctx.split_diag.big_ref_front    = sp.big_azimuth.empty() ? 0.0 : sp.big_azimuth.front();
        ctx.split_diag.small_ref_front  = sp.small_azimuth.empty() ? 0.0 : sp.small_azimuth.front();

        // 序列 set：{自动瞄准开, 大 yaw 仅力矩, 小 yaw 仅力矩, ψ_big 序列, ψ_small 序列,
        //            pitch 序列, fire 序列, 积分补偿}
        recordSent("predict", /*auto_aim_enable=*/true, sp.big_azimuth, sp.small_azimuth,
                   pitch_out, fire_out, seq.integral_enable);
        ctrl_.controller().set(/*auto_aim_enable=*/true, big_torque_only_, small_torque_only_,
                               sp.big_azimuth, sp.small_azimuth, pitch_out, fire_out,
                               /*integral_enable=*/seq.integral_enable);
    } else {
        // ── 预测不可用：自瞄关闭 ──
        //  - 哨兵扫描控制器关闭（enabled = false）：**原行为完全不变**——大/小 yaw
        //    保持当前严格反解世界方位角；
        //  - 开启且已进入扫描模式：生成扫描序列——大/小 yaw 取**同一条**序列
        //    （同一个世界方位角，不保持关节角差），pitch 走锯齿波
        //    （见 common/SentryController.h），auto_aim_enable = true、fire 全 false；
        //  - 开启但尚未超过 idle_timeout_sec：保持段——用上一个有效输出序列首值
        //    填充整条序列（从未有过有效输出时退化为本帧实测角度），自瞄关闭。
        sentry_.update(/*valid=*/false, result.frame_timestamp);

        last_ = LastOutput{};
        ctx.split_diag = OutputContext::BigSmallSplitDiag{};
        if (sentry_.enabled()) {
            const int n = std::max(1, scan_seq_points_);
            std::vector<double> big_seq, small_seq, pitch_scan_out;
            bool auto_aim = false;
            if (sentry_.scanning()) {
                // 大/小 yaw 使用**同一条**目标序列（同一个世界方位角，不保持关节角差）：
                // 小 yaw 参考关节角为 0，两级同步旋转到同一角度。基准取本帧实测小 yaw
                // 输出方位角（即当前枪线方向，"当前角度"），再按 max_dev 限幅。
                sentry_.buildYawSequence(n, dt_control_, st.yaw_small_azimuth,
                                         result.frame_timestamp, big_seq);
                small_seq = big_seq;
                sentry_.buildPitchSequence(n, dt_control_, result.frame_timestamp, pitch_scan_out);
                auto_aim = true;
            } else {
                const double hold_big   = have_last_valid_ ? last_valid_big_ : st.yaw_big_azimuth;
                const double hold_small = have_last_valid_ ? last_valid_small_ : st.yaw_small_azimuth;
                const double hold_pitch = have_last_valid_ ? last_valid_pitch_ : st.pitch_joint;
                big_seq.assign((size_t)n, hold_big);
                small_seq.assign((size_t)n, hold_small);
                pitch_scan_out.assign((size_t)n, hold_pitch);
            }
            std::vector<bool> fire_scan_out((size_t)n, false);
            last_.auto_aim_enable = auto_aim;
            last_.big_yaw_seq     = big_seq;
            last_.small_yaw_seq   = small_seq;
            last_.pitch_seq       = pitch_scan_out;
            last_.fire_seq        = fire_scan_out;
            ctx.fire_out = fire_scan_out;   // 预测不可用：无有效 fire（首元素 false）
            recordSent(sentry_.scanning() ? "scan" : "hold(sentry)", auto_aim,
                       big_seq, small_seq, pitch_scan_out, fire_scan_out, /*integral_enable=*/false);
            ctrl_.controller().set(auto_aim, big_torque_only_, small_torque_only_,
                                   big_seq, small_seq, pitch_scan_out, fire_scan_out,
                                   /*integral_enable=*/false);
        } else {
            // 原行为（未开启哨兵控制器时与引入本功能前完全一致）
            const double hold_big   = st.yaw_big_azimuth;
            const double hold_small = st.yaw_small_azimuth;
            const double hold_pitch = st.pitch_joint;
            ctx.fire_out = std::vector<bool>{false};
            const std::vector<double> hold_big_seq{hold_big};
            const std::vector<double> hold_small_seq{hold_small};
            const std::vector<double> hold_pitch_seq{hold_pitch};
            const std::vector<bool>   hold_fire_seq{false};
            recordSent("hold", /*auto_aim_enable=*/false, hold_big_seq, hold_small_seq,
                       hold_pitch_seq, hold_fire_seq, /*integral_enable=*/false);
            ctrl_.controller().set(/*auto_aim_enable=*/false, big_torque_only_, small_torque_only_,
                                   hold_big_seq, hold_small_seq, hold_pitch_seq, hold_fire_seq,
                                   /*integral_enable=*/false);
        }

        // 持续保持超过一个规划时域（一个序列时长）后，拆分器的“上一轮计划”已无参考价值：
        // 复位，重新捕获目标时从当前瞄准序列重新起步（避免用陈旧轨迹当初值）
        if (!holding_) {
            holding_ = true;
            hold_start_ = result.frame_timestamp;
        } else {
            const double hold_s = std::chrono::duration<double>(
                result.frame_timestamp - hold_start_).count();
            const size_t horizon_pts = (size_t)std::max(1, RobotConfig::instance().common.predictSequence.predictionPoints);
            if (hold_s > (double)horizon_pts * dt_control_) {
                splitter_.reset();
            }
        }
    }
}

} // namespace bsy
