// BigSmallYawSplitter.cpp — 大小 yaw 拆分器实现
#include "common/BigSmallYaw/BigSmallYawSplitter.h"

#include <algorithm>
#include <cmath>

namespace bsy {

namespace {
constexpr double kTwoPi = 2.0 * M_PI;

// 把 angle 加减 2π 整数倍，落到与 ref 最近的同一圈
double wrapNear(double angle, double ref) {
    return angle - kTwoPi * std::round((angle - ref) / kTwoPi);
}
} // namespace

BigSmallYawSplitter::BigSmallYawSplitter(const BigSmallYawSplitterConfig& cfg)
    : cfg_(cfg),
      planner_(cfg.plannerMaxVelocity, cfg.plannerMaxAcceleration, cfg.plannerMaxJerk) {
    // 小 yaw 软限位：与子模组 MPC 同一公式（距任一侧限位 (1−ratio)·行程 以内开始惩罚）
    const double span = cfg_.smallMaxAngle - cfg_.smallMinAngle;
    soft_min_ = cfg_.smallMinAngle + (1.0 - cfg_.softLimitRatio) * span;
    soft_max_ = cfg_.smallMaxAngle - (1.0 - cfg_.softLimitRatio) * span;
}

void BigSmallYawSplitter::reset() {
    has_prev_ = false;
    prev_ts_ = std::chrono::steady_clock::time_point{};
    prev_pos_.clear();
    prev_vel_.clear();
    prev_acc_.clear();
}

BigSmallYawSplitter::Output BigSmallYawSplitter::split(
    const std::vector<double>& aim_azimuth, double dt, double psi_small_now,
    const std::chrono::steady_clock::time_point& frame_ts) {
    Output out;
    out.soft_min = soft_min_;
    out.soft_max = soft_max_;
    const size_t n = aim_azimuth.size();
    if (n == 0 || dt <= 0.0) return out;

    // ── 1. 瞄准序列连续化（解卷绕）：首元素对齐到当帧实测 ψ_small 所在圈，
    //       其后按相邻差解卷绕（item.yaw 已 wrap 到 (−π,π]，相邻差很小） ──
    std::vector<double> a(n);
    a[0] = wrapNear(aim_azimuth[0], psi_small_now);
    for (size_t i = 1; i < n; ++i) {
        const double d = aim_azimuth[i] - aim_azimuth[i - 1];
        a[i] = a[i - 1] + (d - kTwoPi * std::round(d / kTwoPi));
    }

    // ── 1b. 规划目标：使小 yaw 关节角尽量等于回中目标 ⇒ 大 yaw 目标 = 瞄准方位角 − 回中目标 ──
    std::vector<double> target(n);
    for (size_t i = 0; i < n; ++i) target[i] = a[i] - cfg_.smallCenterAngle;

    // ── 2. 初值：上一轮计划按“实际经过时间”取插值（超过覆盖区间用最后一点保持）；
    //       位置再做 ±2π 修正到与 target[0] 最近；无上一轮时 = (target[0], 0, 0) ──
    double p0 = target[0], v0 = 0.0, a0 = 0.0;
    if (has_prev_ && !prev_pos_.empty()) {
        const double elapsed = std::max(0.0, std::chrono::duration<double>(frame_ts - prev_ts_).count());
        const size_t last = prev_pos_.size() - 1;
        double idx_f = elapsed / dt;
        if (idx_f > (double)last) idx_f = (double)last;   // 超出覆盖区间：最后一个点保持
        const size_t i0 = (size_t)std::floor(idx_f);
        const size_t i1 = std::min(i0 + 1, last);
        const double f = idx_f - (double)i0;
        p0 = prev_pos_[i0] * (1.0 - f) + prev_pos_[i1] * f;
        v0 = prev_vel_[i0] * (1.0 - f) + prev_vel_[i1] * f;
        a0 = prev_acc_[i0] * (1.0 - f) + prev_acc_[i1] * f;
        p0 = wrapNear(p0, target[0]);
    }

    // ── 3. 以 trajectory_planner 扫描目标序列 → 平滑的大 yaw 轨迹 b[]（含 v/a/j）──
    std::vector<double> b, vel, acc, jerk;
    planner_.scan(target, p0, v0, a0, dt, cfg_.plannerSubsteps, b, vel, acc, jerk);

    // ── 4. 越界修正（“无限幅”动作）：θ_small[i] = a[i] − b[i] 超出软限位时，
    //       把 b[i] 增减最小量使其恰好回到边界，v/a 不变并从该点起重规划后续轨迹 ──
    bool prev_corrected = false;   // 上一个点是否被修正（用于统计连续的无限幅段数）
    for (size_t i = 0; i < n; ++i) {
        const double theta = a[i] - b[i];
        double corrected = b[i];
        if (theta > soft_max_) {
            corrected = a[i] - soft_max_;      // b 向目标方向移动（θ_small 降到 +soft_max）
        } else if (theta < soft_min_) {
            corrected = a[i] - soft_min_;      // θ_small 升到 −soft_min
        } else {
            prev_corrected = false;
            continue;
        }
        ++out.jump_count;
        if (!prev_corrected) ++out.unlimited_episodes;   // 连续修正算同一段“无限幅运动”
        prev_corrected = true;
        out.over_limit = true;
        b[i] = corrected;                       // 位置跳变（速度/加速度保持）
        if (i + 1 < n) {
            planner_.scanFrom(target, i + 1, b[i], vel[i], acc[i], dt, cfg_.plannerSubsteps,
                              b, vel, acc, jerk);
        }
    }

    // ── 5. 输出 ──
    out.big_azimuth = b;                 // ψ_big*（世界方位角序列）
    out.small_azimuth = a;               // ψ_small*（= 瞄准序列）
    out.pos = b;                         // 规划轨迹状态（诊断 + 下一轮初值来源）
    out.vel = vel;
    out.acc = acc;
    out.jerk = jerk;
    out.theta_small.resize(n);
    for (size_t i = 0; i < n; ++i) {
        out.theta_small[i] = a[i] - b[i];
        out.theta_small_max_abs = std::max(out.theta_small_max_abs, std::fabs(out.theta_small[i]));
    }

    // ── 6. 记录本轮计划（下一轮初值）──
    prev_pos_ = out.pos;
    prev_vel_ = out.vel;
    prev_acc_ = out.acc;
    prev_ts_ = frame_ts;
    has_prev_ = true;
    return out;
}

} // namespace bsy
