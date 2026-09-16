// BigSmallYawTrajectoryPlanner.cpp — 移植自子模组
// python/scripts/trajectory_planner.py（TrajectoryPlanner + StepRefinementWrapper），
// 公式、分支与符号与 Python 版逐行一致，便于交叉验证（见 test/）。
#include "common/BigSmallYaw/BigSmallYawTrajectoryPlanner.h"

#include <algorithm>
#include <cmath>

namespace bsy {

TrajectoryPlanner::TrajectoryPlanner(double max_velocity, double max_acceleration, double max_jerk)
    : V_(max_velocity), A_(max_acceleration), J_(max_jerk) {}

int TrajectoryPlanner::sign(double x) {
    if (std::fabs(x) < 1e-12) return 0;
    return (x < 0.0) ? -1 : 1;
}

bool TrajectoryPlanner::willHitVelocityLimit(double v, double a) const {
    const double dv = a * std::fabs(a) / (2.0 * J_);
    return std::fabs(v + dv) >= V_;
}

// 计算停止时相对位移 dp、整个刹车过程位移的极值区间 (left, right)、
// 以及本策略下当前时刻的 jerk 方向（Python _brake_distance）
void TrajectoryPlanner::brakeDistance(double v, double a, double& dp, double& dp_left,
                                      double& dp_right, double& jerk_sign) const {
    const int sv = sign(v);
    if (v * a >= 0.0) {
        // ── 速度与加速度同向（加速中 / 静止起步）──
        const double utmost_v = v + a * std::fabs(a) / (2.0 * J_);
        const double utmost_a = -sv * std::sqrt(std::fabs(utmost_v) * J_);
        if (std::fabs(utmost_a) <= A_) {
            const double dt1 = std::fabs(utmost_a - a) / J_;
            const double dp1 = v * dt1 + 0.5 * a * dt1 * dt1
                             + (1.0 / 6.0) * (-sv) * J_ * dt1 * dt1 * dt1;
            const double dt2 = std::fabs(utmost_a) / J_;
            const double dp2 = (1.0 / 6.0) * sv * J_ * dt2 * dt2 * dt2;
            dp = dp1 + dp2;
        } else {
            const double true_utmost_a = -sv * A_;
            const double dt1 = std::fabs(true_utmost_a - a) / J_;
            const double dp1 = v * dt1 + 0.5 * a * dt1 * dt1
                             + (1.0 / 6.0) * (-sv) * J_ * dt1 * dt1 * dt1;
            const double dv1 = a * dt1 + 0.5 * (-sv) * J_ * dt1 * dt1;
            const double dt3 = A_ / J_;
            const double dp3 = (1.0 / 6.0) * sv * J_ * dt3 * dt3 * dt3;
            const double dv3 = 0.5 * (-sv) * J_ * dt3 * dt3;
            const double v2 = v + dv1;
            const double v3 = -dv3;
            const double dt2 = std::fabs(v3 - v2) / A_;
            const double dp2 = v2 * dt2 + 0.5 * true_utmost_a * dt2 * dt2;
            dp = dp1 + dp2 + dp3;
        }
        if (dp > 0.0) {
            dp_left = 0.0;
            dp_right = dp;
        } else {
            dp_left = dp;
            dp_right = 0.0;
        }
        jerk_sign = (double)(-sv);
        return;
    }

    // ── 速度与加速度反向 ──
    const double brake_a_dv = a * std::fabs(a) / (2.0 * J_);
    if (v * (v + brake_a_dv) >= 0.0) {
        const double utmost_a = -sv * std::sqrt(std::fabs(v) * J_ + a * a / 2.0);
        if (std::fabs(utmost_a) <= A_) {
            const double dt1 = std::fabs(utmost_a - a) / J_;
            const double dp1 = v * dt1 + 0.5 * a * dt1 * dt1
                             + (1.0 / 6.0) * (-sv) * J_ * dt1 * dt1 * dt1;
            const double dt2 = std::fabs(utmost_a) / J_;
            const double dp2 = (1.0 / 6.0) * sv * J_ * dt2 * dt2 * dt2;
            dp = dp1 + dp2;
        } else {
            const double true_utmost_a = -sv * A_;
            const double dt1 = std::fabs(true_utmost_a - a) / J_;
            const double dp1 = v * dt1 + 0.5 * a * dt1 * dt1
                             + (1.0 / 6.0) * (-sv) * J_ * dt1 * dt1 * dt1;
            const double dv1 = a * dt1 + 0.5 * (-sv) * J_ * dt1 * dt1;
            const double dt3 = A_ / J_;
            const double dp3 = (1.0 / 6.0) * sv * J_ * dt3 * dt3 * dt3;
            const double dv3 = 0.5 * (-sv) * J_ * dt3 * dt3;
            const double v2 = v + dv1;
            const double v3 = -dv3;
            const double dt2 = std::fabs(v3 - v2) / A_;
            const double dp2 = v2 * dt2 + 0.5 * true_utmost_a * dt2 * dt2;
            dp = dp1 + dp2 + dp3;
        }
        if (dp > 0.0) {
            dp_left = 0.0;
            dp_right = dp;
        } else {
            dp_left = dp;
            dp_right = 0.0;
        }
        jerk_sign = (double)(-sv);
        return;
    }

    // ── 反向且速度即将换向：分段（先按当前加速度反向，再递归一次零加速度刹车；
    //    a = 0 时只会走 v*a >= 0 分支，递归深度最多为 1，与 Python 一致）──
    const double dt1 = std::fabs(a) / J_;
    const double dp1 = v * dt1 + 0.5 * a * dt1 * dt1
                     + (1.0 / 6.0) * sv * J_ * dt1 * dt1 * dt1;
    const double sq = a * a - 2.0 * std::fabs(v) * J_;
    const double utmost_dp1_dt = (std::fabs(a) - std::sqrt(std::max(0.0, sq))) / J_;
    const double utmost_dp1 = v * utmost_dp1_dt + 0.5 * a * utmost_dp1_dt * utmost_dp1_dt
                            + (1.0 / 6.0) * sv * J_ * utmost_dp1_dt * utmost_dp1_dt * utmost_dp1_dt;
    const double dp1_left  = std::min(0.0, std::min(dp1, utmost_dp1));
    const double dp1_right = std::max(0.0, std::max(dp1, utmost_dp1));

    const double v2 = v + 0.5 * a * dt1;
    const double a2 = 0.0;
    double dp2 = 0.0, dp2_left = 0.0, dp2_right = 0.0, jerk2 = 0.0;
    brakeDistance(v2, a2, dp2, dp2_left, dp2_right, jerk2);

    dp = dp1 + dp2;
    dp_left  = std::min(0.0, std::min(dp2_left + dp1, dp1_left));
    dp_right = std::max(0.0, std::max(dp2_right + dp1, dp1_right));
    jerk_sign = (double)sv;
}

TrajectoryPlanner::Step TrajectoryPlanner::step(double target, double p, double v, double a,
                                                double dt) const {
    Step out;
    const double error = target - p;
    double jerk;
    if (willHitVelocityLimit(v, a)) {
        jerk = -(double)sign(v) * J_;
    } else {
        double dp = 0.0, dp_left = 0.0, dp_right = 0.0, jerk_sign = 0.0;
        brakeDistance(v, a, dp, dp_left, dp_right, jerk_sign);
        jerk = (double)sign(error - dp) * J_;
    }
    if (jerk * a > 0.0 && std::fabs(a + jerk * dt) > A_) {
        jerk = 0.0;
    }
    out.acc  = a + jerk * dt;
    out.vel  = v + out.acc * dt;
    out.pos  = p + out.vel * dt;
    out.jerk = jerk;
    return out;
}

TrajectoryPlanner::Step TrajectoryPlanner::stepRefined(double target, double p, double v, double a,
                                                       double dt, int substeps) const {
    if (substeps <= 1) return step(target, p, v, a, dt);
    const double sub_dt = dt / (double)substeps;
    double jerk_sum = 0.0;
    Step s;
    s.pos = p; s.vel = v; s.acc = a;
    for (int i = 0; i < substeps; ++i) {
        s = step(target, s.pos, s.vel, s.acc, sub_dt);
        jerk_sum += s.jerk;
    }
    s.jerk = jerk_sum / (double)substeps;   // 与 Python 一致：jerk 取子步平均
    return s;
}

void TrajectoryPlanner::scan(const std::vector<double>& target, double p0, double v0, double a0,
                             double dt, int substeps,
                             std::vector<double>& pos, std::vector<double>& vel,
                             std::vector<double>& acc, std::vector<double>& jerk) const {
    const size_t n = target.size();
    pos.assign(n, p0);
    vel.assign(n, v0);
    acc.assign(n, a0);
    jerk.assign(n, 0.0);
    double p = p0, v = v0, a = a0;
    for (size_t i = 0; i < n; ++i) {
        const Step s = stepRefined(target[i], p, v, a, dt, substeps);
        p = s.pos; v = s.vel; a = s.acc;
        pos[i] = p; vel[i] = v; acc[i] = a; jerk[i] = s.jerk;
    }
}

void TrajectoryPlanner::scanFrom(const std::vector<double>& target, size_t from_index,
                                 double p0, double v0, double a0, double dt, int substeps,
                                 std::vector<double>& pos, std::vector<double>& vel,
                                 std::vector<double>& acc, std::vector<double>& jerk) const {
    const size_t n = target.size();
    if (from_index >= n) return;
    double p = p0, v = v0, a = a0;
    for (size_t i = from_index; i < n; ++i) {
        const Step s = stepRefined(target[i], p, v, a, dt, substeps);
        p = s.pos; v = s.vel; a = s.acc;
        pos[i] = p; vel[i] = v; acc[i] = a; jerk[i] = s.jerk;
    }
}

} // namespace bsy
