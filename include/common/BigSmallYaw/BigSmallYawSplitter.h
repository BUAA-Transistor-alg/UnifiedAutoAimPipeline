// BigSmallYawSplitter.h — 大小 yaw 拆分器（新构型专用适配器）
//
// 输入：流水线解算出的**瞄准世界方位角序列**（= SequencePredictor 每个返回点的 item.yaw，
//       该序列就是小 yaw 输出的目标世界方位角 ψ_small*）、序列间隔 dt（= dt_control）、
//       当帧实测小 yaw 输出方位角（用于把整段序列解卷绕到与实机同一圈）。
// 输出：给 tcbs::RobotController 序列 set() 的两条世界方位角序列
//       （ψ_big* 大 yaw 参考、ψ_small* 小 yaw 参考），以及规划轨迹的完整状态（诊断/下一轮初值）。
//
// 算法（与用户确认的规则一致）：
//   1) a[] = 瞄准方位角序列（**连续化/unwrap**，首元素对齐到当帧实测 ψ_small 所在圈）；
//   2) 用 TrajectoryPlanner（jerk 限幅，V/A/J 来自
//      common.big_small_yaw.big_small.splitter.planner_*）
//      以 a[] 为**移动目标**扫描，得到平滑的大 yaw 目标轨迹 b[]（同时记录 v/a/j）。
//      初值 (p0,v0,a0)：
//        - 有上一轮计划：按“距上次输出的实际经过时间”（帧时间戳之差，超过覆盖区间用
//          最后一个点保持）在上一轮计划上取插值，位置再做 ±2π 修正到与 a[0] 最近；
//        - 没有：位置 = a[0]，速度/加速度 = 0。
//   3) 从第一个点向后遍历：小 yaw 需要偏离回中目标的角度
//        θ_small[i] = a[i] − b[i]（回中目标 = config small_center_angle，本工程取 0）
//      超过小 yaw **软限位**（与 MPC 同一公式/同一 ratio）时，把 b[i] 增减一个绝对值最小
//      的量，使其恰好回到边界；速度/加速度保持不变，并从该点起用 TrajectoryPlanner
//      重新规划后续轨迹（后续 v/a/j 一并覆盖）。重复直到每个点都不越界。
//      —— 这一步就是“小 yaw 到限位也指不到目标时让大 yaw 无限幅快速运动（可阶跃，交 MPC
//         追赶）”，且**每次修正都从最早越界点触发一次**，所以无限幅动作尽可能少。
//   4) 输出 ψ_big* = b[]、ψ_small* = a[]。
#ifndef BSY_BIG_SMALL_YAW_SPLITTER_H
#define BSY_BIG_SMALL_YAW_SPLITTER_H

#include <chrono>
#include <vector>

#include "common/BigSmallYaw/BigSmallYawTrajectoryPlanner.h"

namespace bsy {

// 拆分器参数（构造时从 common.big_small_yaw.big_small 分支读取）
struct BigSmallYawSplitterConfig {
    double smallMinAngle = 0.0;      // 小 yaw 行程下界（rad）
    double smallMaxAngle = 0.0;      // 小 yaw 行程上界（rad）
    double smallCenterAngle = 0.0;   // 小 yaw 回中目标关节角（rad，0 = 关节零位）
    double softLimitRatio = 0.75;    // 软限位比例（与 MPC small_limit_soft_ratio 同一值）
    double plannerMaxVelocity = 0.0;     // rad/s
    double plannerMaxAcceleration = 0.0; // rad/s²
    double plannerMaxJerk = 0.0;         // rad/s³
    int    plannerSubsteps = 1;          // 单步细化倍数
};

class BigSmallYawSplitter {
public:
    struct Output {
        std::vector<double> big_azimuth;    // ψ_big*（世界方位角序列，交 MPC）
        std::vector<double> small_azimuth;  // ψ_small*（= 瞄准序列，交 MPC）
        std::vector<double> theta_small;    // 诊断：小 yaw 关节角 = small − big
        std::vector<double> pos, vel, acc, jerk;   // 大 yaw 规划轨迹状态（长度同序列）
        int    jump_count = 0;              // 本帧被修正的点数（越界点个数）
        int    unlimited_episodes = 0;      // 本帧“无限幅运动”段数（**连续被修正的点算一段**；
                                            // 该指标才是“无限幅运动尽可能少”的度量：
                                            // 目标持续快于大 yaw 平滑能力时是一段连续修正，
                                            // 而不是多次独立动作）
        bool   over_limit = false;          // 本帧是否发生过越界修正
        double soft_min = 0.0, soft_max = 0.0;   // 本次使用的小 yaw 软限位边界（rad）
        double theta_small_max_abs = 0.0;   // 本帧 |θ_small| 最大值（诊断）
    };

    explicit BigSmallYawSplitter(const BigSmallYawSplitterConfig& cfg);

    /// @param aim_azimuth   瞄准世界方位角序列（弧度；可含跳变/已 wrap，内部会解卷绕）
    /// @param dt            序列间隔（秒，= dt_control）
    /// @param psi_small_now 当帧实测小 yaw 输出世界方位角（用于选圈；无实测传 0）
    /// @param frame_ts      该帧时间戳（**shared_frame_timestamp**，不是 now：
    ///                      用于按实际经过时间在上一轮计划上取初值）
    Output split(const std::vector<double>& aim_azimuth, double dt,
                 double psi_small_now,
                 const std::chrono::steady_clock::time_point& frame_ts);

    /// 清空跨帧状态（上一轮计划）：切换目标/流水线、重新进入自瞄时调用
    void reset();

    /// 小 yaw 软限位边界（rad）：soft = 限位向内收 (1−ratio)·行程
    double softMin() const { return soft_min_; }
    double softMax() const { return soft_max_; }

private:
    BigSmallYawSplitterConfig cfg_;
    TrajectoryPlanner        planner_;
    double soft_min_ = 0.0;
    double soft_max_ = 0.0;

    // 上一轮计划（初值来源；速度/加速度用于保证跨帧连续）
    bool has_prev_ = false;
    std::chrono::steady_clock::time_point prev_ts_{};
    std::vector<double> prev_pos_, prev_vel_, prev_acc_;
};

} // namespace bsy

#endif // BSY_BIG_SMALL_YAW_SPLITTER_H
