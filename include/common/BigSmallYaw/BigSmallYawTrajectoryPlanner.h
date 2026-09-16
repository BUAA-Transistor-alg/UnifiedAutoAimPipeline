// BigSmallYawTrajectoryPlanner.h — 同时满足最大速度 / 最大加速度 / 最大加加速度
// 限制的一维轨迹规划器（**移植自子模组
// sub_module/TorqueControllerForBigSmallYaw/python/scripts/trajectory_planner.py，
// 逐行对应，含 StepRefinementWrapper**）
//
// 用途：大小 yaw 拆分器用它把“瞄准方位角序列”（移动目标）扫描成平滑的大 yaw 目标轨迹：
// 每个步长把当前位置/速度/加速度朝该步的目标值推进一次 jerk 限幅动作（jerk ∈ {−J, 0, +J}），
// 由“能否在限幅内刹停到目标”决定 jerk 的符号（brake distance 判据），并保证不越过速度上限。
//
// 与 Python 版本的对应关系：
//   TrajectoryPlanner(max_velocity, max_acceleration, max_jerk) ↔ TrajectoryPlanner(V, A, J)
//   step(target, p, v, a, dt)                                   ↔ step(target, p, v, a, dt)
//   StepRefinementWrapper(step_func, n)                         ↔ scan(..., substeps = n)
// Python 版返回 (new_p, new_v, new_a, jerk)，此处返回 Step{pos, vel, acc, jerk}（语义相同）。
#ifndef BSY_TRAJECTORY_PLANNER_H
#define BSY_TRAJECTORY_PLANNER_H

#include <cstddef>
#include <vector>

namespace bsy {

// ---------------------------------------------------------------------------
// 一维 jerk 限幅轨迹规划器（与 Python 版逐行对应）
// ---------------------------------------------------------------------------
class TrajectoryPlanner {
public:
    // 单步结果（对应 Python step() 的 4 元组）
    struct Step {
        double pos = 0.0;
        double vel = 0.0;
        double acc = 0.0;
        double jerk = 0.0;
    };

    TrajectoryPlanner(double max_velocity, double max_acceleration, double max_jerk);

    /// 单步：以 dt 为步长把 (p, v, a) 朝静止目标 target 推进一步
    /// （与 python/scripts/trajectory_planner.py 的 TrajectoryPlanner::step 等价）
    Step step(double target, double p, double v, double a, double dt) const;

    /// 单步（带细化）：把 dt 拆成 substeps 个子步依次调用 step()，
    /// jerk 取各子步平均值（等价于 Python 的 StepRefinementWrapper；substeps <= 1 时退化为 step）
    Step stepRefined(double target, double p, double v, double a, double dt, int substeps) const;

    double maxVelocity() const { return V_; }
    double maxAcceleration() const { return A_; }
    double maxJerk() const { return J_; }

    /// 扫描整条目标序列：从初值 (p0, v0, a0) 出发逐点推进，输出每点的
    /// 位置/速度/加速度/加加速度（长度 = target.size()；dt = 相邻目标点间隔）
    /// @param target  目标序列（**已连续化/unwrap**，见拆分器）
    /// @param pos/vel/acc/jerk  [out] 逐点轨迹状态与 jerk（长度同 target）
    void scan(const std::vector<double>& target, double p0, double v0, double a0, double dt,
              int substeps,
              std::vector<double>& pos, std::vector<double>& vel,
              std::vector<double>& acc, std::vector<double>& jerk) const;

    /// 从 from_index 起覆盖式续规划（拆分器“跳变后重新规划”用）：
    /// 从 (p0, v0, a0) 出发，对 target[from_index..] 逐点推进并**覆盖** pos/vel/acc/jerk
    /// 对应下标之后的部分（长度须与 target 一致）
    void scanFrom(const std::vector<double>& target, size_t from_index,
                  double p0, double v0, double a0, double dt, int substeps,
                  std::vector<double>& pos, std::vector<double>& vel,
                  std::vector<double>& acc, std::vector<double>& jerk) const;

private:
    // 以下三个静态函数与 Python 版 _will_hit_velocity_limit / _brake_distance / _sign 一一对应
    bool willHitVelocityLimit(double v, double a) const;
    // 计算“以当前 (v, a) 在限幅下刹停”的位移 dp、全过程位移极值区间 [left, right]、
    // 以及该策略下的 jerk 方向（+1/-1/0）
    void brakeDistance(double v, double a, double& dp, double& dp_left, double& dp_right,
                       double& jerk_sign) const;
    static int sign(double x);

    double V_ = 0.0;   // 最大速度（rad/s）
    double A_ = 0.0;   // 最大加速度（rad/s²）
    double J_ = 0.0;   // 最大加加速度（rad/s³）
};

} // namespace bsy

#endif // BSY_TRAJECTORY_PLANNER_H
