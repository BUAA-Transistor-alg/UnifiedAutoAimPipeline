// NewtonPredictedBallisticSolver.h — 中低速 Armor 的联合拦截解算器。
// 将 yaw、pitch、飞行时间作为未知量，利用 RK4 灵敏度构造 Jacobian 并做阻尼 Newton
// 迭代，支持同一预测器内相邻时刻的一阶热启动；输出兼容旧弹道结果。
// 只解算候选点（屏蔽索引跳过并占位），不选板、不重复
// 补偿延迟、不修改预测器。
#ifndef NEWTON_PREDICTED_BALLISTIC_SOLVER_H
#define NEWTON_PREDICTED_BALLISTIC_SOLVER_H

#include <Eigen/Core>
#include <limits>
#include <memory>
#include <vector>

#include "common/Ballistic/PredictedBallisticSolver.h"
#include "common/Ballistic/Rk4BallisticIntegrator.h"

class NewtonPredictedBallisticSolver {
public:
    using Predictor = PredictedBallisticSolver::Predictor;
    using Result = PredictedBallisticSolver::Result;
    using LaunchGeometry = GimbalSolver::LaunchGeometry;

    // ============ Newton 数值参数（不增加算法切换开关） ============
    struct Options {
        int max_iterations = 20;
        int max_backtracking = 10;
        double angle_step_limit = 0.2;        // 单轮 yaw / pitch 最大增量，rad
        double time_step_limit = 0.1;         // 单轮飞行时间最大增量，s
        double min_flight_time = 1e-4;
        double max_flight_time = 3.0;
        double target_difference_step = 1e-3; // predictor 返回 float，不宜用过小差分步长
        double solve_tolerance = 1e-4;        // 期望残差，m；最终仍检查输入命中容差
    };

    struct Evaluation {
        bool valid = false;
        Eigen::Vector3d residual = Eigen::Vector3d::Zero();
        Eigen::Vector3d target_position = Eigen::Vector3d::Zero();
        Eigen::Vector3d target_velocity = Eigen::Vector3d::Zero();
        Eigen::Matrix3d jacobian = Eigen::Matrix3d::Zero();
    };

    enum class Status {
        CONVERGED, INVALID_INPUT, INVALID_TARGET, INTEGRATION_FAILED,
        SINGULAR_JACOBIAN, NO_DESCENT, ITERATION_LIMIT, CONSTRAINT_VIOLATION
    };

    // ============ 同一装甲板的跨时间点初值 ============
    // 由调用方按预测器快照管理生命周期，不跨帧缓存；target_index < 0 表示不可复用。
    // q 保留 double 与连续 yaw，time_derivative 是固定发射几何下 dq / d(extra_time)。
    struct WarmStart {
        int target_index = -1;
        double extra_predict_time = 0.0;
        Eigen::Vector3d q = Eigen::Vector3d::Zero();
        Eigen::Vector3d time_derivative = Eigen::Vector3d::Zero();
    };

    struct TargetSolution {
        Result result;
        Status status = Status::INVALID_INPUT;
        int iterations = 0;
        double residual_norm = std::numeric_limits<double>::infinity();
        WarmStart warm_start;
    };

    // 工程入口：复用已同步的 GimbalSolver 几何、弹速、命中容差及机器弹丸参数。
    explicit NewtonPredictedBallisticSolver(std::shared_ptr<GimbalSolver> gimbal);
    // 显式物理参数入口：供独立验证使用，不依赖机器配置或实时坐标树。
    NewtonPredictedBallisticSolver(const Rk4BallisticIntegrator::Parameters& parameters,
                                  double distance_tolerance, const Options& options);

    // extra_predict_time 已含快照年龄、额外延迟和序列点偏移；本类只再加飞行时间。
    // yawBig 固定为此序列点的大 yaw 关节角，Newton 调整飞行时间时不改变枪口快照。
    // masked_indices（可选）：屏蔽的瞄准点索引（预测函数返回列表下标），其中的点
    // **不做 Newton 解算**，仅返回占位符（Result::masked = true，见
    // PredictedBallisticSolver::Result），保持返回向量与目标点列表下标对齐。
    std::vector<Result> solve(const Predictor& predictor, double extra_predict_time,
                             float yawBig = std::numeric_limits<float>::quiet_NaN(),
                             const std::vector<int>& masked_indices = {}) const;
    std::vector<Result> solveWithGeometry(const Predictor& predictor, double extra_predict_time,
                                         const LaunchGeometry& geometry,
                                         const std::vector<int>& masked_indices = {}) const;

    // 单目标详细入口：可复用同一快照、同一板的上次成功解，无共享可变状态。
    // 热启动未达到迭代目标时重试原几何初值；最终命中 / 范围校验不变。
    TargetSolution solveTarget(const Predictor& predictor, int target_index,
                               double extra_predict_time, const LaunchGeometry& geometry,
                               const WarmStart* warm_start = nullptr) const;

    // q = [相对底盘的总 yaw, pitch, 飞行时间]；公开只读接口便于差分验证 Jacobian。
    Evaluation evaluateResidualAndJacobian(const Predictor& predictor, int target_index,
                                           double extra_predict_time,
                                           const LaunchGeometry& geometry,
                                           const Eigen::Vector3d& q) const;

private:
    // 内部入口复用外层已校验的 predictor / 几何 / Options，避免每个目标、每轮重复检查。
    TargetSolution solveTargetImpl(const Predictor& predictor, int target_index,
                                   double extra_predict_time, const LaunchGeometry& geometry,
                                   const Eigen::Vector3d* initial_guess = nullptr) const;
    Evaluation evaluateCandidate(const Predictor& predictor, int target_index,
                                 double extra_predict_time, const LaunchGeometry& geometry,
                                 const Eigen::Vector3d& q) const;
    bool validInput(const Predictor& predictor, double extra_predict_time,
                    const LaunchGeometry& geometry) const;
    bool sampleTargetPosition(const Predictor& predictor, int target_index, double time,
                              Eigen::Vector3d& position) const;
    bool sampleTargetState(const Predictor& predictor, int target_index, double time,
                           Eigen::Vector3d& position, Eigen::Vector3d& velocity) const;
    bool makeInitialGuess(const Predictor& predictor, int target_index, double extra_predict_time,
                          const LaunchGeometry& geometry, Eigen::Vector3d& q) const;
    bool solveLinearSystem(const Eigen::Matrix3d& jacobian, const Eigen::Vector3d& rhs,
                           Eigen::Vector3d& solution) const;
    bool computeNewtonStep(const Evaluation& evaluation, Eigen::Vector3d& delta) const;
    bool applyDampedStep(const Predictor& predictor, int target_index, double extra_predict_time,
                         const LaunchGeometry& geometry, const Eigen::Vector3d& delta,
                         Eigen::Vector3d& q, Evaluation& evaluation) const;
    Result packResult(int target_index, double extra_predict_time, const LaunchGeometry& geometry,
                      const Eigen::Vector3d& q, const Evaluation& evaluation) const;

    std::shared_ptr<GimbalSolver> gimbal_;
    Rk4BallisticIntegrator integrator_;
    double distance_tolerance_;
    Options options_;
};

#endif // NEWTON_PREDICTED_BALLISTIC_SOLVER_H