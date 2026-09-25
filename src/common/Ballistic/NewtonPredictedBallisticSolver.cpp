// NewtonPredictedBallisticSolver.cpp — 联合求解 yaw、pitch 和飞行时间。
// 通过目标预测器构造拦截残差，RK4 同步传播弹道和角度灵敏度，使用带回溯的 Newton
// 修正。同一板可用上一时刻的解及隐函数导数预测初值；状态由调用方局部持有，
// 解算器不缓存跨帧状态，不同板可并行求解。
#include "common/Ballistic/NewtonPredictedBallisticSolver.h"

#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <utility>

#include "common/RobotConfig.h"

namespace {
constexpr double kPi = 3.14159265358979323846;

Rk4BallisticIntegrator::Parameters configuredPhysics() {
    const auto& cfg = RobotConfig::instance().common.gimbal;
    Rk4BallisticIntegrator::Parameters parameters;
    parameters.drag_coefficient =
        Rk4BallisticIntegrator::sphericalDragCoefficient(cfg.bulletDiameter, cfg.bulletMass);
    parameters.step = cfg.integrationStep;
    return parameters;
}

bool positiveFinite(double value) {
    return std::isfinite(value) && value > 0.0;
}
} // namespace

// ============ 构造与输入校验 ============
NewtonPredictedBallisticSolver::NewtonPredictedBallisticSolver(
    std::shared_ptr<GimbalSolver> gimbal)
    : gimbal_(std::move(gimbal)), integrator_(configuredPhysics()),
      distance_tolerance_(gimbal_ ? gimbal_->distanceThreshold() : 0.0) {}

NewtonPredictedBallisticSolver::NewtonPredictedBallisticSolver(
    const Rk4BallisticIntegrator::Parameters& parameters,
    double distance_tolerance, const Options& options)
    : integrator_(parameters), distance_tolerance_(distance_tolerance), options_(options) {}

bool NewtonPredictedBallisticSolver::validInput(
    const Predictor& predictor, double extra_predict_time, const LaunchGeometry& geometry) const {
    return predictor && std::isfinite(extra_predict_time) && extra_predict_time >= 0.0 &&
        geometry.chassis_position.allFinite() && geometry.chassis_rotation.allFinite() &&
        geometry.yaw_position.allFinite() && geometry.pitch_position.allFinite() &&
        geometry.muzzle_offset.allFinite() && positiveFinite(geometry.bullet_velocity) &&
        std::isfinite(geometry.current_yaw) && std::isfinite(geometry.current_pitch) &&
        std::isfinite(geometry.pitch_min) && std::isfinite(geometry.pitch_max) &&
        geometry.pitch_min <= geometry.pitch_max && std::isfinite(geometry.stop_z) &&
        positiveFinite(distance_tolerance_) && options_.max_iterations > 0 &&
        options_.max_backtracking > 0 && positiveFinite(options_.angle_step_limit) &&
        positiveFinite(options_.time_step_limit) && positiveFinite(options_.min_flight_time) &&
        positiveFinite(options_.max_flight_time) &&
        options_.min_flight_time < options_.max_flight_time &&
        positiveFinite(options_.target_difference_step) && positiveFinite(options_.solve_tolerance);
}

// ============ 目标采样与速度差分 ============
bool NewtonPredictedBallisticSolver::sampleTargetPosition(
    const Predictor& predictor, int target_index, double time, Eigen::Vector3d& position) const {
    // predictor 与索引已由入口保证；各时刻的返回值仍可能缺板或包含无效观测。
    if (!std::isfinite(time) || time < 0.0) return false;
    const auto prediction = predictor(time);
    if (static_cast<size_t>(target_index) >= prediction.second.size()) return false;
    const cv::Point3f& point = prediction.second[static_cast<size_t>(target_index)];
    position = Eigen::Vector3d(point.x, point.y, point.z);
    return position.allFinite();
}

bool NewtonPredictedBallisticSolver::sampleTargetState(
    const Predictor& predictor, int target_index, double time,
    Eigen::Vector3d& position, Eigen::Vector3d& velocity) const {
    if (!sampleTargetPosition(predictor, target_index, time, position)) return false;
    const double h = options_.target_difference_step;
    Eigen::Vector3d before, after;
    if (!sampleTargetPosition(predictor, target_index, time + h, after)) return false;
    if (time >= h) {
        if (!sampleTargetPosition(predictor, target_index, time - h, before)) return false;
        velocity = (after - before) / (2.0 * h);
    } else {
        // 靠近时间零点时使用前向差分，避免调用只定义在未来的 predictor 的负时间。
        velocity = (after - position) / h;
    }
    return velocity.allFinite();
}

// ============ 初值（几何朝向 + 重力补偿） ============
bool NewtonPredictedBallisticSolver::makeInitialGuess(
    const Predictor& predictor, int target_index, double extra_predict_time,
    const LaunchGeometry& geometry, Eigen::Vector3d& q) const {
    q = Eigen::Vector3d(geometry.current_yaw,
                       std::clamp(geometry.current_pitch, geometry.pitch_min, geometry.pitch_max),
                       options_.min_flight_time);
    Eigen::Vector3d target;
    if (!sampleTargetPosition(predictor, target_index, extra_predict_time, target)) return false;
    auto launch = GimbalSolver::evaluateLaunchState(geometry, q[0], q[1]);
    q[2] = std::clamp((target - launch.position).norm() / geometry.bullet_velocity,
                      options_.min_flight_time, options_.max_flight_time);

    // 少量无阻力初值修正只用于靠近低仰角解，正式命中验证始终用完整二次阻力模型。
    // 每轮重新计算偏心枪口位置，不调用旧 pitch 粗搜索，也不复用已飞行弹丸的速度。
    for (int i = 0; i < 3; ++i) {
        if (!sampleTargetPosition(predictor, target_index, extra_predict_time + q[2], target))
            return false;
        launch = GimbalSolver::evaluateLaunchState(geometry, q[0], q[1]);
        const Eigen::Vector3d displacement = target - launch.position;
        const Eigen::Vector3d direction = geometry.chassis_rotation.transpose() *
            (displacement - 0.5 * integrator_.acceleration(Eigen::Vector3d::Zero()) * q[2] * q[2]);
        if (!direction.allFinite() || direction.norm() < 1e-9) return false;
        const double yaw = std::atan2(-direction.x(), direction.y());
        // 内部保持当前 yaw 附近的连续分支；最终打包时才归一化。
        q[0] = geometry.current_yaw + std::remainder(yaw - geometry.current_yaw, 2.0 * kPi);
        q[1] = std::clamp(std::atan2(direction.z(), std::hypot(direction.x(), direction.y())),
                          geometry.pitch_min, geometry.pitch_max);
        q[2] = std::clamp(displacement.norm() / geometry.bullet_velocity,
                          options_.min_flight_time, options_.max_flight_time);
    }
    return q.allFinite();
}

// ============ 命中残差与一阶 Jacobian ============
NewtonPredictedBallisticSolver::Evaluation
NewtonPredictedBallisticSolver::evaluateResidualAndJacobian(
    const Predictor& predictor, int target_index, double extra_predict_time,
    const LaunchGeometry& geometry, const Eigen::Vector3d& q) const {
    if (!validInput(predictor, extra_predict_time, geometry) || target_index < 0) return {};
    return evaluateCandidate(predictor, target_index, extra_predict_time, geometry, q);
}

NewtonPredictedBallisticSolver::Evaluation NewtonPredictedBallisticSolver::evaluateCandidate(
    const Predictor& predictor, int target_index, double extra_predict_time,
    const LaunchGeometry& geometry, const Eigen::Vector3d& q) const {
    Evaluation evaluation;
    // 固定输入已在公开入口校验；每轮只检查变化的角度、时间及预测结果。
    if (!q.allFinite() ||
        q[1] < geometry.pitch_min || q[1] > geometry.pitch_max ||
        q[2] < options_.min_flight_time || q[2] > options_.max_flight_time) return evaluation;

    if (!sampleTargetState(predictor, target_index, extra_predict_time + q[2],
                           evaluation.target_position, evaluation.target_velocity)) return evaluation;

    const auto launch = GimbalSolver::evaluateLaunchState(geometry, q[0], q[1]);
    Rk4BallisticIntegrator::State initial;
    initial.position = launch.position;
    initial.velocity = launch.velocity;
    initial.position_jacobian = launch.position_jacobian;
    initial.velocity_jacobian = launch.velocity_jacobian;
    Rk4BallisticIntegrator::State endpoint;
    if (!integrator_.integrateTo(initial, q[2], endpoint)) return evaluation;

    evaluation.residual = endpoint.position - evaluation.target_position;
    evaluation.jacobian.leftCols<2>() = endpoint.position_jacobian;
    // τ 同时改变弹丸端点与目标采样时刻，第三列必须是两者的相对速度。
    evaluation.jacobian.col(2) = endpoint.velocity - evaluation.target_velocity;
    evaluation.valid = evaluation.residual.allFinite() && evaluation.jacobian.allFinite();
    return evaluation;
}

// ============ 线性方程求解与步长限制 ============
bool NewtonPredictedBallisticSolver::solveLinearSystem(
    const Eigen::Matrix3d& jacobian, const Eigen::Vector3d& rhs,
    Eigen::Vector3d& solution) const {
    // 用各变量允许步长做列缩放，减少 rad / s 量纲和幅值差异对秩判断的影响。
    const Eigen::Vector3d scale(options_.angle_step_limit, options_.angle_step_limit,
                                options_.time_step_limit);
    const Eigen::Matrix3d scaled_jacobian = jacobian * scale.asDiagonal();
    Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr(scaled_jacobian);
    qr.setThreshold(1e-10);
    if (qr.rank() < 3) return false;
    solution = scale.asDiagonal() * qr.solve(rhs);
    return solution.allFinite();
}

bool NewtonPredictedBallisticSolver::computeNewtonStep(
    const Evaluation& evaluation, Eigen::Vector3d& delta) const {
    if (!solveLinearSystem(evaluation.jacobian, -evaluation.residual, delta)) return false;
    // 所有分量等比例缩小，保持 Newton 方向；不是把逆矩阵显式算出来。
    const double ratio = std::max({1.0, std::fabs(delta[0]) / options_.angle_step_limit,
                                  std::fabs(delta[1]) / options_.angle_step_limit,
                                  std::fabs(delta[2]) / options_.time_step_limit});
    delta /= ratio;
    return true;
}

bool NewtonPredictedBallisticSolver::applyDampedStep(
    const Predictor& predictor, int target_index, double extra_predict_time,
    const LaunchGeometry& geometry, const Eigen::Vector3d& delta,
    Eigen::Vector3d& q, Evaluation& evaluation) const {
    const double old_error = evaluation.residual.norm();
    double damping = 1.0;
    for (int i = 0; i < options_.max_backtracking; ++i, damping *= 0.5) {
        Eigen::Vector3d candidate = q + damping * delta;
        candidate[1] = std::clamp(candidate[1], geometry.pitch_min, geometry.pitch_max);
        candidate[2] = std::clamp(candidate[2], options_.min_flight_time, options_.max_flight_time);
        if ((candidate - q).norm() < 1e-12) continue;
        Evaluation trial = evaluateCandidate(
            predictor, target_index, extra_predict_time, geometry, candidate);
        if (trial.valid && trial.residual.norm() < old_error) {
            q = candidate;
            evaluation = trial;
            return true;
        }
    }
    return false;
}

// ============ 结果打包（保持旧角度、时间及索引语义） ============
NewtonPredictedBallisticSolver::Result NewtonPredictedBallisticSolver::packResult(
    int target_index, double extra_predict_time, const LaunchGeometry& geometry,
    const Eigen::Vector3d& q, const Evaluation& evaluation) const {
    Result result{};
    result.target_index = target_index;
    // 仅打包已通过候选校验的最终值；数值与角度 / 时间范围不再重复检查。

    result.predicted_point = cv::Vec3f(static_cast<float>(evaluation.target_position.x()),
                                       static_cast<float>(evaluation.target_position.y()),
                                       static_cast<float>(evaluation.target_position.z()));
    result.predict_time = extra_predict_time + q[2];
    result.gimbal.yaw = static_cast<float>(std::remainder(q[0], 2.0 * kPi));
    result.gimbal.pitch = static_cast<float>(q[1]);
    result.gimbal.flight_time = q[2];
    // 新算法此字段为同一命中时刻的三维残差，而非轨迹到静态目标的最近距离。
    result.gimbal.distance = evaluation.residual.norm();
    const double endpoint_z = evaluation.target_position.z() + evaluation.residual.z();
    result.success = result.gimbal.distance <= distance_tolerance_ &&
                     endpoint_z >= geometry.stop_z;
    result.gimbal.success = result.success;
    return result;
}

// ============ 单目标阻尼 Newton 迭代 ============
NewtonPredictedBallisticSolver::TargetSolution NewtonPredictedBallisticSolver::solveTarget(
    const Predictor& predictor, int target_index, double extra_predict_time,
    const LaunchGeometry& geometry, const WarmStart* warm_start) const {
    if (!validInput(predictor, extra_predict_time, geometry) || target_index < 0) {
        TargetSolution invalid;
        invalid.result.target_index = target_index;
        invalid.result.gimbal.distance = std::numeric_limits<double>::infinity();
        return invalid;
    }
    // ============ 上次解的一阶时间预测 + Newton 校正 ============
    // 时间增量只取发射时刻（extra_predict_time）之差，不能把飞行时间再加一次。
    if (warm_start && warm_start->target_index == target_index) {
        const double dt = extra_predict_time - warm_start->extra_predict_time;
        if (std::isfinite(dt) && dt >= 0.0) {
            Eigen::Vector3d q = warm_start->q + dt * warm_start->time_derivative;
            // 沿上一解的连续 yaw 分支外推，仅在 packResult 中归一化输出角。
            q[1] = std::clamp(q[1], geometry.pitch_min, geometry.pitch_max);
            q[2] = std::clamp(q[2], options_.min_flight_time, options_.max_flight_time);
            TargetSolution solution = solveTargetImpl(
                predictor, target_index, extra_predict_time, geometry, &q);
            if (solution.result.success &&
                solution.residual_norm <= std::min(distance_tolerance_, options_.solve_tolerance)) {
                return solution;
            }
        }
    }
    // 预测初值无效或未充分收敛时，用已有几何初值重试；不把外推值直接交给后续序列。
    return solveTargetImpl(predictor, target_index, extra_predict_time, geometry);
}

NewtonPredictedBallisticSolver::TargetSolution NewtonPredictedBallisticSolver::solveTargetImpl(
    const Predictor& predictor, int target_index, double extra_predict_time,
    const LaunchGeometry& geometry, const Eigen::Vector3d* initial_guess) const {
    TargetSolution solution;
    solution.result.target_index = target_index;
    solution.result.gimbal.distance = std::numeric_limits<double>::infinity();

    Eigen::Vector3d q;
    if (initial_guess) {
        q = *initial_guess;
    } else if (!makeInitialGuess(predictor, target_index, extra_predict_time, geometry, q)) {
        solution.status = Status::INVALID_TARGET;
        return solution;
    }
    Evaluation evaluation = evaluateCandidate(
        predictor, target_index, extra_predict_time, geometry, q);
    if (!evaluation.valid) {
        solution.status = Status::INTEGRATION_FAILED;
        return solution;
    }

    const double desired_error = std::min(distance_tolerance_, options_.solve_tolerance);
    solution.status = Status::ITERATION_LIMIT;
    for (int iter = 0; iter < options_.max_iterations; ++iter) {
        if (evaluation.residual.norm() <= desired_error) {
            solution.status = Status::CONVERGED;
            break;
        }
        Eigen::Vector3d delta;
        if (!computeNewtonStep(evaluation, delta)) {
            solution.status = Status::SINGULAR_JACOBIAN;
            break;
        }
        ++solution.iterations;
        if (!applyDampedStep(predictor, target_index, extra_predict_time, geometry,
                             delta, q, evaluation)) {
            solution.status = Status::NO_DESCENT;
            break;
        }
    }
    // 每一步都要求残差下降，当前值就是本次有效最佳值。即使迭代耗尽，也必须通过
    // 显式的命中残差与范围检查才能使用；绝不因“增量很小”而把失败解标记为成功。
    solution.result = packResult(target_index, extra_predict_time, geometry, q, evaluation);
    solution.residual_norm = evaluation.residual.norm();
    if (solution.result.success) {
        solution.status = Status::CONVERGED;
        // ============ 隐函数求导（固定发射几何） ============
        // F(q, t) = p_bullet(q) - p_target(t + tau) = 0，故 J * dq/dt = v_target。
        // 此处求的是变化率，不使用 Newton 单步限幅；下一时刻仍按完整模型校正。
        solution.warm_start.target_index = target_index;
        solution.warm_start.extra_predict_time = extra_predict_time;
        solution.warm_start.q = q;
        if (!solveLinearSystem(evaluation.jacobian, evaluation.target_velocity,
                               solution.warm_start.time_derivative)) {
            // 当前命中有效但局部导数不可用时，退化为直接复用上次解（零阶热启动）。
            solution.warm_start.time_derivative.setZero();
        }
    } else if (solution.residual_norm <= distance_tolerance_) {
        // 残差合格仍可能违反截止高度；不能保留 CONVERGED 而同时返回失败。
        solution.status = Status::CONSTRAINT_VIOLATION;
    }
    return solution;
}

// ============ 全候选入口（不在解算器内部选板） ============
std::vector<NewtonPredictedBallisticSolver::Result> NewtonPredictedBallisticSolver::solve(
    const Predictor& predictor, double extra_predict_time, float yawBig,
    const std::vector<int>& masked_indices) const {
    if (!gimbal_) return {};
    return solveWithGeometry(predictor, extra_predict_time, gimbal_->captureLaunchGeometry(yawBig),
                             masked_indices);
}

std::vector<NewtonPredictedBallisticSolver::Result>
NewtonPredictedBallisticSolver::solveWithGeometry(
    const Predictor& predictor, double extra_predict_time, const LaunchGeometry& geometry,
    const std::vector<int>& masked_indices) const {
    if (!validInput(predictor, extra_predict_time, geometry)) return {};
    const auto now = predictor(0.0);
    std::vector<Result> results;
    results.reserve(now.second.size());
    for (size_t i = 0; i < now.second.size(); ++i) {
        // 屏蔽点：不做 Newton 迭代（也不采样 predictor 各时刻），占位保持下标对齐
        if (PredictedBallisticSolver::isMaskedIndex(masked_indices, static_cast<int>(i))) {
            Result placeholder;
            placeholder.target_index = static_cast<int>(i);
            placeholder.masked = true;
            results.push_back(placeholder);
            continue;
        }
        results.push_back(solveTargetImpl(predictor, static_cast<int>(i), extra_predict_time, geometry).result);
    }
    return results;
}