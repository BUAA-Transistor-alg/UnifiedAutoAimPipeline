// 本文件实现三维重力、二次空气阻力及其角度灵敏度的 RK4 联合积分
// 积分器计算给定初始条件的弹道
#include "common/Ballistic/Rk4BallisticIntegrator.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

// ============ 联合状态辅助运算 ============
bool isFinite(const Rk4BallisticIntegrator::State& state) {
    return state.position.allFinite() && state.velocity.allFinite()
        && state.position_jacobian.allFinite() && state.velocity_jacobian.allFinite();
}

Rk4BallisticIntegrator::State addScaled(
    const Rk4BallisticIntegrator::State& state,
    const Rk4BallisticIntegrator::State& derivative, double scale) {
    Rk4BallisticIntegrator::State result;
    result.position = state.position + scale * derivative.position;
    result.velocity = state.velocity + scale * derivative.velocity;
    result.position_jacobian = state.position_jacobian + scale * derivative.position_jacobian;
    result.velocity_jacobian = state.velocity_jacobian + scale * derivative.velocity_jacobian;
    return result;
}

// 公开动力学接口与内部积分共用公式，调用方提供已计算的速度模长。
Eigen::Matrix3d dragJacobian(double coefficient, const Eigen::Vector3d& velocity, double speed) {
    if (speed == 0.0 || coefficient == 0.0) return Eigen::Matrix3d::Zero();
    // A(v) = -k * (|v| I + v v^T / |v|)，零速度单独处理。
    const Eigen::Vector3d direction = velocity / speed;
    return -coefficient * speed
        * (Eigen::Matrix3d::Identity() + direction * direction.transpose());
}

}  // namespace

// ============ 构造函数与阻力参数 ============
Rk4BallisticIntegrator::Rk4BallisticIntegrator(const Parameters& parameters)
    : parameters_(parameters) {}

double Rk4BallisticIntegrator::sphericalDragCoefficient(double diameter, double mass) {
    if (!std::isfinite(diameter) || diameter <= 0.0 || !std::isfinite(mass) || mass <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double rho_air = 1.225;   // kg/m^3
    const double Cd = 0.47;         // 球体阻力系数
    const double radius = diameter / 2.0;
    const double area = std::acos(-1.0) * radius * radius;
    const double coefficient = 0.5 * rho_air * Cd * area / mass;
    return std::isfinite(coefficient) ? coefficient : std::numeric_limits<double>::quiet_NaN();
}

// ============ 加速度计算（三维世界系） ============
Eigen::Vector3d Rk4BallisticIntegrator::acceleration(const Eigen::Vector3d& velocity) const {
    if (!velocity.allFinite() || !parameters_.gravity.allFinite()
        || !std::isfinite(parameters_.drag_coefficient) || parameters_.drag_coefficient < 0.0) {
        return Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    }
    if (parameters_.drag_coefficient == 0.0) return parameters_.gravity;
    const double speed = velocity.stableNorm();
    return parameters_.gravity - parameters_.drag_coefficient * speed * velocity;
}

// ============ 加速度对速度的一阶导数 ============
Eigen::Matrix3d Rk4BallisticIntegrator::accelerationJacobian(const Eigen::Vector3d& velocity) const {
    if (!velocity.allFinite() || !std::isfinite(parameters_.drag_coefficient)
        || parameters_.drag_coefficient < 0.0) {
        return Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
    }
    const double speed = velocity.stableNorm();
    if (speed == 0.0 || parameters_.drag_coefficient == 0.0) return Eigen::Matrix3d::Zero();
    if (!std::isfinite(speed)) {
        return Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
    }
    return dragJacobian(parameters_.drag_coefficient, velocity, speed);
}

// ============ 位置、速度与角度灵敏度的联合微分方程 ============
Rk4BallisticIntegrator::State Rk4BallisticIntegrator::stateDerivative(const State& state) const {
    // 物理参数已在 integrateTo 入口校验，每个 RK4 阶段共用一次速度模长计算。
    const double speed = state.velocity.stableNorm();
    State derivative;
    derivative.position = state.velocity;
    derivative.velocity = parameters_.gravity;
    if (parameters_.drag_coefficient != 0.0) {
        derivative.velocity -= parameters_.drag_coefficient * speed * state.velocity;
    }
    derivative.position_jacobian = state.velocity_jacobian;
    derivative.velocity_jacobian = dragJacobian(parameters_.drag_coefficient, state.velocity, speed)
        * state.velocity_jacobian;
    return derivative;
}

// ============ RK4 单步积分（共 18 个状态分量） ============
bool Rk4BallisticIntegrator::stepRK4(State& state, double step) const {
    const State k1 = stateDerivative(state);
    if (!isFinite(k1)) return false;
    const State middle1 = addScaled(state, k1, step * 0.5);
    const State k2 = stateDerivative(middle1);
    if (!isFinite(k2)) return false;
    const State middle2 = addScaled(state, k2, step * 0.5);
    const State k3 = stateDerivative(middle2);
    if (!isFinite(k3)) return false;
    const State end = addScaled(state, k3, step);
    const State k4 = stateDerivative(end);
    if (!isFinite(k4)) return false;

    const double weight = step / 6.0;
    State next;
    next.position = state.position + weight * (k1.position + 2.0 * k2.position + 2.0 * k3.position + k4.position);
    next.velocity = state.velocity + weight * (k1.velocity + 2.0 * k2.velocity + 2.0 * k3.velocity + k4.velocity);
    next.position_jacobian = state.position_jacobian + weight
        * (k1.position_jacobian + 2.0 * k2.position_jacobian + 2.0 * k3.position_jacobian + k4.position_jacobian);
    next.velocity_jacobian = state.velocity_jacobian + weight
        * (k1.velocity_jacobian + 2.0 * k2.velocity_jacobian + 2.0 * k3.velocity_jacobian + k4.velocity_jacobian);
    if (!isFinite(next)) return false;
    state = next;
    return true;
}

// ============ 积分到候选飞行时间 ============
bool Rk4BallisticIntegrator::integrateTo(const State& initial, double flight_time, State& endpoint) const {
    if (!isFinite(initial) || !std::isfinite(flight_time) || flight_time < 0.0
        || !std::isfinite(parameters_.step) || parameters_.step <= 0.0
        || !std::isfinite(parameters_.drag_coefficient) || parameters_.drag_coefficient < 0.0
        || !parameters_.gravity.allFinite() || parameters_.max_steps <= 0) {
        return false;
    }
    if (flight_time == 0.0) {
        endpoint = initial;
        return true;
    }

    if (flight_time > parameters_.step * static_cast<double>(parameters_.max_steps)) return false;
    State state = initial;
    for (int i = 0; i < parameters_.max_steps; ++i) {
        // 从步号计算时间，避免反复累加步长产生漂移；末步精确落在 flight_time。
        const double start_time = static_cast<double>(i) * parameters_.step;
        const double end_time = std::min(static_cast<double>(i + 1) * parameters_.step, flight_time);
        const double step = end_time - start_time;
        if (!std::isfinite(step) || step <= 0.0 || !stepRK4(state, step)) return false;
        if (end_time == flight_time) {
            endpoint = state;
            return true;
        }
    }
    return false;
}
