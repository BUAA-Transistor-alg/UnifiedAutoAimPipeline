// 本文件定义三维弹道与角度灵敏度积分器，为联合 Newton 解算提供指定飞行时间的弹丸状态。
// 位置和速度均在世界系下表达，灵敏度两列依次对应 yaw、pitch，角度单位为弧度。
#ifndef RK4BALLISTICINTEGRATOR_H
#define RK4BALLISTICINTEGRATOR_H

#include <Eigen/Dense>

class Rk4BallisticIntegrator {
public:
    // ============ 弹丸状态与角度灵敏度 ============
    struct State {
        Eigen::Vector3d position = Eigen::Vector3d::Zero();
        Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
        Eigen::Matrix<double, 3, 2> position_jacobian = Eigen::Matrix<double, 3, 2>::Zero();
        Eigen::Matrix<double, 3, 2> velocity_jacobian = Eigen::Matrix<double, 3, 2>::Zero();
    };

    struct Parameters {
        double drag_coefficient = 0.0;  // 二次阻力系数 k = rho * Cd * A / (2m)，单位 1/m
        double step = 0.01;             // RK4 最大积分步长，单位 s
        Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
        int max_steps = 10000;
    };

    explicit Rk4BallisticIntegrator(const Parameters& parameters);

    // 积分到指定飞行时间；最后一步按剩余时间缩短，失败时不修改 endpoint。
    bool integrateTo(const State& initial, double flight_time, State& endpoint) const;

    // ============ 动力学与速度 Jacobian ============
    Eigen::Vector3d acceleration(const Eigen::Vector3d& velocity) const;
    Eigen::Matrix3d accelerationJacobian(const Eigen::Vector3d& velocity) const;

    // 与旧 BallisticSolver 使用相同的空气密度和球体阻力系数；非法输入返回 NaN。
    static double sphericalDragCoefficient(double diameter, double mass);

private:
    Parameters parameters_;

    State stateDerivative(const State& state) const;
    bool stepRK4(State& state, double step) const;
};

#endif
