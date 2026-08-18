#ifndef BALLISTICSOLVER_H
#define BALLISTICSOLVER_H

#include <vector>
#include <cmath>

class BallisticSolver {
public:
    // 最近点结果结构体（公开）
    struct NearestResult {
        double x, y, z;      // 三维坐标
        double vx, vy, vz;   // 速度
        double distance;     // 与目标点的欧氏距离（米）
        double distance_plane;    // 距离在弹道平面内的分量（米）
        double distance_vertical; // 距离垂直弹道平面的分量（米），distance² = distance_plane² + distance_vertical²
        double time;         // 从发射到该点的飞行时间
    };

    // 构造函数：直径(m)，质量(kg)，积分步长(s)，细化容差(m), 最大细化次数
    BallisticSolver(double diameter, double mass, double step, double tolerance=1e-2, int max_refine_depth=3);

    // 主计算接口
    // 弹道方向约定与云台 muzzle 系 +y 轴一致：
    //   v0/|v0| = Rz(yaw) * Rx(pitch) * [0,1,0]
    //           = [-sin(yaw)*cos(pitch), cos(yaw)*cos(pitch), sin(pitch)]
    // target_x/y/z 为发射点（muzzle 原点）坐标系下的目标位置（即目标相对发射点的坐标）。
    NearestResult solve(double v0_mag, double pitch, double yaw,
                        double target_x, double target_y, double target_z,
                        double stop_z);

private:
    // ---------- 成员变量（统一加下划线后缀） ----------
    double dt_;          // 积分步长
    double tol_;         // 细化容差
    double mass_;        // 弹丸质量
    double k_;           // 空气阻力系数 (0.5*rho*Cd*A/m)
    double area_;        // 截面积

    int max_refine_depth_;

    struct Vec3 {
        double x, y, z;
        Vec3(double x_ = 0, double y_ = 0, double z_ = 0) : x(x_), y(y_), z(z_) {}
        Vec3 operator+(const Vec3& o) const;
        Vec3 operator*(double s) const;
        Vec3 operator-(const Vec3& o) const;
        double dot(const Vec3& o) const;
        Vec3 cross(const Vec3& o) const;
        double norm() const;
        Vec3 normalized() const;
    };
    Vec3 g_;             // 重力向量 (0, 0, -9.81)

    struct TrajectoryPoint {
        double t;        // 时间
        double x, y;     // 二维平面坐标
        double vx, vy;   // 二维速度
    };

    // ---------- 私有方法 ----------
    void acceleration(double x, double y, double vx, double vy,
                      double& ax, double& ay, const Vec3& e1, const Vec3& e2);
    void step(TrajectoryPoint& p, double dt, const Vec3& e1, const Vec3& e2);
    void integrate(double x0, double y0, double vx0, double vy0,
                   double dt, const Vec3& e1, const Vec3& e2,
                   double stop_z, std::vector<TrajectoryPoint>& points);
    int findNearest(const std::vector<TrajectoryPoint>& pts,
                    double s_target, double t_target);
    TrajectoryPoint refine(const std::vector<TrajectoryPoint>& coarse,
                           int idx, double s, double t,
                           const Vec3& e1, const Vec3& e2, double stop_z, int left_depth);
};

#endif
