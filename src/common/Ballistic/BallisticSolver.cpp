#include "common/Ballistic/BallisticSolver.h"
#include <algorithm>
#include <stdexcept>

// ============ Vec3 成员函数实现 ============
BallisticSolver::Vec3 BallisticSolver::Vec3::operator+(const Vec3& o) const {
    return {x + o.x, y + o.y, z + o.z};
}
BallisticSolver::Vec3 BallisticSolver::Vec3::operator*(double s) const {
    return {x * s, y * s, z * s};
}
BallisticSolver::Vec3 BallisticSolver::Vec3::operator-(const Vec3& o) const {
    return {x - o.x, y - o.y, z - o.z};
}
double BallisticSolver::Vec3::dot(const Vec3& o) const {
    return x * o.x + y * o.y + z * o.z;
}
BallisticSolver::Vec3 BallisticSolver::Vec3::cross(const Vec3& o) const {
    return Vec3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
}
double BallisticSolver::Vec3::norm() const {
    return std::sqrt(dot(*this));
}
BallisticSolver::Vec3 BallisticSolver::Vec3::normalized() const {
    double n = norm();
    if (n > 0) return *this * (1.0 / n);
    return Vec3(0, 0, 0);
}

// ============ 构造函数 ============
BallisticSolver::BallisticSolver(double diameter, double mass, double step, double tolerance, int max_refine_depth)
    : dt_(step), tol_(tolerance), mass_(mass), g_(0, 0, -9.81), max_refine_depth_(max_refine_depth) {
    const double rho_air = 1.225;   // kg/m³
    const double Cd = 0.47;         // 球体阻力系数 (亚临界区)
    double r = diameter / 2.0;
    area_ = M_PI * r * r;
    k_ = 0.5 * rho_air * Cd * area_ / mass_;
}

// ============ 加速度计算（二维平面） ============
void BallisticSolver::acceleration(double x, double y, double vx, double vy,
                                   double& ax, double& ay, const Vec3& e1, const Vec3& e2) {
    Vec3 g_2d = {g_.dot(e1), g_.dot(e2)};
    double speed = std::sqrt(vx * vx + vy * vy);
    double drag_mag = k_ * speed;   // a_drag = -k * |v| * v
    ax = g_2d.x - drag_mag * vx;
    ay = g_2d.y - drag_mag * vy;
}

// ============ 速度Verlet变体单步 ============
void BallisticSolver::step(TrajectoryPoint& p, double dt, const Vec3& e1, const Vec3& e2) {
    double ax, ay;
    acceleration(p.x, p.y, p.vx, p.vy, ax, ay, e1, e2);
    // 预估位置
    double x_new = p.x + p.vx * dt + 0.5 * ax * dt * dt;
    double y_new = p.y + p.vy * dt + 0.5 * ay * dt * dt;
    // 预估速度（用于计算新加速度）
    double vx_pred = p.vx + ax * dt;
    double vy_pred = p.vy + ay * dt;
    // 新加速度
    double ax_new, ay_new;
    acceleration(x_new, y_new, vx_pred, vy_pred, ax_new, ay_new, e1, e2);
    // 校正速度
    p.vx = p.vx + 0.5 * (ax + ax_new) * dt;
    p.vy = p.vy + 0.5 * (ay + ay_new) * dt;
    // 更新位置
    p.x = x_new;
    p.y = y_new;
    p.t += dt;
}

// ============ 积分至停止条件 ============
void BallisticSolver::integrate(double x0, double y0, double vx0, double vy0,
                                double dt, const Vec3& e1, const Vec3& e2,
                                double stop_z, std::vector<TrajectoryPoint>& points) {
    TrajectoryPoint p{0.0, x0, y0, vx0, vy0};
    points.clear();
    points.push_back(p);
    const int MAX_STEPS = 100000;
    for (int i = 0; i < MAX_STEPS; ++i) {
        step(p, dt, e1, e2);
        points.push_back(p);
        // 三维位置与速度的 z 分量
        Vec3 pos = e1 * p.x + e2 * p.y;
        Vec3 vel = e1 * p.vx + e2 * p.vy;
        if (pos.z < stop_z && vel.z < 0.0) {
            break;
        }
    }
}

// ============ 寻找离散最近点索引 ============
int BallisticSolver::findNearest(const std::vector<TrajectoryPoint>& pts,
                                 double s_target, double t_target) {
    int idx = 0;
    double min_dist2 = 1e100;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        double dx = pts[i].x - s_target;
        double dy = pts[i].y - t_target;
        double d2 = dx * dx + dy * dy;
        if (d2 < min_dist2) {
            min_dist2 = d2;
            idx = static_cast<int>(i);
        }
    }
    return idx;
}

// ============ 二次插值细化（必要时局部加密） ============
BallisticSolver::TrajectoryPoint BallisticSolver::refine(const std::vector<TrajectoryPoint>& coarse,
                                                         int idx, double s, double t,
                                                         const Vec3& e1, const Vec3& e2,
                                                         double stop_z, int left_depth) {
    int i0 = std::max(0, idx - 1);
    int i2 = std::min(static_cast<int>(coarse.size()) - 1, idx + 1);
    if (i0 == i2) return coarse[idx];

    auto dist2 = [&](const TrajectoryPoint& p) {
        double dx = p.x - s, dy = p.y - t;
        return dx * dx + dy * dy;
    };
    double d0 = dist2(coarse[i0]);
    double d1 = dist2(coarse[idx]);
    double d2 = dist2(coarse[i2]);

    // 二次插值求极小点 λ ∈ [-1,1]
    double a = (d0 + d2 - 2 * d1) / 2.0;
    double b = (d2 - d0) / 2.0;
    double lambda = -b / (2.0 * a);
    lambda = std::max(-1.0, std::min(1.0, lambda));

    double d_min_interp = a * lambda * lambda + b * lambda + d1;
    double d_min_disc = d1;

    // 精度检查：插值最小距离与离散最小距离之差小于 tol_²
    if ((std::fabs(d_min_interp - d_min_disc) < tol_ * tol_) || (left_depth <= 0)) {
        double alpha = (lambda + 1.0) / 2.0;   // 0~1 插值系数
        TrajectoryPoint res;
        res.t = coarse[i0].t + alpha * (coarse[i2].t - coarse[i0].t);
        res.x = coarse[i0].x + alpha * (coarse[i2].x - coarse[i0].x);
        res.y = coarse[i0].y + alpha * (coarse[i2].y - coarse[i0].y);
        res.vx = coarse[i0].vx + alpha * (coarse[i2].vx - coarse[i0].vx);
        res.vy = coarse[i0].vy + alpha * (coarse[i2].vy - coarse[i0].vy);
        return res;
    } else {
        // 局部加密积分（步长缩小20倍）
        TrajectoryPoint left = coarse[i0];
        TrajectoryPoint right = coarse[i2];
        double dt_fine = (right.t - left.t) / 20.0;
        std::vector<TrajectoryPoint> fine_points;
        TrajectoryPoint p_fine = left;
        fine_points.push_back(p_fine);
        while (p_fine.t < right.t - 1e-12) {
            step(p_fine, dt_fine, e1, e2);
            fine_points.push_back(p_fine);
            // 若意外触发停止条件则跳出
            Vec3 pos = e1 * p_fine.x + e2 * p_fine.y;
            Vec3 vel = e1 * p_fine.vx + e2 * p_fine.vy;
            if (pos.z < stop_z && vel.z < 0) break;
        }
        int fine_idx = findNearest(fine_points, s, t);
        // 递归细化（可迭代）
        return refine(fine_points, fine_idx, s, t, e1, e2, stop_z, left_depth - 1);
    }
}

// ============ 主求解接口 ============
BallisticSolver::NearestResult BallisticSolver::solve(double v0_mag, double pitch, double yaw,
                                                      double target_x, double target_y, double target_z,
                                                      double stop_z) {
    // 1. 计算初始速度三维向量
    double sp = std::sin(pitch), cp = std::cos(pitch);
    double sy = std::sin(yaw), cy = std::cos(yaw);
    // 方向约定：与云台 muzzle 系 +y 轴一致，即先绕 x 轴转 pitch（抬头），再绕 z 轴转 yaw：
    //   v = Rz(yaw) * Rx(pitch) * [0,1,0] = [-sin(yaw)*cos(pitch), cos(yaw)*cos(pitch), sin(pitch)]
    double dx = -sy * cp;
    double dy =  cy * cp;
    double dz =  sp;
    Vec3 v0 = Vec3(dx, dy, dz) * v0_mag;

    // 2. 建立弹道平面基向量
    Vec3 e1 = v0.normalized();
    Vec3 e2 = (g_ - e1 * (g_.dot(e1))).normalized();
    if (e2.norm() < 1e-12) {  // 竖直发射，选任意水平方向
        Vec3 ref = (std::fabs(e1.x) < 0.9) ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
        e2 = (ref - e1 * (ref.dot(e1))).normalized();
    }

    // 3. 目标点在平面上的投影
    Vec3 target(target_x, target_y, target_z);
    double s_target = target.dot(e1);
    double t_target = target.dot(e2);

    // 4. 二维初始状态（沿e1方向发射）
    double vx0 = v0_mag, vy0 = 0.0;
    double x0 = 0.0, y0 = 0.0;

    // 5. 粗积分
    std::vector<TrajectoryPoint> coarse_points;
    integrate(x0, y0, vx0, vy0, dt_, e1, e2, stop_z, coarse_points);

    // 6. 找离散最近点索引
    int idx = findNearest(coarse_points, s_target, t_target);

    // 7. 细化
    TrajectoryPoint best = 
        max_refine_depth_ >= 1 ?
        refine(coarse_points, idx, s_target, t_target, e1, e2, stop_z, max_refine_depth_ - 1) :
        coarse_points[idx];

    // 8. 转换回三维并构造结果
    Vec3 pos = e1 * best.x + e2 * best.y;
    Vec3 vel = e1 * best.vx + e2 * best.vy;
    double dist = (pos - target).norm();

    NearestResult result;
    result.x = pos.x;   result.y = pos.y;   result.z = pos.z;
    result.vx = vel.x;  result.vy = vel.y;  result.vz = vel.z;
    result.distance = dist;
    result.time = best.t;
    // 距离分解：弹道平面内分量（2D 最近距离）与垂直弹道平面分量（目标相对平面的法向偏移），
    // 两者平方和等于 distance²（e1、e2、n 构成正交基）。
    result.distance_plane = std::sqrt((s_target - best.x) * (s_target - best.x) +
                                      (t_target - best.y) * (t_target - best.y));
    result.distance_vertical = std::fabs(target.dot(e1.cross(e2)));
    return result;
}
