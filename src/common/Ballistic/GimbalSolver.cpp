// GimbalSolver.cpp — 云台角度解算器实现
#include "Ballistic/GimbalSolver.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "RobotConfig.h"

GimbalSolver::GimbalSolver() : tree_(std::make_shared<RobotTfTree>()) {
    // 偏移 / 弹丸 / 搜索参数均来自 RobotConfig（config/config.yaml）
    const RobotConfig& cfg = RobotConfig::instance();
    ballistic_ = std::make_shared<BallisticSolver>(
        cfg.common.gimbal.bulletDiameter, cfg.common.gimbal.bulletMass, cfg.common.gimbal.integrationStep);
    bullet_velocity_    = cfg.common.gimbal.bulletVelocity;
    distance_threshold_ = cfg.common.gimbal.distanceThreshold;
    distance_iterate_threshold_ = cfg.common.gimbal.distanceIterateThreshold;
    stop_z_             = cfg.common.gimbal.stopZ;
    pitch_min_          = cfg.common.gimbal.pitchMin;
    pitch_max_          = cfg.common.gimbal.pitchMax;
    pitch_step_         = cfg.common.gimbal.pitchSearchStep;
}

void GimbalSolver::setChassisPosition(float x, float y, float z) {
    tree_->setChassisPosition(x, y, z);
}

void GimbalSolver::setChassisEuler(float yaw, float pitch, float roll) {
    tree_->setChassisEuler(yaw, pitch, roll);
}

void GimbalSolver::setYaw(float yaw) {
    tree_->setYaw(yaw);
}

void GimbalSolver::setPitch(float pitch) {
    tree_->setPitch(pitch);
}

// ============================================================================
// yaw 解算（由 RobotTfTree::computeYawToAimTarget 迁移而来，推广到任意 pitch）
// ============================================================================
bool GimbalSolver::computeYawToAimTarget(const cv::Vec3f& targetWorld, float pitch, float& yawOut) const {
    // 直接读取内部树节点数据（不经缓存、不检查锁），与是否 lockAndComputeCache 无关
    const TransformTreeManager& mgr = tree_->manager();
    auto chassisNode = mgr.getNode(RobotTfTree::CHASSIS);
    auto yawNode     = mgr.getNode(RobotTfTree::YAW);
    auto pitchNode   = mgr.getNode(RobotTfTree::PITCH);
    auto headNode    = mgr.getNode(RobotTfTree::HEAD);
    auto muzzleNode  = mgr.getNode(RobotTfTree::MUZZLE);
    if (!chassisNode || !yawNode || !pitchNode || !headNode || !muzzleNode) {
        yawOut = 0.0f;
        return false;
    }

    const cv::Vec3f chassisPos   = chassisNode->getPosition();
    const cv::Vec3f chassisEuler = chassisNode->getEuler();
    const cv::Vec3f yawPos       = yawNode->getPosition();  // yaw 旋转中心相对 chassis 的位置
    const float     currentYaw   = yawNode->getEuler()[0];  // 退化情况下的回退值

    const float cp = std::cos(pitch), sp = std::sin(pitch);

    // muzzle 原点在 pitch 系中的位置 u = headPos + muzzlePos；
    // 在 yaw 系中的位置 v = pitchPos + Rx(pitch)*u（pitch = 0 时退化为 pitchPos+headPos+muzzlePos）。
    const cv::Vec3f u = headNode->getPosition() + muzzleNode->getPosition();
    const cv::Vec3f pitchPos = pitchNode->getPosition();
    const float vx = pitchPos[0] + u[0];
    const float vy = pitchPos[1] + u[1] * cp - u[2] * sp;
    const float vz = pitchPos[2] + u[1] * sp + u[2] * cp;

    // chassis 坐标系 -> world 坐标系的旋转（Rc = Rz(yaw_c) * Rx(pitch_c) * Ry(roll_c)）
    const cv::Mat Rc = CoordinateTransform::eulerToRotationMatrix(chassisEuler);
    const float r11 = Rc.at<float>(0, 0);
    const float r12 = Rc.at<float>(0, 1);
    const float r13 = Rc.at<float>(0, 2);
    const float r21 = Rc.at<float>(1, 0);
    const float r22 = Rc.at<float>(1, 1);
    const float r23 = Rc.at<float>(1, 2);

    // 在给定 pitch（且 yaw/pitch/head/muzzle 各节点 roll 均为 0）时：
    //   muzzle +y 方向在 world 系的 xy 投影：
    //     D_xy(yaw) = (Rc * Rz(yaw) * [0, cp, sp]).xy
    //               = cp*Rc2*[-sin(yaw), cos(yaw)] + sp*[r13, r23]
    //   muzzle 原点相对 K（yaw 旋转中心在 world xy 的投影）的 xy 偏移：
    //     m_xy(yaw) = Rc2*Rz2(yaw)*[vx, vy] + vz*[r13, r23]
    // 条件：cross(target_xy - K - m_xy, D_xy) = 0 且 dot(...) >= 0（射线向前经过目标）。
    // 展开 cross 后仍为 A''*sin(yaw) + B''*cos(yaw) = r'' 的标准形式：
    //   记 W = target_xy - K - vz*[r13, r23]，则
    //     A'' = cp*(Wy*r11 - Wx*r21) + sp*(vx*D + vy*C)
    //     B'' = cp*(Wx*r22 - Wy*r12) + sp*(vy*D - vx*C)
    //     r'' = cp*det(Rc2)*vx - sp*(Wx*r23 - Wy*r13)
    //   其中 C = r23*r11 - r13*r21，D = r13*r22 - r23*r12，det(Rc2) = r11*r22 - r12*r21。
    // （pitch = 0 时 sp = 0，v = pitchPos+headPos+muzzlePos，退化为原公式。）
    cv::Mat yawPosMat = (cv::Mat_<float>(3, 1) << yawPos[0], yawPos[1], yawPos[2]);
    cv::Mat yawPosWorld = Rc * yawPosMat;
    const float kx = chassisPos[0] + yawPosWorld.at<float>(0, 0);
    const float ky = chassisPos[1] + yawPosWorld.at<float>(1, 0);

    const float Wx = targetWorld[0] - kx - vz * r13;
    const float Wy = targetWorld[1] - ky - vz * r23;

    const float A1  = Wy * r11 - Wx * r21;
    const float B1  = Wx * r22 - Wy * r12;
    const float C0  = Wx * r23 - Wy * r13;
    const float C   = r23 * r11 - r13 * r21;
    const float D   = r13 * r22 - r23 * r12;
    const float det = r11 * r22 - r12 * r21;

    const float Ap = cp * A1 + sp * (vx * D + vy * C);   // p
    const float Bp = cp * B1 + sp * (vy * D - vx * C);   // q
    const float r  = cp * det * vx - sp * C0;            // 右端项

    const float R2 = Ap * Ap + Bp * Bp;
    if (R2 < 1e-12f) {
        yawOut = currentYaw;  // 退化：系数全零，无法确定 yaw
        return false;
    }
    const float R  = std::sqrt(R2);
    const float rr = r / R;
    if (rr < -1.0f || rr > 1.0f) {
        yawOut = currentYaw;  // 无解：目标在可瞄准范围之外
        return false;
    }
    const float phi   = std::atan2(Bp, Ap);                      // Ap*sin + Bp*cos = R*sin(yaw+phi)
    const float alpha = std::asin(std::max(-1.0f, std::min(1.0f, rr)));

    // 两个候选解（相差 π - 2*alpha，不一定恰好相差 π）
    const float twoPi = 2.0f * static_cast<float>(CV_PI);
    auto normalize = [twoPi](float a) {
        while (a > static_cast<float>(CV_PI)) a -= twoPi;
        while (a <= -static_cast<float>(CV_PI)) a += twoPi;
        return a;
    };

    float best = currentYaw;
    float bestDist = 1e30f;
    bool found = false;
    const float yawCands[2] = {
        normalize(alpha - phi),
        normalize(static_cast<float>(CV_PI) - alpha - phi)
    };
    for (const float yaw : yawCands) {
        const float sy = std::sin(yaw);
        const float cy = std::cos(yaw);

        // D_xy = cp*Rc2*[-sy, cy] + sp*[r13, r23]
        const float dx = cp * (-r11 * sy + r12 * cy) + sp * r13;
        const float dy = cp * (-r21 * sy + r22 * cy) + sp * r23;
        const float d2 = dx * dx + dy * dy;
        if (d2 < 1e-12f) {
            continue;  // 射线在 xy 平面的投影退化为一个点，无法瞄准
        }

        // X = Rc2*Rz2(yaw)*[vx, vy]，q = W - X = 目标相对 muzzle 原点（xy）的偏移
        const float a  = vx * cy - vy * sy;
        const float b  = vx * sy + vy * cy;
        const float qx = Wx - (r11 * a + r12 * b);
        const float qy = Wy - (r21 * a + r22 * b);
        const float q2 = qx * qx + qy * qy;
        const float dot = qx * dx + qy * dy;
        const float cross = qx * dy - qy * dx;
        if (dot < -1e-6f) {
            continue;  // 射线背向目标（目标在 muzzle 后方）
        }
        // 数值校验：cross 应约为 0（由方程构造保证，此处仅做防御）
        if (std::fabs(cross) > 1e-3f * std::sqrt(q2 * d2) + 1e-6f) {
            continue;
        }

        // 多个候选均有效时，取最接近当前 yaw 的一个（利于云台连续性）
        found = true;
        const float dist = std::fabs(normalize(yaw - currentYaw));
        if (dist < bestDist) {
            bestDist = dist;
            best = yaw;
        }
    }
    if (!found) {
        yawOut = currentYaw;
        return false;
    }
    yawOut = best;
    return true;
}

// ============================================================================
// pitch 解算
// ============================================================================
GimbalSolver::EvalContext GimbalSolver::buildContext() const {
    EvalContext ctx;
    const TransformTreeManager& mgr = tree_->manager();
    if (auto n = mgr.getNode(RobotTfTree::CHASSIS)) {
        ctx.chassisPos = n->getPosition();
        ctx.Rc = CoordinateTransform::eulerToRotationMatrix(n->getEuler());
    }
    if (auto n = mgr.getNode(RobotTfTree::YAW))    ctx.yawPos   = n->getPosition();
    if (auto n = mgr.getNode(RobotTfTree::PITCH))  ctx.pitchPos = n->getPosition();
    if (auto n = mgr.getNode(RobotTfTree::HEAD))   ctx.u += n->getPosition();
    if (auto n = mgr.getNode(RobotTfTree::MUZZLE)) ctx.u += n->getPosition();
    ctx.stopZ = stop_z_;
    return ctx;
}

BallisticSolver::NearestResult GimbalSolver::evaluateDistance(const EvalContext& ctx,
                                                              const cv::Vec3f& targetWorld,
                                                              float yaw, float pitch,
                                                              double bulletVelocity) const {
    // 解析求 muzzle 原点与 +y 指向在 world 系的位置（不修改内部树 → 可被线程池并行调用）：
    //   muzzle 原点在 yaw 系 = pitchPos + Rx(pitch)·u（u = headPos + muzzlePos），
    //   再经 Rz(yaw) 与 chassis 旋转 Rc 到 world。
    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const float sy = std::sin(yaw), cy = std::cos(yaw);
    const float vx = ctx.pitchPos[0] + ctx.u[0];
    const float vy = ctx.pitchPos[1] + ctx.u[1] * cp - ctx.u[2] * sp;
    const float vz = ctx.pitchPos[2] + ctx.u[1] * sp + ctx.u[2] * cp;
    cv::Mat m = ctx.Rc * (cv::Mat_<float>(3, 1) <<
        ctx.yawPos[0] + vx * cy - vy * sy,
        ctx.yawPos[1] + vx * sy + vy * cy,
        ctx.yawPos[2] + vz);
    const cv::Vec3f muzzle(ctx.chassisPos[0] + m.at<float>(0, 0),
                           ctx.chassisPos[1] + m.at<float>(1, 0),
                           ctx.chassisPos[2] + m.at<float>(2, 0));

    // muzzle +y 方向（world）：D = Rc·Rz(yaw)·Rx(pitch)·[0,1,0] = Rc·[-sin(yaw)cos(pitch), cos(yaw)cos(pitch), sin(pitch)]
    cv::Mat d = ctx.Rc * (cv::Mat_<float>(3, 1) << -sy * cp, cy * cp, sp);
    const cv::Vec3f dir(d.at<float>(0, 0), d.at<float>(1, 0), d.at<float>(2, 0));
    const double len = cv::norm(dir);
    if (len < 1e-9) {
        BallisticSolver::NearestResult degenerate;
        degenerate.distance = 1e30;  // 指向退化：视为不可达
        return degenerate;
    }

    // world 系下目标点 3D 坐标相对 muzzle 系原点的相对位置
    const cv::Vec3f rel = targetWorld - muzzle;

    // BallisticSolver 的弹道方向约定为 Rz(yaw)*Rx(pitch)*[0,1,0] = [-sin(yaw)cos(pitch), cos(yaw)cos(pitch), sin(pitch)]，
    // 即相对"底盘系"的指向；而 muzzle 实际指向还包含底盘旋转（Rc）。为保证弹道方向与 muzzle 精确一致，
    // 先把 muzzle 实际 world 指向换算成 BallisticSolver 的 (pitch, yaw) 参数：
    //   pitch' = asin(dz)，yaw' = atan2(-dx, dy)（cos(pitch') >= 0，换算精确）
    const double dx = dir[0] / len, dy = dir[1] / len, dz = dir[2] / len;
    const double pitchS = std::asin(std::max(-1.0, std::min(1.0, dz)));
    const double yawS   = std::atan2(-dx, dy);

    // 弹道计算截止高度（world 系，stopZ，默认 -3.0）：弹丸 world z 低于该值即停止积分。
    // 因 BallisticSolver 的坐标系原点在发射点（muzzle），需换算为发射点系高度：stopZ - muzzle_z。
    const double stop_z = ctx.stopZ - static_cast<double>(muzzle[2]);

    return ballistic_->solve(bulletVelocity, pitchS, yawS, rel[0], rel[1], rel[2], stop_z);
}

float GimbalSolver::refinePitch(const EvalContext& ctx, const cv::Vec3f& targetWorld, float yaw,
                                double bulletVelocity, float coarseBest) const {
    // 以粗搜索最优点为中心，在 [best-step, best+step] 内做黄金分割细化
    constexpr double GR = 0.6180339887498949;  // 黄金分割比例
    float a = std::max(pitch_min_, coarseBest - pitch_step_);
    float b = std::min(pitch_max_, coarseBest + pitch_step_);
    if (b - a < 1e-6f) return coarseBest;

    float c = a + static_cast<float>((1.0 - GR) * (b - a));
    float d = a + static_cast<float>(GR * (b - a));
    double fc = evaluateDistance(ctx, targetWorld, yaw, c, bulletVelocity).distance;
    double fd = evaluateDistance(ctx, targetWorld, yaw, d, bulletVelocity).distance;
    for (int i = 0; i < 24 && (b - a) > 1e-4f; ++i) {
        if (fc < fd) {
            b = d; d = c; fd = fc;
            c = a + static_cast<float>((1.0 - GR) * (b - a));
            fc = evaluateDistance(ctx, targetWorld, yaw, c, bulletVelocity).distance;
        } else {
            a = c; c = d; fc = fd;
            d = a + static_cast<float>(GR * (b - a));
            fd = evaluateDistance(ctx, targetWorld, yaw, d, bulletVelocity).distance;
        }
    }
    return 0.5f * (a + b);
}

bool GimbalSolver::computePitchToAimTarget(const cv::Vec3f& targetWorld, float yaw,
                                           double bulletVelocity,
                                           float& pitchOut, double& minDistanceOut,
                                           double& minDistancePlaneOut) const {
    double flightTime = 0.0;
    return computePitchToAimTarget(targetWorld, yaw, bulletVelocity,
                                   pitchOut, minDistanceOut, minDistancePlaneOut, flightTime);
}

bool GimbalSolver::computePitchToAimTarget(const cv::Vec3f& targetWorld, float yaw,
                                           double bulletVelocity,
                                           float& pitchOut, double& minDistanceOut,
                                           double& minDistancePlaneOut,
                                           double& flightTimeOut) const {
    // 读取一次不变的树数据（评估函数只读、不修改内部树，可并行执行）
    const EvalContext ctx = buildContext();

    // 粗搜索：先取整条距离曲线，用于收集全部局部极小值。
    // 各候选 pitch 的评估相互独立，交给线程池（TaskPool）并行执行。
    const int n = std::max(1, static_cast<int>(std::ceil((pitch_max_ - pitch_min_) / pitch_step_)));
    std::vector<double> ds(n + 1);
    pool_.run_parallel(n + 1, [&](int i) {
        const float pitch = std::min(pitch_max_, pitch_min_ + pitch_step_ * i);
        ds[i] = evaluateDistance(ctx, targetWorld, yaw, pitch, bulletVelocity).distance;
    });

    // 收集距离曲线的全部局部极小值点（含两端边界），取其中 pitch 最小的一个进行细化。
    // 目标较近时距离曲线可能有两个（弹道较水平 / 较高）甚至更多（非线性阻力）极小值，
    // 统一选取 pitch 最小的（弹道较水平）解；扫描自 pitch_min 升序，第一个极小值即 pitch 最小。
    float bestPitch = pitch_min_;
    bool found = false;
    for (int i = 0; i <= n; ++i) {
        bool isMin;
        if (n == 0) {
            isMin = true;
        } else if (i == 0) {
            isMin = ds[0] <= ds[1];
        } else if (i == n) {
            isMin = ds[n] <= ds[n - 1];
        } else {
            isMin = ds[i] <= ds[i - 1] && ds[i] <= ds[i + 1];
        }
        if (isMin) {
            bestPitch = pitch_min_ + pitch_step_ * i;
            found = true;
            break;
        }
    }
    if (!found) {
        // 兜底：取整段曲线最小距离对应的 pitch
        int bi = 0;
        for (int i = 1; i <= n; ++i) if (ds[i] < ds[bi]) bi = i;
        bestPitch = pitch_min_ + pitch_step_ * bi;
    }

    // 细化选中的极小值点（黄金分割依赖前序结果，顺序执行）
    const float refined = refinePitch(ctx, targetWorld, yaw, bulletVelocity, bestPitch);
    const BallisticSolver::NearestResult best =
        evaluateDistance(ctx, targetWorld, yaw, refined, bulletVelocity);

    pitchOut = refined;
    minDistanceOut = best.distance;
    minDistancePlaneOut = best.distance_plane;
    flightTimeOut = best.time;   // 选中 pitch 对应的弹道飞行时间
    return best.distance <= distance_threshold_;
}

// ============================================================================
// 打包解算 yaw + pitch
// ============================================================================
GimbalSolver::AimResult GimbalSolver::solveAim(const cv::Vec3f& targetWorld,
                                               double bulletVelocity) const {
    AimResult result;

    // 1) yaw（pitch = 0 假设）
    float yaw = 0.0f;
    if (!computeYawToAimTarget(targetWorld, 0.0f, yaw)) {
        return result;  // yaw 一步失效
    }

    // 2) pitch
    float pitch = 0.0f;
    double dist = 0.0, distPlane = 0.0, flight = 0.0;
    bool ok = computePitchToAimTarget(targetWorld, yaw, bulletVelocity,
                                      pitch, dist, distPlane, flight);

    // 3) 首轮最近距离超过迭代触发阈值（distance_iterate_threshold）时，触发 yaw-pitch 迭代优化：
    //    使用新 pitch 重算 yaw，并再次解算 pitch；两轮结果保留更优者（迭代只用于改进，不会使结果变差）。
    //    最终是否成功仍以原阈值（distance_threshold）为准：最终距离不超过该值才算解算成功。
    if (!ok && dist > distance_iterate_threshold_) {
        float yaw2 = yaw;
        if (computeYawToAimTarget(targetWorld, pitch, yaw2)) {
            float pitch2 = 0.0f;
            double dist2 = 0.0, distPlane2 = 0.0, flight2 = 0.0;
            const bool ok2 = computePitchToAimTarget(targetWorld, yaw2, bulletVelocity,
                                                     pitch2, dist2, distPlane2, flight2);
            if (dist2 < dist) {
                yaw = yaw2;
                pitch = pitch2;
                dist = dist2;
                distPlane = distPlane2;
                flight = flight2;
                ok = ok2;
            }
        }
    }

    // 无论成败都返回当前最优结果（失败时 success = false，调用方可查看最优尝试值）
    result.yaw = yaw;
    result.pitch = pitch;
    result.distance = dist;
    result.flight_time = flight;   // 选取的云台角度参数所用的弹道飞行时间
    if (!ok || dist > distance_threshold_) {
        return result;  // 失效
    }
    result.success = true;
    return result;
}

GimbalSolver::AimResult GimbalSolver::solveAim(const cv::Vec3f& targetWorld) const {
    return solveAim(targetWorld, bullet_velocity_);
}

cv::Vec3f GimbalSolver::muzzleWorldOrigin() const {
    // 与 evaluateDistance 的 muzzle 解析一致，但使用内部树当前的 yaw/pitch 关节角
    const TransformTreeManager& mgr = tree_->manager();
    auto chassisNode = mgr.getNode(RobotTfTree::CHASSIS);
    auto yawNode     = mgr.getNode(RobotTfTree::YAW);
    auto pitchNode   = mgr.getNode(RobotTfTree::PITCH);
    auto headNode    = mgr.getNode(RobotTfTree::HEAD);
    auto muzzleNode  = mgr.getNode(RobotTfTree::MUZZLE);
    if (!chassisNode || !yawNode || !pitchNode || !headNode || !muzzleNode) {
        return cv::Vec3f(0.0f, 0.0f, 0.0f);
    }

    const cv::Vec3f chassisPos = chassisNode->getPosition();
    const cv::Mat   Rc         = CoordinateTransform::eulerToRotationMatrix(chassisNode->getEuler());
    const cv::Vec3f yawPos     = yawNode->getPosition();
    const float     yaw        = yawNode->getEuler()[0];
    const float     pitch      = pitchNode->getEuler()[0];

    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const float sy = std::sin(yaw),   cy = std::cos(yaw);

    // muzzle 原点在 yaw 系 = pitchPos + Rx(pitch)·u（u = headPos + muzzlePos）
    const cv::Vec3f u = headNode->getPosition() + muzzleNode->getPosition();
    const float vx = pitchNode->getPosition()[0] + u[0];
    const float vy = pitchNode->getPosition()[1] + u[1] * cp - u[2] * sp;
    const float vz = pitchNode->getPosition()[2] + u[1] * sp + u[2] * cp;
    // 经 Rz(yaw) 与 chassis 旋转 Rc 到 world
    cv::Mat m = Rc * (cv::Mat_<float>(3, 1) <<
        yawPos[0] + vx * cy - vy * sy,
        yawPos[1] + vx * sy + vy * cy,
        yawPos[2] + vz);
    return cv::Vec3f(chassisPos[0] + m.at<float>(0, 0),
                     chassisPos[1] + m.at<float>(1, 0),
                     chassisPos[2] + m.at<float>(2, 0));
}

cv::Vec3f GimbalSolver::yawWorldOrigin() const {
    // yaw 系原点 = chassis 原点 + Rc·yawPos（yaw 节点位置相对 chassis，旋转不移动原点）
    const TransformTreeManager& mgr = tree_->manager();
    auto chassisNode = mgr.getNode(RobotTfTree::CHASSIS);
    auto yawNode     = mgr.getNode(RobotTfTree::YAW);
    if (!chassisNode || !yawNode) {
        return cv::Vec3f(0.0f, 0.0f, 0.0f);
    }
    const cv::Vec3f chassisPos = chassisNode->getPosition();
    const cv::Mat   Rc         = CoordinateTransform::eulerToRotationMatrix(chassisNode->getEuler());
    const cv::Vec3f yawPos     = yawNode->getPosition();
    cv::Mat m = Rc * (cv::Mat_<float>(3, 1) << yawPos[0], yawPos[1], yawPos[2]);
    return cv::Vec3f(chassisPos[0] + m.at<float>(0, 0),
                     chassisPos[1] + m.at<float>(1, 0),
                     chassisPos[2] + m.at<float>(2, 0));
}
