// big_small_yaw_test.cpp — 大小 yaw 构型离线自测（无硬件、无需相机/串口）
//
// 用法：
//   ./big_small_yaw_test planner                      # 打印 C++ 轨迹规划器扫描结果（与 Python 对照）
//   ./big_small_yaw_test mode single|big_small        # 变换树构型 + 越界调用 + 瞄准解算几何校验
//   ./big_small_yaw_test split                        # 拆分器：限幅/跳变最少化行为的合成用例
//
// 注意：RobotConfig::instance() 按 config/selector.yaml 选择机器配置，因此 mode 用例需要
// 先把 selector 指向期望的配置（huhu233PC = single / Sentry1 = big_small）。
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "common/Ballistic/GimbalSolver.h"
#include "common/BigSmallYaw/BigSmallYawSplitter.h"
#include "common/BigSmallYaw/BigSmallYawTrajectoryPlanner.h"
#include "common/RobotConfig.h"
#include "common/TransformTree/RobotTfTree.h"

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::cout << (ok ? "  [ok]   " : "  [FAIL] ") << what << std::endl;
    if (!ok) ++g_fail;
}

// ── 1) 轨迹规划器：打印扫描结果（供与 Python 版逐点对比）──
int runPlanner() {
    bsy::TrajectoryPlanner planner(3.0, 20.0, 300.0);
    const std::vector<double> targets = {0.1, 0.2, 0.5, 0.5, 0.5, 0.4, 0.2, -0.1, -0.5, -0.5};
    std::vector<double> pos, vel, acc, jerk;
    planner.scan(targets, 0.0, 0.0, 0.0, 0.01, /*substeps=*/1, pos, vel, acc, jerk);
    std::printf("# i target pos vel acc jerk\n");
    for (size_t i = 0; i < targets.size(); ++i) {
        std::printf("%zu %.10f %.10f %.10f %.10f %.10f\n", i, targets[i], pos[i], vel[i],
                    acc[i], jerk[i]);
    }
    return 0;
}

// ── 1b) 轨迹规划器：从文件读取用例（每行：V A J substeps p0 v0 a0 dt n target...）──
int runPlannerFile(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) { std::cerr << "cannot open " << path << std::endl; return 2; }
    double V, A, J, p0, v0, a0, dt;
    int substeps, n;
    while (std::fscanf(f, "%lf %lf %lf %d %lf %lf %lf %lf %d", &V, &A, &J, &substeps,
                       &p0, &v0, &a0, &dt, &n) == 9) {
        std::vector<double> targets((size_t)n);
        for (int i = 0; i < n; ++i) std::fscanf(f, "%lf", &targets[(size_t)i]);
        bsy::TrajectoryPlanner planner(V, A, J);
        std::vector<double> pos, vel, acc, jerk;
        planner.scan(targets, p0, v0, a0, dt, substeps, pos, vel, acc, jerk);
        for (int i = 0; i < n; ++i) {
            std::printf("%.12f %.12f %.12f %.12f\n", pos[(size_t)i], vel[(size_t)i],
                        acc[(size_t)i], jerk[(size_t)i]);
        }
    }
    std::fclose(f);
    return 0;
}

// ── 2) 构型/几何校验 ──
int runMode(const std::string& expect) {
    const RobotConfig& cfg = RobotConfig::instance();
    const YawMode mode = cfg.common.bigSmallYaw.mode;
    const bool want_big_small = (expect == "big_small");
    // 配置分支约定：只填当前构型那一支（另一支可整段省略）
    check(cfg.common.bigSmallYaw.singlePresent == !want_big_small,
          "配置文件只写了当前构型的分支（singlePresent / bigSmallPresent 与 mode 一致）");
    check((mode == YawMode::BIG_SMALL) == want_big_small,
          "配置构型 = " + expect + "（selector 指向的机器配置）");

    RobotTfTree tree;
    check(tree.isBigSmallYaw() == want_big_small, "变换树按配置定型构型");
    // 越界调用必须抛异常
    bool threw = false;
    try {
        if (want_big_small) tree.setYaw(0.1f); else tree.setYawBig(0.1f);
    } catch (const std::logic_error&) {
        threw = true;
    }
    check(threw, "越界调用（不属当前构型的方法）抛 std::logic_error");

    // 节点链：BIG_SMALL 时应存在 yaw_big / yaw_small 且无 yaw
    const bool has_big = tree.manager().getNode(RobotTfTree::YAW_BIG) != nullptr;
    const bool has_small = tree.manager().getNode(RobotTfTree::YAW_SMALL) != nullptr;
    const bool has_single = tree.manager().getNode(RobotTfTree::YAW) != nullptr;
    if (want_big_small) {
        check(has_big && has_small && !has_single, "变换链为 chassis -> yaw_big -> yaw_small");
    } else {
        check(!has_big && !has_small && has_single, "变换链为 chassis -> yaw");
    }

    // 瞄准解算几何：解出的 yaw 应使 muzzle +y 射线在水平面内指向目标
    GimbalSolver solver;
    const double chassis_yaw = 0.0, pitch = 0.0;
    const float big_joint = want_big_small ? 0.30f : 0.0f;
    const float small_joint_start = want_big_small ? 0.05f : 0.20f;
    solver.tree().unlock();
    solver.setChassisPosition(0.0f, 0.0f, 0.0f);
    solver.setChassisEuler((float)chassis_yaw, 0.0f, 0.0f);
    if (want_big_small) {
        solver.setYawBig(big_joint);
        solver.setYawSmall(small_joint_start);
    } else {
        solver.setYaw(small_joint_start);
    }
    solver.setPitch((float)pitch);

    // 目标点：世界系 (5, 1, 1)（离轴，便于检验小 yaw 轴偏移的影响）
    const cv::Vec3f target(5.0f, 1.0f, 1.0f);
    float yaw_out = 0.0f;
    const bool ok = want_big_small
        ? solver.computeYawToAimTarget(target, (float)pitch, big_joint, yaw_out)
        : solver.computeYawToAimTarget(target, (float)pitch, yaw_out);
    check(ok, "yaw 解算成功");

    // 用解出的 θ_total 摆放云台，检查 muzzle +y 方向的 xy 方位角 == 目标 xy 方位角
    solver.tree().unlock();
    if (want_big_small) {
        solver.setYawBig(big_joint);
        solver.setYawSmall(yaw_out - big_joint);
    } else {
        solver.setYaw(yaw_out);
    }
    solver.tree().lockAndComputeCache();
    const cv::Vec3f dir_world = solver.tree().transformVector(
        RobotTfTree::MUZZLE, RobotTfTree::WORLD, cv::Vec3f(0.0f, 1.0f, 0.0f));
    const cv::Vec3f muzzle = solver.tree().transformPoint(
        RobotTfTree::MUZZLE, RobotTfTree::WORLD, cv::Vec3f(0.0f, 0.0f, 0.0f));
    const double aim_az = std::atan2(dir_world[1], dir_world[0]);
    const double los_az = std::atan2(target[1] - muzzle[1], target[0] - muzzle[0]);
    const double az_err = std::remainder(aim_az - los_az, 2.0 * M_PI);
    std::printf("  muzzle=(%.4f, %.4f, %.4f) aim_az=%.6f rad los_az=%.6f rad err=%.6f rad\n",
                muzzle[0], muzzle[1], muzzle[2], aim_az, los_az, az_err);
    check(std::fabs(az_err) < 1e-5, "解出的 yaw 使 muzzle 指向目标（方位角误差 < 1e-5 rad）");
    return 0;
}

// ── 3) 拆分器行为 ──
int runSplit() {
    const RobotConfig& cfg = RobotConfig::instance();
    const auto& bp = cfg.common.bigSmallYaw.bigSmall;   // big_small 分支
    bsy::BigSmallYawSplitterConfig sc;
    sc.smallMinAngle = bp.joints.smallMinAngle;
    sc.smallMaxAngle = bp.joints.smallMaxAngle;
    sc.smallCenterAngle = bp.joints.smallCenterAngle;
    sc.softLimitRatio = bp.robotController.mpc.smallLimitSoftRatio;
    sc.plannerMaxVelocity = bp.splitter.plannerMaxVelocity;
    sc.plannerMaxAcceleration = bp.splitter.plannerMaxAcceleration;
    sc.plannerMaxJerk = bp.splitter.plannerMaxJerk;
    sc.plannerSubsteps = bp.splitter.plannerSubsteps;
    bsy::BigSmallYawSplitter splitter(sc);
    std::printf("  软限位边界: [%.4f, %.4f] rad ([%.2f, %.2f] deg)\n", splitter.softMin(),
                splitter.softMax(), splitter.softMin() * 180.0 / M_PI,
                splitter.softMax() * 180.0 / M_PI);

    const double dt = cfg.common.dtControl();
    const int N = 24;
    auto runCase = [&](const std::string& name, const std::vector<double>& aim,
                       int expect_episodes_max) {
        splitter.reset();   // 各用例相互独立（真实运行时连续帧共用同一实例，见“连续多帧”用例）
        const auto t0 = std::chrono::steady_clock::now();
        const auto out = splitter.split(aim, dt, aim.empty() ? 0.0 : aim[0], t0);
        double worst = 0.0;
        for (double th : out.theta_small) worst = std::max(worst, std::fabs(th));
        const bool within = worst <= std::max(splitter.softMax(), -splitter.softMin()) + 1e-9;
        std::printf("  %-12s |theta_s|max=%.4f rad (%.2f deg)  被修正点=%d  无限幅段=%d  over_limit=%d\n",
                    name.c_str(), worst, worst * 180.0 / M_PI, out.jump_count,
                    out.unlimited_episodes, (int)out.over_limit);
        check(within, name + " 每点 |θ_small| 均在软限位内");
        check(out.unlimited_episodes <= expect_episodes_max,
              name + " 无限幅运动段数 <= " + std::to_string(expect_episodes_max));
        return out;
    };

    // A) 慢速目标（约 0.2 rad/s）：大 yaw 平滑跟随，θ_small 接近 0，无需无限幅动作
    {
        std::vector<double> aim(N);
        for (int i = 0; i < N; ++i) aim[i] = 0.2 * (i + 1) * dt;
        const auto out = runCase("慢速目标", aim, 0);
        double worst = 0.0;
        for (double th : out.theta_small) worst = std::max(worst, std::fabs(th));
        check(worst < 0.01, "慢速目标：小 yaw 偏离 0 点 < 0.01 rad");
    }
    // B) 快速目标（约 8 rad/s，远超大 yaw 平滑规划器 V=3 rad/s）：必须让大 yaw 无限幅跟进，
    //    表现为**一段连续的修正**（而非多次独立动作）
    {
        std::vector<double> aim(N);
        for (int i = 0; i < N; ++i) aim[i] = 8.0 * (i + 1) * dt;
        runCase("快速目标", aim, 1);
    }
    // C) 静止目标：大 yaw 停在目标方位角，θ_small ≡ 0，无无限幅动作
    {
        std::vector<double> aim(N, 0.3);
        const auto out = runCase("静止目标", aim, 0);
        double worst = 0.0;
        for (double th : out.theta_small) worst = std::max(worst, std::fabs(th));
        check(worst < 1e-9, "静止目标：小 yaw 关节角 ≈ 0");
    }
    // D) 跨 ±π 的目标（wrap）：解卷绕后不应产生修正
    {
        std::vector<double> aim(N);
        for (int i = 0; i < N; ++i) {
            const double a = 3.0 + 0.5 * (i + 1) * dt;   // 越过 π，输入被 wrap
            aim[i] = std::remainder(a, 2.0 * M_PI);
        }
        runCase("跨 pi 目标", aim, 0);
    }
    // E) 连续多帧（模拟真实重规划）：同一实例、时间戳递增、目标慢速运动 →
    //    跨帧暖启动不应产生额外无限幅动作
    {
        splitter.reset();
        int episodes_total = 0;
        auto t0 = std::chrono::steady_clock::now();
        double psi = 0.0;
        for (int frame = 0; frame < 30; ++frame) {
            std::vector<double> aim(N);
            for (int i = 0; i < N; ++i) aim[i] = psi + 0.3 * (i + 1) * dt;   // 0.3 rad/s
            const auto out = splitter.split(aim, dt, aim[0], t0);
            episodes_total += out.unlimited_episodes;
            psi += 0.3 * dt * 4.0;   // 每帧推进 4 个控制周期（≈ 相机帧率下的重规划间隔）
            t0 += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(4.0 * dt));
        }
        std::printf("  连续多帧    目标 0.3 rad/s，30 帧累计无限幅段=%d\n", episodes_total);
        check(episodes_total == 0, "连续多帧：慢速目标下无无限幅动作（暖启动无抖动）");
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::string cmd = (argc > 1) ? argv[1] : "planner";
    if (cmd == "planner") return runPlanner();
    if (cmd == "plannerfile") return runPlannerFile((argc > 2) ? argv[2] : "");
    if (cmd == "mode") return runMode((argc > 2) ? argv[2] : "single");
    if (cmd == "split") return runSplit();
    std::cerr << "usage: " << argv[0] << " planner|mode <single|big_small>|split" << std::endl;
    return 2;
}
