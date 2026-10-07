// azimuth_wrap_test.cpp — 「整圈表示差」修复的回归自测
//
// ★ 回归对象（一次真实故障）：
//   严格反解里的底盘方位角由 atan2 给出，落在 (−π, π]；底盘每转一圈就跳 ±2π。
//   而 mpc::DualYawMpcController::measure() 按 psi_b = chassis_azimuth + θ_b
//   （θ_b = 电控给的多圈关节角）合成世界方位角进 MPC 代价，参考序列却只在
//   McuMpcController::set()（上位机每帧一次）对齐过一次圈数 ⇒ 底盘跨过 ±π 到下一帧之间
//   MPC 代价里出现 w_psi_b·(2π)² ≈ 39.5 的整圈残差（正常跟踪 ~4e-4）⇒ 底盘每转一圈
//   打出一个恒定幅值的力矩脉冲（大 yaw 抖一下），且固定在同一底盘角度。
//
// 本测试覆盖两层修复：
//   A) tcbs::mpc::angle_wrap（每拍整圈对齐）：只做整数圈平移、不改形状、小偏差恒等；
//   B) tcbs::com::FullStrictPoseBuilder：chassis_azimuth 多圈连续（不再每圈跳 2π），
//      big_azimuth = chassis_azimuth + θ_b 同步连续；chassis_euler_yaw 仍保持包裹值。
//
// 编译运行（不需要相机/模型/串口；FullStrictPoseBuilder 只依赖 libstdc++）。
// 注意：子模组的 include 路径是"按目录分别加"，所以要同时给 include / include/com /
// include/mpc（与子模组 CMakeLists 的 target_include_directories 一致）。
//
//   g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic
//       -Isub_module/TorqueControllerForBigSmallYaw_v2/include
//       -Isub_module/TorqueControllerForBigSmallYaw_v2/include/com
//       -Isub_module/TorqueControllerForBigSmallYaw_v2/include/mpc
//       test/azimuth_wrap_test.cpp
//       sub_module/TorqueControllerForBigSmallYaw_v2/src/com/FullStrictPoseBuilder.cpp
//       -o /tmp/azimuth_wrap_test -lpthread
//   /tmp/azimuth_wrap_test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "com/FullStrictPoseBuilder.h"
#include "mpc/angle_wrap.hpp"

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "  [ok]   " : "  [FAIL] ") << what << std::endl;
    if (!ok) ++g_fail;
}

constexpr double kPi    = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

/// 把角包裹到 (−π, π]（模拟 IMU 报文给出的单圈值；具体取哪一圈不影响反解，
/// 因为 cos/sin 是 2π 周期函数）。
double wrapPi(double a) {
    double r = std::fmod(a + kPi, kTwoPi);
    if (r < 0.0) r += kTwoPi;
    return r - kPi;
}

// ── A) 每拍整圈对齐（tcbs::mpc::angle_wrap）──
void testAngleWrap() {
    std::cout << "\n[A] angle_wrap：整圈对齐" << std::endl;

    // 小偏差 ⇒ 恒等（绝不能把跟踪误差吃掉）
    for (double e : {0.0, 1e-9, 1e-3, 0.05, 0.2, 3.0}) {
        const double ref = -1.234;
        char buf[128];
        std::snprintf(buf, sizeof(buf), "|差| < π 时 wrapToNearest 恒等（e=%.3g）", e);
        check(tcbs::mpc::wrapToNearest(ref + e, ref) == ref + e, buf);
    }
    // 整圈差 ⇒ 拉回同圈，且平移量恰好是整数圈、方向相反
    for (double turns : {1.0, -1.0, 3.0, -4.0, 2.0}) {
        const double ref = 2.5;
        const double v = ref + 0.3 + turns * kTwoPi;
        const double out = tcbs::mpc::wrapToNearest(v, ref);
        const double k = std::round((out - v) / kTwoPi);
        check(std::fabs(out - (ref + 0.3)) < 1e-12 && k == -turns,
              "整圈差被拉回同圈、平移量为整数圈（turns=" + std::to_string(turns) + "）");
    }
    // 序列：只整体平移 ⇒ 形状不变；首元素落到被参考最近的一圈
    std::vector<double> seq;
    for (int i = 0; i < 20; ++i) seq.push_back(-7.0 - kTwoPi + 0.01 * i);
    const std::vector<double> before = seq;
    tcbs::mpc::alignSeqToNearest(seq, -7.0);
    bool shape_ok = (seq.size() == before.size());
    for (size_t i = 0; i < seq.size() && shape_ok; ++i) {
        shape_ok = std::fabs((seq[i] - seq[0]) - (before[i] - before[0])) < 1e-12;
    }
    check(shape_ok, "alignSeqToNearest 只整体平移：各点相对关系逐点不变");
    check(std::fabs(seq.front() - (-7.0)) < 1e-12,
          "alignSeqToNearest 首元素落到最近的同圈（−7.0−2π → −7.0）");
    // 小偏差序列 ⇒ 一个元素都不许动
    std::vector<double> seq2;
    for (int i = 0; i < 20; ++i) seq2.push_back(-7.0 + 0.02 + 0.01 * i);
    const std::vector<double> before2 = seq2;
    tcbs::mpc::alignSeqToNearest(seq2, -7.0);
    check(seq2 == before2, "alignSeqToNearest 在 |差| < π 时逐元素恒等（不动跟踪偏差）");
    std::vector<double> empty;
    tcbs::mpc::alignSeqToNearest(empty, 1.0);
    check(empty.empty(), "alignSeqToNearest 空序列不崩");
}

// ── B) 严格反解：底盘转多圈（正/反/高速）时 chassis_azimuth 必须连续 ──
struct RunResult {
    double max_step_az = 0.0;    // |Δ chassis_azimuth| 最大值（应 ≈ |ω|·dt，绝不能 ≈ 2π）
    double max_step_big = 0.0;   // |Δ big_azimuth| 最大值
    double max_err_az = 0.0;     // 与多圈真值之差（去掉首样本锚定常数）的最大值
    double max_step_euler = 0.0; // |Δ chassis_euler_yaw| 的包裹差（应 ≤ π）
    bool   euler_wrapped = true; // chassis_euler_yaw 始终在 (−π, π]
    bool   big_identity = true;  // big_azimuth == chassis_azimuth + θ_b
};

/// 合成"底盘匀速自转 + 大 yaw 关节另转"的 IMU/MCU 样本流，喂给严格反解（ON_HEAD）。
RunResult runChassisSpin(double rate, double joint_rate, double pitch) {
    tcbs::com::FullStrictPoseBuilder builder(
        tcbs::com::FullStrictPoseBuilder::ImuLocation::ON_HEAD);

    const double dt = 0.01;
    const int steps = 900;                   // 9 s
    const double joint_b0 = 6.0907;          // 首样本关节角（与实车日志同量级）

    RunResult res;
    double prev_az = 0.0, prev_big = 0.0, prev_euler = 0.0;
    double anchor = 0.0;
    for (int k = 0; k < steps; ++k) {
        const double t = k * dt;
        const double chassis_true = rate * t;               // 底盘真实（多圈）方位角
        const double joint_b = joint_b0 + joint_rate * t;   // 大 yaw 关节角（多圈）
        // ON_HEAD 正解：头（IMU）世界方位角 = 底盘方位角 + θ_b + θ_s（θ_s = 0）
        const double imu_yaw_true = chassis_true + joint_b;

        tcbs::com::FullStrictPoseBuilder::ImuSample imu;
        imu.euler_yaw = wrapPi(imu_yaw_true);   // IMU 报文只给单圈值
        imu.euler_pitch = pitch;
        imu.euler_roll = 0.0;
        builder.onImu(imu);

        tcbs::com::FullStrictPoseBuilder::McuSample mcu;
        mcu.yaw_big_angle = joint_b;            // 电控上报的是多圈关节角
        mcu.yaw_small_angle = 0.0;
        mcu.pitch_angle = pitch;
        builder.onMcu(mcu);

        const tcbs::com::FullStrictPoseBuilder::StrictPose sp = builder.strictPose();

        if (k == 0) {
            anchor = sp.chassis_azimuth - chassis_true;   // 子模组只保证"相对连续"
        } else {
            res.max_step_az = std::max(res.max_step_az,
                                       std::fabs(sp.chassis_azimuth - prev_az));
            res.max_step_big = std::max(res.max_step_big,
                                        std::fabs(sp.big_azimuth - prev_big));
            res.max_step_euler = std::max(res.max_step_euler,
                                          std::fabs(wrapPi(sp.chassis_euler_yaw - prev_euler)));
        }
        res.max_err_az = std::max(res.max_err_az,
                                  std::fabs((sp.chassis_azimuth - chassis_true) - anchor));
        if (sp.chassis_euler_yaw > kPi + 1e-9 || sp.chassis_euler_yaw < -kPi - 1e-9) {
            res.euler_wrapped = false;
        }
        if (sp.big_azimuth != sp.chassis_azimuth + sp.yaw_big_angle) {
            res.big_identity = false;
        }
        prev_az = sp.chassis_azimuth;
        prev_big = sp.big_azimuth;
        prev_euler = sp.chassis_euler_yaw;
    }
    return res;
}

void testStrictPoseUnwrap() {
    std::cout << "\n[B] FullStrictPoseBuilder：chassis_azimuth 多圈连续" << std::endl;

    struct Case {
        const char* name;
        double rate;
        double joint_rate;
        double pitch;
    };
    const Case cases[] = {
        {"底盘 +2.0 rad/s、关节 +3.8（实车日志工况）", 2.0, 3.8, 0.0},
        {"底盘 −2.0 rad/s（反向）", -2.0, 3.8, 0.0},
        {"底盘 +1.2、关节 −0.5（反向）", 1.2, -0.5, 0.0},
        {"底盘 +2.0、关节 +3.8、pitch 0.2", 2.0, 3.8, 0.2},
        {"底盘 +6.0 rad/s（高速）", 6.0, 1.0, 0.0},
    };
    for (const Case& c : cases) {
        const RunResult r = runChassisSpin(c.rate, c.joint_rate, c.pitch);
        char buf[256];

        const double expect_az = std::fabs(c.rate) * 0.01;
        std::snprintf(buf, sizeof(buf),
                      "%s：|Δchassis_azimuth|max=%.6f（期望≈%.6f；修复前每圈会出现 6.28）",
                      c.name, r.max_step_az, expect_az);
        check(r.max_step_az < expect_az + 1e-6 && r.max_step_az < 1.0, buf);

        std::snprintf(buf, sizeof(buf), "%s：与多圈真值一致（max 偏差 %.2e rad）", c.name,
                      r.max_err_az);
        check(r.max_err_az < 1e-6, buf);

        const double expect_big = std::fabs(c.rate + c.joint_rate) * 0.01;
        std::snprintf(buf, sizeof(buf), "%s：|Δbig_azimuth|max=%.6f（期望≈%.6f）同样连续",
                      c.name, r.max_step_big, expect_big);
        check(r.max_step_big < expect_big + 1e-6 && r.max_step_big < 1.0, buf);

        std::snprintf(buf, sizeof(buf),
                      "%s：chassis_euler_yaw 仍是 (−π,π] 包裹值（max 包裹差 %.4f）且 "
                      "big_azimuth ≡ chassis_azimuth + θ_b",
                      c.name, r.max_step_euler);
        check(r.euler_wrapped && r.big_identity, buf);
    }

    // 零样本：首个样本的锚定语义（输出 = 该包裹值 ⇒ 与修复前的启动数值一致）
    {
        tcbs::com::FullStrictPoseBuilder b(
            tcbs::com::FullStrictPoseBuilder::ImuLocation::ON_HEAD);
        check(std::fabs(b.strictPose().chassis_azimuth) < kPi,
              "首个（全零）样本：chassis_azimuth 仍锚定在 (−π,π] 内");
    }
}

}  // namespace

int main() {
    std::cout << "=== azimuth wrap 回归测试 ===" << std::endl;
    testAngleWrap();
    testStrictPoseUnwrap();
    std::cout << (g_fail == 0 ? "\n全部通过" : "\n有失败项") << std::endl;
    return g_fail == 0 ? 0 : 1;
}
