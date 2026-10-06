// mpc_angle_align_test.cpp — 「MPC 每拍把参考整圈对齐到实测方位角」的行为回归测试
//
// ★ 回归对象（与 azimuth_wrap_test 同一故障的两层修复之第 2 层）：
//   参考序列只在 McuMpcController::set()（上位机每帧一次）对齐过圈数；100Hz 的拍之间，
//   若实测 psi_b = chassis_azimuth + θ_b 因底盘跨过 ±π 而跳了 ±2π，参考还在旧的一圈，
//   代价里就出现 w_psi_b·(2π)² ≈ 39.5 的整圈残差（正常跟踪 ~4e-4）⇒ 每圈一个力矩脉冲。
//   修复：DualYawMpcController::solveWith() 每拍把整条参考整体平移整数圈对齐到当拍实测值。
//
// 本测试直接从**公开 API**（离线 step(measurement, ref, ref)）验证该性质：
//   把同一条参考序列整体加 k·2π（k = ±1, ±2, ±3, ±5，模拟"参考停在别的圈"），
//   得到的两轴力矩 / 预测 / 对齐后的参考序列必须与 k = 0 逐位相同（≤1e-12）——
//   修复前这些结果会相差几个数量级（一个近乎饱和、一个几乎不动）。
//   同时断言"未对齐时残差确实是整圈"，把这个故障的触发条件固化在测试里。
//
// 编译运行（需要子模组已构建出 libtcbs.so；不需要相机/模型/串口）：
//   g++ -std=c++17 -O2 -I sub_module/TorqueControllerForBigSmallYaw_v2/include
//       -I .../include/mpc -I .../include/dm -I .../include/com -I .../include/common
//       -I /usr/include/eigen3 test/mpc_angle_align_test.cpp
//       -o /tmp/mpc_angle_align_test -L build -ltcbs -lceres -lglog -ludev -lpthread
//   LD_LIBRARY_PATH=build /tmp/mpc_angle_align_test
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "dual_yaw_mpc_controller.hpp"
#include "mpc/angle_wrap.hpp"

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "  [ok]   " : "  [FAIL] ") << what << std::endl;
    if (!ok) ++g_fail;
}

constexpr double kPi    = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr int    kN     = 20;     // 与 Sentry1.yaml 的 mpc.pred_n 一致
constexpr double kDt    = 0.01;   // = dt_control

/// 动力学参数 = config/robots/Sentry1.yaml 的 robot_controller.model（辨识值）。
/// 本测试只关心"圈数"，不关心模型精度，故直接固化一份（改配置后同步即可）。
tcbs::dm::Params makeParams() {
    return tcbs::dm::Params(
        /*mb=*/1.8271, /*Ib=*/0.00237991, /*Pbx=*/0.0408972, /*Pby=*/0.0225917,
        /*ms=*/0.331926, /*Is=*/0.000344412, /*Psx=*/-0.000204337, /*Psy=*/-5.95902e-05,
        /*Dx=*/-0.299491, /*Dy=*/-0.184257, /*gx=*/0.0, /*gy=*/0.0,
        /*fbc=*/0.00226296, /*fbv=*/0.0993675,
        /*fsc=*/0.000403778, /*fsv=*/0.00145031, /*lambda=*/100.0,
        /*kb=*/4.0, /*ks=*/0.045313);
}

/// MPC 配置 = config/robots/Sentry1.yaml 的 robot_controller.mpc。
tcbs::mpc::MPCController::Options makeMpcOptions() {
    tcbs::mpc::MPCController::Options o;
    o.dt           = kDt;
    o.refinement   = 16;
    o.N            = kN;
    o.max_torque_b = 1.0;
    o.max_torque_s = 1.0;
    o.w_psi_b      = 1.0;
    o.w_psi_s      = 1.0;
    o.w_dpsi_b     = 0.01;
    o.w_dpsi_s     = 0.0;
    o.w_tau_b      = 0.0;
    o.w_tau_s      = 0.0;
    o.w_x_b        = 1.0;
    o.w_x_s        = 0.1;
    o.w_dx_b       = 10.0;
    o.w_dx_s       = 0.1;
    o.max_iter     = 16;
    o.use_gravity  = true;
    return o;
}

/// 复现日志里的工况：底盘刚跨过 +π（实测底盘方位角被包裹成 −3.10），
/// 大 yaw 关节角 θ_b 多圈（6.20），世界系扫描 2.0 rad/s。
tcbs::mpc::DualYawMpcController::Measurement makeMeas() {
    tcbs::mpc::DualYawMpcController::Measurement m;
    m.valid   = true;
    m.theta_c = -3.10;               // 包裹后的底盘方位角（刚跨过 +π）
    m.theta_b = 6.20;                // 电控给的多圈关节角
    m.theta_s = 0.0;
    m.dtheta_c = 2.0;                // 底盘自转
    m.dtheta_b = 3.8;                // 大 yaw 关节角速度（世界 2.0 + 底盘 1.8）
    m.dtheta_s = 0.0;
    m.psi_b = m.theta_c + m.theta_b; // = 3.10（世界系方位角）
    m.psi_s = m.psi_b + m.theta_s;
    m.dpsi_b = m.dtheta_c + m.dtheta_b;
    m.dpsi_s = m.dpsi_b + m.dtheta_s;
    m.gx = 0.0;
    m.gy = 0.0;
    return m;
}

/// 扫描参考序列（世界系方位角，2.0 rad/s 斜坡），整体平移 k 圈。
/// 首点比实测超前 0.2 rad —— 与实车 SCAN 时"参考略超前实测"的工况一致。
std::vector<double> makeRef(const tcbs::mpc::DualYawMpcController::Measurement& m, double k) {
    std::vector<double> ref;
    ref.reserve(kN);
    for (int i = 0; i < kN; ++i) {
        ref.push_back(m.psi_b + 0.2 + 2.0 * (i + 1) * kDt + k * kTwoPi);
    }
    return ref;
}

struct Outcome {
    double torque_b = 0.0, torque_s = 0.0;
    double pred_theta_b = 0.0, pred_dtheta_b = 0.0;
    double target_psi_b = 0.0, ref_front = 0.0;
    bool   valid = false;
};

/// 用**全新**控制器跑一次离线 step（避免热启动状态跨用例污染）。
Outcome run(double k) {
    const auto meas = makeMeas();
    auto ctrl = std::make_unique<tcbs::mpc::DualYawMpcController>(
        nullptr, makeParams(), makeMpcOptions());
    const std::vector<double> ref_b = makeRef(meas, k);
    const std::vector<double> ref_s = ref_b;   // 扫描模式下大/小 yaw 同一条序列
    const auto res = ctrl->step(meas, ref_b, ref_s, /*integral_enable_b=*/false,
                                /*integral_enable_s=*/false);
    Outcome o;
    o.valid = res.valid;
    o.torque_b = res.torque_b;
    o.torque_s = res.torque_s;
    o.pred_theta_b = res.pred_theta_b;
    o.pred_dtheta_b = res.pred_dtheta_b;
    o.target_psi_b = res.target_psi_b;
    o.ref_front = res.ref_psi_b.empty() ? 0.0 : res.ref_psi_b.front();
    return o;
}

void testWholeTurnInvariance() {
    std::cout << "\n[A] 参考整体加 k·2π 后，控制量必须逐位相同（每拍整圈对齐生效）"
              << std::endl;

    const Outcome base = run(0.0);
    check(base.valid, "k=0 求解有效（MPC 配置/参数合法）");

    // 触发条件固化：k=0 时残差只有 lead rad；k=−1 时不加对齐就是整整一圈
    const auto meas = makeMeas();
    const double lead = 0.2 + 2.0 * kDt;   // 参考首点相对实测的超前量 = 0.22 rad
    std::printf("  实测 psi_b = %.4f；k=0 首点 %.4f（残差 %.3f rad）；"
                "k=-1 首点 %.4f（残差 %.3f rad ≈ 2π）\n",
                meas.psi_b, makeRef(meas, 0.0).front(),
                std::fabs(makeRef(meas, 0.0).front() - meas.psi_b),
                makeRef(meas, -1.0).front(),
                std::fabs(makeRef(meas, -1.0).front() - meas.psi_b));
    check(std::fabs(std::fabs(makeRef(meas, -1.0).front() - meas.psi_b) - (kTwoPi - lead)) < 1e-9,
          "k=-1 的未对齐残差 = 2π − 超前量（= 修复前代价里那个 2π 残差）");
    // 代价量级：w_psi_b·r²/2 ⇒ 18.5 vs 0.02（相差约 1000 倍）
    const double cost_bad = 1.0 * std::pow(kTwoPi - lead, 2) / 2.0;
    const double cost_ok = 1.0 * lead * lead / 2.0;
    std::printf("  w_psi_b·r²/2：对齐前 %.3f，对齐后 %.5f（相差 %.0f 倍）\n", cost_bad, cost_ok,
                cost_bad / cost_ok);

    for (double k : {1.0, -1.0, 2.0, -3.0, 5.0}) {
        const Outcome o = run(k);
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "k=%+.0f：torque_b=%.9f（基准 %.9f）、torque_s=%.9f、"
                      "pred_theta_b=%.9f 全部一致",
                      k, o.torque_b, base.torque_b, o.torque_s, o.pred_theta_b);
        const bool same = o.valid == base.valid &&
                          std::fabs(o.torque_b - base.torque_b) < 1e-12 &&
                          std::fabs(o.torque_s - base.torque_s) < 1e-12 &&
                          std::fabs(o.pred_theta_b - base.pred_theta_b) < 1e-12 &&
                          std::fabs(o.pred_dtheta_b - base.pred_dtheta_b) < 1e-12;
        check(same, buf);

        // 对齐后的参考序列必须真的落在实测所在圈（这就是"每拍对齐"的直接证据）
        const double psi_ref = makeMeas().psi_b;
        std::snprintf(buf, sizeof(buf),
                      "k=%+.0f：对齐后 ref 首点 %.4f 与实测 psi_b %.4f 同圈（差 %.4f < π）", k,
                      o.ref_front, psi_ref, std::fabs(o.ref_front - psi_ref));
        check(std::fabs(o.ref_front - psi_ref) < kPi, buf);
    }

    // 跟踪偏差必须被原样保留（对齐只允许平移整数圈，不许把误差吃掉）
    check(std::fabs(base.target_psi_b - (makeMeas().psi_b + lead)) < 1e-9,
          "对齐后仍保留同样的跟踪超前量（没有把参考拖到实测上）");
    check(std::fabs(base.ref_front - (makeMeas().psi_b + lead)) < 1e-9,
          "对齐后 ref 首点 = 实测 + 0.2 + ω·dt（形状/斜率不变）");
}

}  // namespace

int main() {
    std::cout << "=== MPC 每拍整圈对齐 回归测试 ===" << std::endl;
    testWholeTurnInvariance();
    std::cout << (g_fail == 0 ? "\n全部通过" : "\n有失败项") << std::endl;
    return g_fail == 0 ? 0 : 1;
}
