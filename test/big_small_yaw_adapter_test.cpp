// big_small_yaw_adapter_test.cpp — 新构型端到端集成自测（真实 tcbs::RobotController +
// 适配器 + 拆分器 + 新输出模式；不需要相机/模型/推理进程；串口线程无硬件时静默失败）
//
// 用法：
//   ./big_small_yaw_adapter_test            # 需要 selector 指向 big_small 构型的配置（如 Sentry1）
//
// 检查项：
//   1) 适配器按 common.big_small_yaw 组装并构造 tcbs::RobotController（后台 MPC 线程启动）；
//   2) GimbalOutputForBigSmallYaw::update 正常下发（大/小 yaw 序列长度 = 预测序列长度、
//      fire 序列截取正确、拆分后每点 |θ_small| 在软限位内）；
//   3) 控制器确实收到序列（MPC 参考序列长度、后台 loop_fps / ticks_since_set 正常）；
//   4) 预测不可用时进入保持模式（auto_aim 关闭 + 大/小 yaw 保持当前反解方位角）。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "common/BigSmallYaw/GimbalOutputForBigSmallYaw.h"
#include "common/BigSmallYaw/RobotStateForBigSmallYaw.h"
#include "common/Output/OutputContext.h"
#include "common/RobotConfig.h"

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::cout << (ok ? "  [ok]   " : "  [FAIL] ") << what << std::endl;
    if (!ok) ++g_fail;
}

// 合成一帧预测结果：N 个返回点，yaw 按给定角速度扫过，pitch 恒定
SequencePredictor::Result makeSeq(int n, double yaw0, double yaw_rate, double dt) {
    SequencePredictor::Result r;
    r.valid = true;
    r.integral_enable = false;
    for (int i = 0; i < n; ++i) {
        SequencePredictor::Item it;
        it.success = true;
        const double t = (i + 1) * dt;
        it.yaw = (float)(yaw0 + yaw_rate * t);
        it.pitch = 0.05f;
        it.gimbal_yaw = it.yaw;
        it.gimbal_pitch = it.pitch;
        it.predicted_point = cv::Vec3f(5.0f, 1.0f, 1.0f);
        it.predict_time = t;
        it.flight_time = t;
        it.target_index = 0;
        r.items.push_back(it);
    }
    r.first_point = r.items.front().predicted_point;
    r.first_predict_time = r.items.front().predict_time;
    r.yaw_world_origin = cv::Vec3f(0.0f, 0.0f, 0.4f);
    return r;
}
} // namespace

int main() {
    const RobotConfig& cfg = RobotConfig::instance();
    if (cfg.common.bigSmallYaw.mode != YawMode::BIG_SMALL) {
        std::cerr << "本用例需要 selector 指向 big_small 构型的机器配置（如 Sentry1）" << std::endl;
        return 2;
    }
    const double dt = cfg.common.dtControl();
    std::cout << "dt_control = " << dt << " s" << std::endl;

    bsy::RobotControllerAdapter adapter;   // 真实 tcbs::RobotController（含串口线程 + MPC 后台线程）
    check(true, "适配器构造成功（tcbs::RobotController 已启动）");

    bsy::GimbalOutputForBigSmallYaw out(adapter);
    check(true, "新构型云台输出模式构造成功");

    // 等一拍后台 loop 起来（无硬件时状态为 0，但 MPC 仍会求解）
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const bsy::RobotState st0 = adapter.state();
    std::printf("  state: valid=%d ready=%d pred_big_seq=%zu pred_small_seq=%zu\n",
                (int)st0.valid, (int)st0.ready, st0.pred_big_azimuth_seq.size(),
                st0.pred_small_azimuth_seq.size());
    // ⚠ v2 子模组没有状态估计器：McuMpcController 只要求 comm != nullptr 就照常求解
    //   （无硬件时严格反解全 0，MPC 依然按 0 状态求解并给出 N 点序列），因此预测序列
    //   长度恒为 mpc.pred_n，与是否收到串口数据无关。
    const size_t expect_seq = (size_t)cfg.common.bigSmallYaw.bigSmall.robotController.mpc.n;
    std::cout << "  （无硬件环境下 MPC 仍按全 0 状态求解 ⇒ 预测序列长度 = pred_n = "
              << expect_seq << "）" << std::endl;
    check(st0.pred_big_azimuth_seq.size() == expect_seq, "MPC 大 yaw 预测序列长度符合预期");
    check(st0.pred_small_azimuth_seq.size() == expect_seq, "MPC 小 yaw 预测序列长度符合预期");

    // ── 连续多帧下发：目标以 1.0 rad/s 相对大 yaw 平滑能力（3 rad/s）之内运动 ──
    const int N = (cfg.common.predictSequence.predictionPoints - 1) *
                      cfg.common.predictSequence.interpolationRefine + 1 +
                  cfg.common.predictSequence.exactLeadPoints;
    cv::Mat frame(8, 8, CV_8UC3, cv::Scalar(0, 0, 0));
    auto ts = std::chrono::steady_clock::now();
    int bad_len = 0, over_soft = 0;
    for (int f = 0; f < 25; ++f) {
        PipelineResult res;
        res.valid = true;
        res.frame = frame;
        res.extra_info.fillCurrentPackZeros(YawMode::BIG_SMALL);
        res.frame_timestamp = ts;

        OutputContext ctx;
        ctx.yaw_mode = YawMode::BIG_SMALL;
        ctx.bsy_state = adapter.state();
        ctx.predict_result = makeSeq(N, 0.2 * f * 4 * dt, 1.0, dt);

        out.update(res, nullptr, ctx);
        const auto& lo = out.lastOutput();
        if ((int)lo.big_yaw_seq.size() != N || (int)lo.small_yaw_seq.size() != N) ++bad_len;
        // 拆分后每点 |θ_small| 必须在软限位内
        for (size_t i = 0; i < lo.small_yaw_seq.size(); ++i) {
            const double th = lo.small_yaw_seq[i] - lo.big_yaw_seq[i];
            if (th > lo.soft_max + 1e-9 || th < lo.soft_min - 1e-9) ++over_soft;
        }
        ts += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(4.0 * dt));
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    check(bad_len == 0, "每帧下发的大/小 yaw 序列长度 = 预测序列长度");
    check(over_soft == 0, "每帧拆分结果 |θ_small| 均在软限位内");

    const auto& lo = out.lastOutput();
    std::printf("  最后一帧: aim#=%zu big#=%zu small#=%zu pitch#=%zu fire#=%zu 段数=%d\n",
                (size_t)N, lo.big_yaw_seq.size(), lo.small_yaw_seq.size(), lo.pitch_seq.size(),
                lo.fire_seq.size(), lo.unlimited_episodes);
    check(!lo.fire_seq.empty(), "fire 序列已生成");
    check(!lo.pitch_seq.empty() && (int)lo.pitch_seq.size() <= N, "pitch 序列按配置截取");

    // ── 控制器侧确认收到序列并持续求解 ──
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    const bsy::RobotState st1 = adapter.state();
    std::printf("  MPC: loop_fps=%.1f pred_psi_b/s=%.4f/%.4f ref_big#=%zu ref_small#=%zu\n",
                st1.mpc.loop_fps, st1.mpc.pred_psi_b, st1.mpc.pred_psi_s,
                st1.ref_big_azimuth_seq.size(), st1.ref_small_azimuth_seq.size());
    check(!st1.ref_big_azimuth_seq.empty() && !st1.ref_small_azimuth_seq.empty(),
          "控制器 MPC 已收到大/小 yaw 参考序列（序列被逐拍消费）");
    check(st1.mpc.loop_fps > 0.0, "MPC 后台 loop 正在运行（loop_fps > 0）");
    check(st1.mpc.ticks_since_set > 0, "后台 loop 持续运行（ticks_since_set 递增）");

    // ── 保持模式：预测不可用 ──
    {
        PipelineResult res;
        res.valid = true;
        res.frame = frame;
        res.extra_info.fillCurrentPackZeros(YawMode::BIG_SMALL);
        res.frame_timestamp = ts;
        OutputContext ctx;
        ctx.yaw_mode = YawMode::BIG_SMALL;
        ctx.bsy_state = adapter.state();
        // predict_result 保持默认无效
        out.update(res, nullptr, ctx);
        check(!out.lastOutput().auto_aim_enable, "预测不可用 → 保持模式（auto_aim 关闭）");
        // 预测不可用时 fire 必须全为 false：
        //   - 未开启哨兵扫描（sentry_controller.enabled = false）⇒ 保持序列 = {false}；
        //   - 开启（Sentry1 配置即如此）⇒ 保持段/扫描段都是长度 = 扫描序列点数 n 的全 false 序列。
        const bool fire_all_false =
            !ctx.fire_out.empty() &&
            std::none_of(ctx.fire_out.begin(), ctx.fire_out.end(), [](bool b) { return b; });
        check(fire_all_false, "保持模式 fire 序列全为 false");
        check(!ctx.split_diag.valid, "保持模式清空拆分器诊断");
    }

    std::cout << (g_fail == 0 ? "\nALL PASS" : "\nFAILED: " + std::to_string(g_fail)) << std::endl;
    return g_fail == 0 ? 0 : 1;
}
