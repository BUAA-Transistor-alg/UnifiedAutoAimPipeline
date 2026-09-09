#include "PowerRune/YAxisFilter.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <chrono>

using Clock = std::chrono::steady_clock;

static cv::Mat I3()
{
    return (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
}

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cout << "[FAIL] " << msg << std::endl; ++failures; } \
    else { std::cout << "[ok]   " << msg << std::endl; } \
} while (0)

int main()
{
    using namespace std::chrono;
    // ---- 构造：alpha_pos=1 便于一步到位观察坐标限位 ----
    YAxisFilter f(0.05f, 1.0f, 1.0f, 0.05f, 0.0f);
    cv::Vec3f center(1.0f, 2.0f, 0.5f);   // 模拟 chassis 在 world 系下的位置
    f.setPositionLimits(center, 16.0f, 5.0f);

    auto t = Clock::now();
    cv::Vec3f obs0(0.f, 0.f, 0.f);

    // 1. 首次更新（初始化）：位置正常
    auto r0 = f.update(obs0, I3(), t, true);
    CHECK(r0 == YAxisFilter::UpdateResult::NORMAL, "init returns NORMAL");
    cv::Vec3f p = f.getPosition();
    CHECK(std::fabs(p[2]) < 1e-4f, "init z unclamped (0)");
    CHECK(std::fabs(p[0]) < 1e-4f && std::fabs(p[1]) < 1e-4f, "init xy unclamped");

    // 2. 超范围有限观测（xy 距离 ~110m > 16m，z=10 > 0.5+5）：应被钳制
    t += milliseconds(33);
    cv::Vec3f obs_far(100.f, 50.f, 10.f);
    auto r1 = f.update(obs_far, I3(), t, true);
    CHECK(r1 == YAxisFilter::UpdateResult::NORMAL, "far-but-finite obs returns NORMAL (not skipped)");
    p = f.getPosition();
    float dx = p[0] - center[0], dy = p[1] - center[1];
    float xy_dist = std::sqrt(dx * dx + dy * dy);
    CHECK(xy_dist <= 16.0f + 1e-3f, "xy clamped within 16m circle");
    CHECK(std::fabs(p[2] - (center[2] + 5.0f)) < 1e-3f, "z clamped to center.z+5");
    // 方向应指向观测方向（(100,50) 相对 center 的方位）
    CHECK(dx > 10.f && dy > 0.f, "clamped point keeps observation direction");

    // 3. 输入 NaN → 跳过该输入，状态保持不变
    cv::Vec3f before = f.getPosition();
    cv::Vec3f nan_pos(0.f, std::numeric_limits<float>::quiet_NaN(), 0.f);
    auto r2 = f.update(nan_pos, I3(), t + milliseconds(10), true);
    CHECK(r2 == YAxisFilter::UpdateResult::INPUT_INVALID_SKIPPED, "NaN position -> INPUT_INVALID_SKIPPED");
    cv::Vec3f after = f.getPosition();
    CHECK(after == before, "state unchanged after NaN input");

    // 4. 输入 ±inf / 超大值 → 跳过
    cv::Vec3f inf_pos(std::numeric_limits<float>::infinity(), 0.f, 0.f);
    CHECK(f.update(inf_pos, I3(), t + milliseconds(10), true) ==
              YAxisFilter::UpdateResult::INPUT_INVALID_SKIPPED, "inf position skipped");
    cv::Vec3f huge_pos(2e6f, 0.f, 0.f);
    CHECK(f.update(huge_pos, I3(), t + milliseconds(10), true) ==
              YAxisFilter::UpdateResult::INPUT_INVALID_SKIPPED, "|v|>1e6 position skipped");

    // 5. 旋转矩阵含 NaN → 跳过
    cv::Mat badR = I3();
    badR.at<float>(0, 0) = std::numeric_limits<float>::quiet_NaN();
    CHECK(f.update(cv::Vec3f(5.f, 6.f, 0.5f), badR, t + milliseconds(10), true) ==
              YAxisFilter::UpdateResult::INPUT_INVALID_SKIPPED, "NaN rotation skipped");

    // 6. 正常观测继续工作
    auto r6 = f.update(cv::Vec3f(5.f, 6.f, 0.5f), I3(), t + milliseconds(20), true);
    CHECK(r6 == YAxisFilter::UpdateResult::NORMAL, "normal obs after skips -> NORMAL");

    // 7. 退化输入（全零旋转矩阵）→ 输出异常 → 自动重置
    cv::Mat zeroR = cv::Mat::zeros(3, 3, CV_32F);
    auto r7 = f.update(cv::Vec3f(4.f, 5.f, 0.4f), zeroR, t + milliseconds(20), true);
    std::cout << "[info] degenerate zero-rot update result = "
              << (r7 == YAxisFilter::UpdateResult::NORMAL ? "NORMAL"
                  : r7 == YAxisFilter::UpdateResult::INPUT_INVALID_SKIPPED ? "INPUT_INVALID_SKIPPED"
                  : "OUTPUT_INVALID_RESET") << std::endl;
    if (r7 == YAxisFilter::UpdateResult::OUTPUT_INVALID_RESET) {
        CHECK(true, "degenerate input -> output invalid auto-reset");
        CHECK(f.getJumpA() == 0 && std::fabs(f.getAngularVelocity()) < 1e-6f,
              "state reset (jump/omega cleared)");
        // 重置后下一帧正常观测重新初始化
        auto r8 = f.update(cv::Vec3f(4.f, 5.f, 0.4f), I3(), t + milliseconds(30), true);
        CHECK(r8 == YAxisFilter::UpdateResult::NORMAL, "re-init after auto-reset -> NORMAL");
    } else {
        std::cout << "[warn] degenerate zero-rot did not produce NaN state (defensive check not exercised)" << std::endl;
    }

    // 8. disablePositionLimits 后不再钳制
    YAxisFilter g(0.05f, 1.0f, 1.0f, 0.05f, 0.0f);   // alpha_pos=1
    g.setPositionLimits(center, 16.0f, 5.0f);
    auto tt = Clock::now();
    g.update(cv::Vec3f(0.f, 0.f, 0.f), I3(), tt, true);
    g.disablePositionLimits();
    g.update(cv::Vec3f(100.f, 0.f, 100.f), I3(), tt + milliseconds(33), true);
    cv::Vec3f qq = g.getPosition();
    CHECK(std::fabs(qq[0] - 100.f) < 1e-3f, "limits disabled -> no clamp");

    if (failures == 0) std::cout << "\nALL CHECKS PASSED" << std::endl;
    else std::cout << "\n" << failures << " CHECK(S) FAILED" << std::endl;
    return failures == 0 ? 0 : 1;
}
