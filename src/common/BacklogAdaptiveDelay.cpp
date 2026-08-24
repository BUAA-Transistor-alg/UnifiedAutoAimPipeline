// BacklogAdaptiveDelay.cpp — 队列积压自适应额外延迟的实现（见头文件说明）
//
// 控制结构（v3，限速比例积分 + EMA 平滑 + 抗饱和）：
//   1) EMA 平滑测量：抑制队列瞬时快照在阈值边缘的抖动（smoothed_backlog_ 仅
//      update() 独占访问）；
//   2) 误差 e = 平滑积压 - 设定点（滞回带中点）；
//   3) 变化速率与误差成正比但封顶（±maxRateSecPerSec），并按实测 dt 积分——
//      速率上限远小于回环延迟期间允许的调整量，从根本上消除旧版 ±step 积分器
//      的三角波极限环；
//   4) 抗饱和：延迟封顶 [0, max]，饱和期间误差同向时速率自然归零。
#include "common/BacklogAdaptiveDelay.h"

#include <algorithm>
#include <chrono>

// 构造：直接信任传入的 Config。参数合法性（阈值关系、非负性、(0,1] 范围、
// 速率上限为正）已由 RobotConfig::load（src/common/RobotConfig.cpp）从
// config.yaml 校验，此处不设默认值、不做静默修正。
BacklogAdaptiveDelay::BacklogAdaptiveDelay(const Config& config) : cfg_(config) {}

void BacklogAdaptiveDelay::update(int non_output_backlog) {
    if (!cfg_.enabled) return;

    // 实测两次 update() 的时间间隔（不假设固定调用频率；首帧只初始化计时基准）
    const auto now = std::chrono::steady_clock::now();
    if (!have_last_update_) {
        have_last_update_ = true;
        last_update_ = now;
        return;
    }
    const double dt_s = std::chrono::duration<double>(now - last_update_).count();
    last_update_ = now;
    if (dt_s <= 0.0) return;

    // 1) EMA 平滑测量
    smoothed_backlog_ =
        smoothed_backlog_ * (1.0 - cfg_.smoothingAlpha) +
        static_cast<double>(non_output_backlog) * cfg_.smoothingAlpha;

    // 2) 误差（滞回带中点为设定点；带半径用于归一化比例增益：平滑积压走到
    //    带边缘时变化速率恰好封顶）
    const double setpoint =
        (cfg_.increaseThreshold + cfg_.decreaseThreshold) / 2.0;
    const double radius =
        (cfg_.increaseThreshold - cfg_.decreaseThreshold) / 2.0;
    const double e = smoothed_backlog_ - setpoint;

    // 3) 限速比例积分：变化速率与误差成正比但封顶（µs/s），按 dt 积分
    const double rate_cap_us = cfg_.maxRateSecPerSec * 1e6;  // s/s → µs/s
    double rate = 0.0;
    if (radius > 0.0) {
        rate = (rate_cap_us / radius) * e;
        rate = std::max(-rate_cap_us, std::min(rate_cap_us, rate));
    }
    delay_us_ += rate * dt_s;

    // 4) 抗饱和：封顶在 [0, max]，超出时速率自然归零（下次 e 反向才松开）
    if (delay_us_ < 0.0) delay_us_ = 0.0;
    const double max_us = cfg_.maxExtraDelaySeconds * 1e6;
    if (delay_us_ > max_us) delay_us_ = max_us;

    // 发布给取帧线程读取（µs，四舍五入；delay_us_ 恒 >= 0）
    extra_delay_us_.store(static_cast<std::int64_t>(delay_us_ + 0.5),
                          std::memory_order_relaxed);
}

double BacklogAdaptiveDelay::extraDelaySeconds() const {
    return static_cast<double>(extra_delay_us_.load(std::memory_order_relaxed)) / 1e6;
}
