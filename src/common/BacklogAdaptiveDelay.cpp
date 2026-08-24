// BacklogAdaptiveDelay.cpp — 队列积压自适应额外延迟的实现（见头文件说明）
//
// 控制结构（v6，PID 式 PI 直接输出 + 抗饱和，无 EMA、无速率上限）：
//   1) 误差 e = 积压总数 - targetBacklog（原始测量，不做 EMA——积分项本身
//      是天然低通，P 项增益小，见头文件说明）；
//   2) 输出额外延迟（µs）= gainP * e + integral，integral += gainI * e * dt，
//      其中 dt 为实测时间间隔（不假设固定调用频率；首帧只初始化计时基准）；
//   3) 抗饱和：输出钳位 [0, max]，积分项单独钳位 [0, max]（不 windup）。
#include "common/BacklogAdaptiveDelay.h"

#include <algorithm>
#include <chrono>

// 构造：直接信任传入的 Config。参数合法性（目标非负、非负性、增益非负）已由
// RobotConfig::load（src/common/RobotConfig.cpp）从 config.yaml 校验，此处不设
// 默认值、不做静默修正。
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

    // 1) 误差（目标积压为单个设定点）
    const double e = static_cast<double>(non_output_backlog) - cfg_.targetBacklog;

    // 2) PI：积分项（µs；gainI 单位为 秒/(单位·s)，先 ×1e6 换算为 µs/(单位·s)，
    //    × e(单位) × dt(s) = µs）
    const double max_us = cfg_.maxExtraDelaySeconds * 1e6;
    integral_us_ += cfg_.gainI * 1e6 * e * dt_s;
    // 抗饱和：积分项钳位在 [0, max]，误差反向时正常松开（无 windup）
    if (integral_us_ < 0.0) integral_us_ = 0.0;
    if (integral_us_ > max_us) integral_us_ = max_us;

    // 3) 输出 = 比例项 + 积分项，钳位 [0, max]（gainP 单位为 秒/单位，×1e6 → µs/单位）
    double us = cfg_.gainP * 1e6 * e + integral_us_;
    if (us < 0.0) us = 0.0;
    if (us > max_us) us = max_us;

    // 发布给取帧线程读取（µs，四舍五入；us 恒 >= 0）
    extra_delay_us_.store(static_cast<std::int64_t>(us + 0.5),
                          std::memory_order_relaxed);
}

double BacklogAdaptiveDelay::extraDelaySeconds() const {
    return static_cast<double>(extra_delay_us_.load(std::memory_order_relaxed)) / 1e6;
}
