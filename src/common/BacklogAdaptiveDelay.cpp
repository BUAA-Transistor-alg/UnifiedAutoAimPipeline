// BacklogAdaptiveDelay.cpp — 队列积压自适应额外延迟的实现（见头文件说明）
#include "common/BacklogAdaptiveDelay.h"

// 构造：直接信任传入的 Config。参数合法性（阈值关系、非负性）已由
// RobotConfig::load（src/common/RobotConfig.cpp）从 config.yaml 校验，
// 此处不设默认值、不做静默修正。
BacklogAdaptiveDelay::BacklogAdaptiveDelay(const Config& config) : cfg_(config) {}

void BacklogAdaptiveDelay::update(int non_output_backlog) {
    if (!cfg_.enabled) return;
    const std::int64_t step_us = static_cast<std::int64_t>(cfg_.stepSeconds * 1e6);
    const std::int64_t max_us  = static_cast<std::int64_t>(cfg_.maxExtraDelaySeconds * 1e6);
    if (step_us <= 0) return;   // 步长为 0（或过小被截断为 0）：无法调节，保持现状

    std::int64_t cur = extra_delay_us_.load(std::memory_order_relaxed);

    // 积压过多 → 增加额外延迟（封顶 max_us）；积压缓解 → 减少（下限 0）；
    // 落在 [decreaseThreshold, increaseThreshold] 滞回带内 → 保持不变
    if (non_output_backlog > cfg_.increaseThreshold) {
        std::int64_t next = cur + step_us;
        if (next > max_us) next = max_us;
        extra_delay_us_.store(next, std::memory_order_relaxed);
    } else if (non_output_backlog < cfg_.decreaseThreshold) {
        std::int64_t next = cur - step_us;
        if (next < 0) next = 0;
        extra_delay_us_.store(next, std::memory_order_relaxed);
    }
}

double BacklogAdaptiveDelay::extraDelaySeconds() const {
    return static_cast<double>(extra_delay_us_.load(std::memory_order_relaxed)) / 1e6;
}
