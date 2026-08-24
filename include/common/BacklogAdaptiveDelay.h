// BacklogAdaptiveDelay.h — 队列积压自适应额外延迟（可选功能，配置开关）
//
// 用途：当流水线处理能力跟不上取帧节奏时，各缓冲队列会持续积压，队列满后新帧
// 直接丢弃。本类根据每次 tryPopFrame() 返回的各缓冲队列积压情况，自适应调整
// 取帧线程（输入线程）的额外延迟，让流水线有时间消化积压：
//   - 除输出缓冲队列外各队列积压总数 non_output_backlog > increaseThreshold 时，
//     每帧将额外延迟增加 stepSeconds（上限 maxExtraDelaySeconds）；
//   - non_output_backlog < decreaseThreshold 时，每帧将额外延迟减少 stepSeconds
//     （下限 0）；
//   - 介于两阈值之间（滞回带）时保持不变。
//
// 线程安全：update() 由处理线程（tryPopFrame 之后）调用，extraDelaySeconds() 由
// 取帧线程（输入线程）调用；当前额外延迟以原子变量（微秒）保存，跨线程读写安全。
#ifndef BACKLOG_ADAPTIVE_DELAY_H
#define BACKLOG_ADAPTIVE_DELAY_H

#include <atomic>
#include <cstdint>

class BacklogAdaptiveDelay {
public:
    // 全部可设置参数（由 config/common.backlog_adaptive_delay 提供，main 组装）。
    // ⚠ 给后续修改者：本结构体所有字段均无默认值，必须显式构造 Config 传入
    //   （缺失即编译错误）；数值合法性校验在 RobotConfig::load 完成
    //   （src/common/RobotConfig.cpp），本类直接信任配置。
    struct Config {
        bool   enabled;              // 功能总开关
        int    increaseThreshold;    // 积压总数 > 该值 → 增加额外延迟
        int    decreaseThreshold;    // 积压总数 < 该值 → 减少额外延迟
        double maxExtraDelaySeconds; // 额外延迟上限（秒）
        double stepSeconds;          // 每帧额外延迟增减步长（秒）
    };

    explicit BacklogAdaptiveDelay(const Config& config);

    // 功能是否开启（main 据此决定是否调用 update）
    bool enabled() const { return cfg_.enabled; }

    // 处理线程在每次 tryPopFrame() 后调用；non_output_backlog 为除输出缓冲队列外
    // 各缓冲队列积压个数之和（input + inter0..inter3）。功能关闭时为空操作。
    void update(int non_output_backlog);

    // 取帧线程读取当前应增加的额外延迟（秒；功能关闭或尚未累积时为 0）
    double extraDelaySeconds() const;

private:
    Config cfg_;
    // 当前额外延迟（微秒）；update() 写入、extraDelaySeconds() 读取，跨线程安全
    std::atomic<std::int64_t> extra_delay_us_{0};
};

#endif // BACKLOG_ADAPTIVE_DELAY_H
