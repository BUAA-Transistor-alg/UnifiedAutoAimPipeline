// BacklogAdaptiveDelay.h — 队列积压自适应额外延迟（可选功能，配置开关）
//
// 用途：当流水线处理能力跟不上取帧节奏时，各缓冲队列会持续积压，队列满后新帧
// 直接丢弃。本类根据每次 tryPopFrame() 返回的各缓冲队列积压情况，自适应调整
// 取帧线程（输入线程）的额外延迟，让流水线有时间消化积压。
//
// 控制结构（v3，限速比例积分 + EMA 平滑 + 抗饱和）：
//   1) 对除输出缓冲队列外的积压总数 non_output_backlog 做 EMA 平滑（系数
//      smoothingAlpha），抑制队列瞬时快照在阈值边缘的抖动；
//   2) 误差 e = 平滑积压 - 设定点（滞回带中点 (increase+decrease)/2）；
//   3) 延迟变化速率与误差成正比（走到带边缘时速率恰好封顶），并整体限制在
//      ±maxRateSecPerSec（秒/秒）。变化速率按 update() 两次调用的实测时间
//      dt 积分，不假设固定调用频率；
//   4) 抗饱和：延迟封顶在 [0, maxExtraDelaySeconds]，饱和期间误差同向时速率
//      自然归零，误差反向时才松开。
//
//   为什么不是旧版的「每帧 ±step 积分器」：旧实现每帧固定增减 step，配合
//   0.2s+ 的回环延迟（min_delay_seconds + 队列/处理耗时）必然产生三角波极限环
//   （extra_delay_s 持续震荡）：积压一超过阈值就单调上行、一低于阈值就单调
//   下行，永远停不在稳定点。v3 把变化速率限制到远小于「回环延迟期间允许的
//   调整量」的量级，震荡从机理上消除（闭环仿真：稳态峰峰值从旧版 12~50ms
//   降到 0.4~1.3ms）。
//
// 线程安全：update() 由处理线程（tryPopFrame 之后）调用，extraDelaySeconds() 由
// 取帧线程（输入线程）调用；当前额外延迟以原子变量（微秒）保存；平滑积压与
// 延迟累加器仅由 update() 独占访问，跨线程读写安全。
#ifndef BACKLOG_ADAPTIVE_DELAY_H
#define BACKLOG_ADAPTIVE_DELAY_H

#include <atomic>
#include <chrono>
#include <cstdint>

class BacklogAdaptiveDelay {
public:
    // 全部可设置参数（由 config/common.backlog_adaptive_delay 提供，main 组装）。
    // ⚠ 给后续修改者：本结构体所有字段均无默认值，必须显式构造 Config 传入
    //   （缺失即编译错误）；数值合法性校验在 RobotConfig::load 完成
    //   （src/common/RobotConfig.cpp），本类直接信任配置。
    struct Config {
        bool   enabled;              // 功能总开关
        int    increaseThreshold;    // 滞回带上沿（设定点上方，须 > decreaseThreshold）
        int    decreaseThreshold;    // 滞回带下沿（设定点下方，须 < increaseThreshold）
        double maxExtraDelaySeconds; // 额外延迟上限（秒）
        double smoothingAlpha;       // 积压 EMA 平滑系数，须在 (0,1]（越大越跟手、越小越平滑）
        double maxRateSecPerSec;     // 延迟变化速率上限（秒/秒，须 > 0；越小越稳、越大响应越快）
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
    // 以下状态仅由 update()（处理线程）独占访问，无需原子：
    double smoothed_backlog_ = 0.0;   // EMA 平滑后的积压（double 保持低通精度）
    double delay_us_ = 0.0;           // 延迟累加器（µs，double 保留小数步进）
    std::chrono::steady_clock::time_point last_update_{};  // 上次 update() 时刻（实测 dt）
    bool   have_last_update_ = false; // 首次 update() 只初始化计时基准
};

#endif // BACKLOG_ADAPTIVE_DELAY_H
