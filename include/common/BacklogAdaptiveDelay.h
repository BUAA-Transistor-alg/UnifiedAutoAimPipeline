// BacklogAdaptiveDelay.h — 队列积压自适应额外延迟（可选功能，配置开关）
//
// 用途：当流水线处理能力跟不上取帧节奏时，各缓冲队列会持续积压，队列满后新帧
// 直接丢弃。本类根据每次 tryPopFrame() 返回的各缓冲队列积压情况，自适应调整
// 取帧线程（输入线程）的额外延迟，让流水线有时间消化积压。
//
// 控制结构（v6，PID 式 PI 直接输出 + 抗饱和，无 EMA、无速率上限）：
//   1) 误差 e = 除输出缓冲队列外各队列积压总数 - targetBacklog
//      （直接用原始测量，不再做 EMA 平滑——见下方说明）；
//   2) 输出额外延迟（微秒）为 PI：
//        extra_delay = gainP * e + integral，integral += gainI * e * dt
//      其中 gainP 为比例增益（秒/单位积压），gainI 为积分增益
//      （秒/单位积压/秒），内部换算为微秒后计算；dt 为 update() 两次调用的
//      实测时间间隔（不假设固定调用频率）。比例项提供即时校正，积分项负责把
//      延迟推到平衡点（稳态时积压收敛到 targetBacklog，误差归零）；
//   3) 抗饱和：输出钳位在 [0, maxExtraDelaySeconds]；积分项单独钳位在
//      [0, max]，饱和期间误差同向时不再累积（不 windup），误差反向时正常松开。
//
//   为什么不需要 EMA 平滑（闭环仿真，含 0.2s 回环延迟、负载 0.3~0.95）：
//   - 积分项本身是天然低通：∫gainI·e·dt 对积压噪声取平均，误差均值被积分项
//     吸收；P 项增益取小量级（≈5~10µs/单位）时，原始积压抖动经 P 项只放大
//     十几 µs，对输出几乎无影响；
//   - 有/无 EMA 的稳态峰峰值几乎相同（kp=8/ki=3 时 133~216µs vs 141~233µs，
//     差 ~5%），EMA 反而多引入一个配置项与少量相位滞后。
//   - EMA 原本是为旧版「阈值判断去抖」与「微分项降噪」服务的，两者均已移除。
//
//   其余设计取舍：微分项（D）经仿真验证为负收益（只加快负载
//   突变时的收敛，代价是稳态震荡从 ~0.2ms 恶化到 3.5~8.4ms），本项目负载
//   基本不突变、以稳态低震荡为目标，故不采用。
//
// 线程安全：update() 由处理线程（tryPopFrame 之后）调用，extraDelaySeconds() 由
// 取帧线程（输入线程）调用；当前额外延迟以原子变量（微秒）保存；积分项仅由
// update() 独占访问，跨线程读写安全。
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
        double targetBacklog;        // 目标积压（积压的设定点，须 >= 1）
        double maxExtraDelaySeconds; // 额外延迟上限（秒）
        double gainP;                // 比例增益（秒/单位积压，须 >= 0）
        double gainI;                // 积分增益（秒/单位积压/秒，须 >= 0）
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
    double integral_us_ = 0.0;        // PI 积分项（µs）
    std::chrono::steady_clock::time_point last_update_{};  // 上次 update() 时刻（实测 dt）
    bool   have_last_update_ = false; // 首次 update() 只初始化计时基准
};

#endif // BACKLOG_ADAPTIVE_DELAY_H
