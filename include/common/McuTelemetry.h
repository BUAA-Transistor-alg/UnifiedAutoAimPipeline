// McuTelemetry.h — 电控 MCU 遥测最新值（弹速 / 颜色）
//
// 用途：main 中维护一个 McuTelemetry 实例，由专门的遥测线程周期从控制器
// （单 yaw：tcs::RobotController；大小 yaw：bsy::RobotControllerAdapter）读取
// 弹速与颜色并写入；其余消费者不再各自 getState() 读这两个字段：
//   - 弹道解算：main 弹道线程把 load() 的弹速传给 SequencePredictor::predict
//     （SequencePredictor 不再从状态包读弹速，见其 predict 的 bullet_velocity 参数）；
//   - 可视化覆盖层**不**经过本结构体：drawOverlay 仍直接读控制器状态显示原始弹速 /
//     颜色（诊断用，不受本段的初始值 / 限位 / 低通影响）。
//
// ★ 取值合法性约定（**不使用 valid 标志**）：
//   - 本结构体**任何时刻都持有合法值**：构造时必须给出合法初始值，来自机器配置
//     文件顶层独立段 mcu_telemetry（bullet_velocity.initial / color.initial，
//     见 RobotConfig::McuTelemetryParams）；
//   - 写入方（main 遥测线程）**只写有效数据**：控制器报告 MCU 数据有效、且该值的
//     update = true 时才写入；未初始化 / 非法值 / update = false 一律不写，
//     保留上一次的合法值（update = false ⇒ 始终为初始值）；
//   - 弹速的实测更新走 updateBulletVelocityLimitedFiltered()：先限位到配置的
//     [min, max]，再按一阶低通系数与当前值融合（详见该函数注释）；颜色无限位/滤波，
//     直接写入；
//   - 因此消费方可以无条件使用 load() 的值，无需再判断“是否有效”。
//
// 其他语义：
//   - 两个值**彼此独立**，允许来自不同时刻：不要求与帧严格时序对齐，取最新值即可；
//   - 本结构只承载“最新值”，不做队列/时间戳/帧对齐；确有对齐需求时应在消费侧
//     自行缓存（当前设计刻意不引入）。
#ifndef MCU_TELEMETRY_H
#define MCU_TELEMETRY_H

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>

/// 电控 MCU 遥测最新值（线程安全：单写者周期写、多读者随时读）
struct McuTelemetry {
    /// 一致性快照（各字段可能来自不同时刻，见文件头说明）
    struct Snapshot {
        float   bullet_velocity = 0.0f;   // 弹速（m/s），恒为合法值
        uint8_t color = 0;                // 电控颜色标志位（原值透传，语义见电控协议）
    };

    /// @param initial_bullet_velocity 合法初始弹速（m/s；config
    ///                                mcu_telemetry.bullet_velocity.initial，须落在
    ///                                配置的 [min, max] 内，由 RobotConfig 校验）
    /// @param initial_color           合法初始颜色（config mcu_telemetry.color.initial，0~255）
    McuTelemetry(float initial_bullet_velocity, uint8_t initial_color)
        : bullet_velocity_(initial_bullet_velocity), color_(initial_color) {}

    // 写入方约定：只写有效值（未初始化 / 非法值不写，见文件头说明）
    void setBulletVelocity(float v) {
        bullet_velocity_.store(v, std::memory_order_relaxed);
    }

    /// 用控制器实测弹速更新（**先限位、再一阶低通滤波**）：
    ///
    ///   v_clamped = clamp(raw, min_velocity, max_velocity)
    ///   v_new     = α·v_clamped + (1-α)·v_prev      （v_prev = 结构体当前弹速）
    ///
    /// 滤波状态就是结构体当前的弹速值（构造时 = initial），不另设状态。
    ///
    /// @param raw          控制器实测弹速（m/s）；**NaN / +Inf / -Inf 时直接跳过本次
    ///                     更新**（保留当前值，不调用方预先判断）
    /// @param min_velocity 限位下限（m/s）
    /// @param max_velocity 限位上限（m/s，须 > min_velocity；由 RobotConfig 校验）
    /// @param alpha        一阶低通系数 ∈ (0, 1]；α = 1 等价于不滤波（直接用限位后值）
    ///
    /// 约束：本函数是读-改-写，只允许单写者（main 遥测线程）调用。
    void updateBulletVelocityLimitedFiltered(double raw, double min_velocity,
                                             double max_velocity, double alpha) {
        // 非有限输入（NaN / +Inf / -Inf）直接跳过本次更新，保留当前弹速值
        if (!std::isfinite(raw)) return;
        const float clamped =
            static_cast<float>(std::clamp(raw, min_velocity, max_velocity));
        const float prev = bullet_velocity_.load(std::memory_order_relaxed);
        const float a    = static_cast<float>(alpha);
        bullet_velocity_.store(a * clamped + (1.0f - a) * prev,
                               std::memory_order_relaxed);
    }
    void setColor(uint8_t color) {
        color_.store(color, std::memory_order_relaxed);
    }

    Snapshot load() const {
        Snapshot s;
        s.bullet_velocity = bullet_velocity_.load(std::memory_order_relaxed);
        s.color           = color_.load(std::memory_order_relaxed);
        return s;
    }

private:
    // 只做“最新值”的原子读写，不承担跨字段同步职责
    std::atomic<float>   bullet_velocity_;
    std::atomic<uint8_t> color_;
};

#endif // MCU_TELEMETRY_H
