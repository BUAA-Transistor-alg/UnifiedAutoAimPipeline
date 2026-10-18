// McuTelemetry.h — 电控 MCU 遥测最新值（弹速 / 颜色）
//
// 用途：main 中维护一个 McuTelemetry 实例，由专门的遥测线程周期从控制器
// （单 yaw：tcs::RobotController；大小 yaw：bsy::RobotControllerAdapter）读取
// 弹速与颜色并写入；其余消费者不再各自 getState() 读这两个字段：
//   - 弹道解算：main 弹道线程把 load() 的弹速传给 SequencePredictor::predict
//     （SequencePredictor 不再从状态包读弹速，见其 predict 的 bullet_velocity 参数）；
//   - 可视化覆盖层：drawOverlay 显示本结构体中的值。
//
// ★ 取值合法性约定（**不使用 valid 标志**）：
//   - 本结构体**任何时刻都持有合法值**：构造时必须给出合法初始值，来自机器配置
//     文件顶层独立段 mcu_telemetry（bullet_velocity.initial / color.initial，
//     见 RobotConfig::McuTelemetryParams）；
//   - 写入方（main 遥测线程）**只写有效数据**：控制器报告 MCU 数据有效、且该值的
//     update = true 时才写入；未初始化 / 非法值 / update = false 一律不写，
//     保留上一次的合法值（update = false ⇒ 始终为初始值）；
//   - 因此消费方可以无条件使用 load() 的值，无需再判断“是否有效”。
//
// 其他语义：
//   - 两个值**彼此独立**，允许来自不同时刻：不要求与帧严格时序对齐，取最新值即可；
//   - 本结构只承载“最新值”，不做队列/时间戳/帧对齐；确有对齐需求时应在消费侧
//     自行缓存（当前设计刻意不引入）。
#ifndef MCU_TELEMETRY_H
#define MCU_TELEMETRY_H

#include <atomic>
#include <cstdint>

/// 电控 MCU 遥测最新值（线程安全：单写者周期写、多读者随时读）
struct McuTelemetry {
    /// 一致性快照（各字段可能来自不同时刻，见文件头说明）
    struct Snapshot {
        float   bullet_velocity = 0.0f;   // 弹速（m/s），恒为合法值
        uint8_t color = 0;                // 电控颜色标志位（原值透传，语义见电控协议）
    };

    /// @param initial_bullet_velocity 合法初始弹速（m/s；config
    ///                                mcu_telemetry.bullet_velocity.initial，须 > 0）
    /// @param initial_color           合法初始颜色（config mcu_telemetry.color.initial，0~255）
    McuTelemetry(float initial_bullet_velocity, uint8_t initial_color)
        : bullet_velocity_(initial_bullet_velocity), color_(initial_color) {}

    // 写入方约定：只写有效值（未初始化 / 非法值不写，见文件头说明）
    void setBulletVelocity(float v) {
        bullet_velocity_.store(v, std::memory_order_relaxed);
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
