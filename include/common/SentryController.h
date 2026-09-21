// SentryController.h — 哨兵扫描控制器（可选功能，config: common.sentry_controller）
//
// 用途：哨兵长时间没有瞄准目标时自动进入“扫描模式”，让云台按设定规律运动，
//       以便重新发现目标。本类只负责“无目标计时 + 扫描目标求解”，不直接接触
//       任何控制器：两个云台输出模式（GimbalOutput / GimbalOutputForBigSmallYaw）
//       各自把生成的序列交给自己的控制器。
//
// 触发判据（与云台输出模式的“保持模式”一致）：本帧**没有有效预测序列**
//   （无目标，或目标存在但弹道解算失败）即视为“未在瞄准目标”，开始计时；
//   一旦重新出现有效预测序列，立即复位计时并退出扫描。
//
// 扫描模式行为（进入扫描后）：
//   - yaw：以 yaw_scan_angular_velocity 持续旋转；序列**每个点**相对本帧实测角度的
//     偏离限幅在 ±yaw_scan_max_deviation 内（防止下发序列把枪线甩出过远）。
//     大小 yaw 构型下由输出模式生成**一条**序列并同时用于大 yaw 与小 yaw
//     （即两级目标取同一个世界方位角，不保持关节角差），因此两级同步旋转。
//   - pitch：在 [pitch_scan_min, pitch_scan_max] 之间做锯齿波往复——先由下界线性
//     升到上界（耗时 pitch_scan_rise_time_sec），再由上界线性回落到下界（耗时
//     pitch_scan_fall_time_sec），周期 = 上升 + 回落；相位零点取“进入扫描的时刻”。
//
// 进入扫描前的“保持段”（预测已无效但未超过 idle_timeout_sec）由各输出模式处理：
// 用上一个有效输出序列的首值填充整条序列（见两个 Gimbal 输出模式），本类不参与。
//
// enabled = false 时本类所有接口均为空操作（scanning() 恒 false），两个云台输出
// 模式走原有分支，行为与未引入本功能时完全一致。
#ifndef SENTRY_CONTROLLER_H
#define SENTRY_CONTROLLER_H

#include <chrono>
#include <vector>

#include "common/RobotConfig.h"

namespace sentry {

class SentryController {
public:
    using TimePoint = std::chrono::steady_clock::time_point;

    explicit SentryController(const RobotConfig::SentryControllerParams& params);

    bool enabled() const { return params_.enabled; }

    /// 本帧是否处于扫描模式（enabled = false 时恒为 false）
    bool scanning() const { return scanning_; }

    /// 每帧更新跨帧计时。云台输出模式在“无有效预测”分支调用本函数；在有效分支
    /// 同样调用（valid = true），用于复位计时并退出扫描。
    /// @param valid 本帧是否有有效预测序列（seq.valid && !items.empty()）
    /// @param now   本帧时间戳（与预测序列同一时间轴，须单调递增）
    /// 无效帧：累计“无目标”时长，首次达到 idle_timeout_sec 时进入扫描，并以该
    /// 帧时刻为扫描相位零点；已进入扫描则保持（相位零点不变）。
    void update(bool valid, const TimePoint& now);

    /// 距进入扫描的秒数（未进入扫描时为 0）
    double scanElapsed(const TimePoint& now) const;

    /// 单轴 yaw 扫描目标：current + ω·t，并限幅 |目标 − current| <= max_deviation
    double yawTargetAt(double current, double t) const;

    /// pitch 扫描目标（t = 距进入扫描的秒数）：rise / fall 两段线性锯齿波
    double pitchTargetAt(double t) const;

    /// 生成 yaw 扫描序列（n 点）：第 k 点目标角速度前进时间 =
    /// scanElapsed(now) + (k+1)·dt（即首点领先本帧一拍），再按本帧实测角度限幅。
    void buildYawSequence(int n, double dt, double current_yaw, const TimePoint& now,
                          std::vector<double>& out) const;

    /// 生成 pitch 扫描序列（n 点）：与 buildYawSequence 同一时间轴，锯齿波相位
    /// 以进入扫描的时刻为零点。
    void buildPitchSequence(int n, double dt, const TimePoint& now,
                            std::vector<double>& out) const;

private:
    RobotConfig::SentryControllerParams params_;

    // ── 跨帧计时状态（仅云台输出线程访问，单帧串行）──
    bool      idle_active_ = false;   // 是否处于一段连续“无目标”中
    TimePoint idle_since_{};          // 本段“无目标”起点
    bool      scanning_ = false;      // 是否处于扫描模式
    TimePoint scan_start_{};          // 进入扫描的时刻（pitch 锯齿波相位零点）
};

} // namespace sentry

#endif // SENTRY_CONTROLLER_H
