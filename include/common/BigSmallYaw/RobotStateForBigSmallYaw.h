// RobotStateForBigSmallYaw.h — 新控制器（TorqueControllerForBigSmallYaw 子模组，tcbs::）
// 的适配器：状态包 + 控制器封装（**本项目里唯一直接接触 tcbs 类型的地方之一**）
//
// 设计约定（与 ExtraInputInfo 的两包约定一致）：
//   - bsy::RobotState 是**新构型（大/小双 yaw）专用状态包**，字段语义与单 yaw 的
//     tcs::RobotController::State 完全不同（θ_big / θ_small 两级关节角、ψ_big / ψ_small
//     两个世界方位角），两者**绝不混用**；
//   - RobotState → ExtraInputInfo 的转换只填 big_small 包（single 包保持 NaN），
//     这样任何按单 yaw 语义读取该帧信息的代码会立刻因 NaN 报错；
//   - 大/小 yaw 的**角度约定**在本文件统一说明（避免“世界方位角 / 关节角 / 电机角”混淆）：
//       θ_big        ：大 yaw 关节角（相对底盘，多圈连续，IMU + 编码器估计）
//       θ_small      ：小 yaw 关节角（相对大 yaw，编码器直测，行程 −25°~+20°）
//       ψ_big        ：大 yaw 平台 x 轴的世界方位角 = ψ_chassis + θ_big（IMU 直测）
//       ψ_small      ：小 yaw 输出 x 轴的世界方位角 = ψ_big + θ_small
//     输出/参考序列一律用**世界方位角**（tcbs 接口语义）；流水线内部的 item.yaw 也是
//     世界方位角，但其中的底盘修正项按严格反解的欧拉 yaw 计算（见 SequencePredictor）。
#ifndef BSY_ROBOT_STATE_FOR_BIG_SMALL_YAW_H
#define BSY_ROBOT_STATE_FOR_BIG_SMALL_YAW_H

#include <cstdint>
#include <memory>
#include <vector>

#include "common/Input/IInputMode.h"
#include "common/RobotConfig.h"
#include "tcbs/RobotController.h"

namespace bsy {

// ── 新构型（大/小双 yaw）状态包（tcbs::RobotController::getState() 的适配子集 + 序列）──
struct RobotState {
    bool   valid = false;             // 是否收到过有效 MCU 数据
    bool   estimator_valid = false;   // 状态估计是否已就绪

    // 底盘姿态（严格反解，ZXY 欧拉角）；chassis_* 与 ExtraInputInfo 共用语义
    double info_chassis_yaw = 0.0, info_chassis_pitch = 0.0, info_chassis_roll = 0.0;
    // 底盘世界坐标（米）：新子模组只有姿态反解、**没有平动/绝对位置观测量**
    // （无 GNSS/UWB），因此恒为 0 = 世界原点取在底盘上（与相机输入模式一致）；
    // 若将来接入里程计/定位，只在本适配器里填这三个字段即可，流水线无需改动。
    double chassis_x = 0.0, chassis_y = 0.0, chassis_z = 0.0;

    // 关节角（可信实时量 / 延迟补偿估计）
    double yaw_big_joint = 0.0;       // θ_big（相对底盘；延迟补偿后的估计）
    double yaw_small_joint = 0.0;     // θ_small（相对大 yaw）
    double pitch_joint = 0.0;         // pitch 关节角
    double yaw_big_rate = 0.0;        // dθ_big/dt
    double yaw_small_rate = 0.0;      // dθ_small/dt

    // 世界方位角（估计器语义，多圈连续）
    double yaw_big_azimuth = 0.0;     // ψ_big（平台 x 轴世界方位角）
    double yaw_small_azimuth = 0.0;   // ψ_small（小 yaw 输出 x 轴世界方位角）
    double chassis_azimuth = 0.0;     // ψ_chassis
    double platform_rate = 0.0;       // ψ_big 角速度（rad/s）

    // IMU 原始欧拉角 + 安装位置（记录/诊断用）
    int    imu_location = 0;          // 0 = ON_BIG_YAW；1 = ON_HEAD
    double imu_euler_yaw = 0.0, imu_euler_pitch = 0.0, imu_euler_roll = 0.0;

    // MCU
    double  bullet_velocity = 0.0;
    uint8_t auto_aim_switch = 0;

    // MPC 输出（世界方位角序列，步长 = dt_control）
    std::vector<double> pred_big_azimuth_seq;    // 大 yaw **预测能达到**的方位角序列（长度 N）
    std::vector<double> pred_small_azimuth_seq;  // 小 yaw 预测方位角序列
    std::vector<double> ref_big_azimuth_seq;     // 本拍使用的大 yaw 参考序列
    std::vector<double> ref_small_azimuth_seq;   // 本拍使用的小 yaw 参考序列
    bool   small_ref_over_limit = false;         // 小 yaw 参考越软限位标志（供可视化）
    double torque_big = 0.0, torque_small = 0.0;
    double target_joint_big = 0.0, target_joint_small = 0.0;
    double solve_ms = 0.0, loop_fps = 0.0;
    uint32_t solve_fail_count = 0;
};

// tcbs 状态 → 本项目状态包（唯一转换点；不改变任何数值语义）
RobotState toRobotState(const tcbs::RobotController::State& st);

// 状态包 → ExtraInputInfo：填底盘位姿 + **big_small 包**（single 包保持 NaN）
ExtraInputInfo toExtraInputInfo(const RobotState& st);

// ============================================================================
// RobotControllerAdapter — tcbs::RobotController 的构造与访问封装
//
//   - 构造时按 config 的 common.big_small_yaw.big_small 分支组装
//     tcbs::RobotController::Config（tf 的小 yaw 偏移 → 模型 dx/dy；
//      robot_controller.model / mpc / estimator / mcu_linear / controller；
//      joints → MPC 行程与回中；控制周期取该分支的 dt_control），并启动后台线程；
//   - 构型必须为 big_small（否则构造即抛异常——单 yaw 构型应该用 tcs::RobotController）；
//   - state() 线程安全（内部 getState 自带锁），供弹道线程 / 输出模式 / 可视化使用；
//   - sampleExtraInfo() 供 CameraInputMode 的后台采样线程使用。
// ============================================================================
class RobotControllerAdapter {
public:
    RobotControllerAdapter();
    ~RobotControllerAdapter();

    RobotControllerAdapter(const RobotControllerAdapter&) = delete;
    RobotControllerAdapter& operator=(const RobotControllerAdapter&) = delete;

    tcbs::RobotController& controller() { return *rc_; }
    /// 本构型（big_small）的配置分支：tf / robot_controller / joints / splitter
    const RobotConfig::BigSmallYawParams::BigSmallBranch& params() const { return params_; }
    /// 控制周期（秒）= 本构型分支的 robot_controller.dt_control（= MPC 步长 = 序列间隔）
    double dtControl() const { return dt_control_; }

    RobotState state();
    ExtraInputInfo sampleExtraInfo();

private:
    // 从配置读取的完整参数（common.big_small_yaw.big_small 分支）
    RobotConfig::BigSmallYawParams::BigSmallBranch params_;
    double dt_control_ = 0.0;
    std::unique_ptr<tcbs::RobotController> rc_;
};

} // namespace bsy

#endif // BSY_ROBOT_STATE_FOR_BIG_SMALL_YAW_H
