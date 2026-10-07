// RobotStateForBigSmallYaw.h — 新控制器（TorqueControllerForBigSmallYaw_v2 子模组，tcbs::）
// 的适配器：状态包 + 控制器封装（**本项目里唯一直接接触 tcbs 类型的地方之一**）
//
// 设计约定（与 ExtraInputInfo 的两包约定一致）：
//   - bsy::RobotState 是**新构型（大/小双 yaw）专用状态包**，字段语义与单 yaw 的
//     tcs::RobotController::State 完全不同（θ_big / θ_small 两级关节角、ψ_big / ψ_small
//     两个世界方位角），两者**绝不混用**；
//   - RobotState → ExtraInputInfo 的转换只填 big_small 包（single 包保持 NaN），
//     这样任何按单 yaw 语义读取该帧信息的代码会立刻因 NaN 报错；
//   - 大/小 yaw 的**角度约定**在本文件统一说明（避免“世界方位角 / 关节角 / 电机角”混淆）：
//       θ_big        ：大 yaw 关节角（相对底盘，**云台侧**）
//       θ_small      ：小 yaw 关节角（相对大 yaw，编码器直测）
//       ψ_big        ：大 yaw 平台 x 轴的世界方位角 = ψ_chassis + θ_big
//       ψ_small      ：小 yaw 输出 x 轴的世界方位角 = ψ_big + θ_small
//     输出/参考序列一律用**世界方位角**（tcbs 接口语义）；流水线内部的 item.yaw 也是
//     世界方位角，但其中的底盘修正项按严格反解的欧拉 yaw 计算（见 SequencePredictor）。
//   - ★ **来源约定**：控制链要用的姿态量统一取自子模组的严格反解包
//     （tcbs::com::FullStrictPoseBuilder::StrictPose，v2 里即 State::strict）；
//     顶层标量是它的“适配子集”（解卷绕后的多圈方位角 + 流水线要用的关节角），
//     mcu / imu / strict / mpc 四个**子模组原始分组**原样保留在 RobotState 里，
//     仅供左侧信息块显示与可视化，不参与解算与下发。
//   - v2 子模组**取消了 YawStateEstimator**（旧版的 est 分组），MPC 的关节限位 /
//     电机侧背隙等字段一并移除；本适配器不保留任何无来源的字段（不静默填 0）。
#ifndef BSY_ROBOT_STATE_FOR_BIG_SMALL_YAW_H
#define BSY_ROBOT_STATE_FOR_BIG_SMALL_YAW_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "common/Input/IInputMode.h"
#include "common/RobotConfig.h"
#include "RobotController.hpp"

namespace bsy {

// ── 新构型（大/小双 yaw）状态包 ──
//   顶层标量 = 控制链 / 流水线要用的适配子集（多圈方位角、关节角等）；
//   mcu / imu / strict / mpc = 子模组 tcbs::RobotController::State 的四个原始分组
//   （直接使用子模组类型，避免镜像结构随子模组演进而漂移）。
struct RobotState {
    bool valid = false;   // MCU 是否收到过有效数据（= mcu.valid）
    bool ready = false;   // MCU 与 IMU **都**收到过有效数据（= 子模组 control_demo 的
                          // “数据就绪”判据；v2 已无状态估计器，MPC 不依赖它就绪与否）

    // 底盘姿态（严格反解，ZXY 欧拉角）；chassis_* 与 ExtraInputInfo 共用语义
    double info_chassis_yaw = 0.0, info_chassis_pitch = 0.0, info_chassis_roll = 0.0;
    // 底盘世界坐标（米）：新子模组只有姿态反解、**没有平动/绝对位置观测量**
    // （无 GNSS/UWB），因此恒为 0 = 世界原点取在底盘上（与相机输入模式一致）；
    // 若将来接入里程计/定位，只在本适配器里填这三个字段即可，流水线无需改动。
    double chassis_x = 0.0, chassis_y = 0.0, chassis_z = 0.0;

    // 关节角（严格反解所用值；θ_big 为云台侧角，θ_small 相对大 yaw）
    double yaw_big_joint = 0.0;       // θ_big（相对底盘）
    double yaw_small_joint = 0.0;     // θ_small（相对大 yaw）
    double pitch_joint = 0.0;         // pitch 关节角

    // 世界方位角（多圈连续；strict 的 chassis_azimuth 已累计圈数解卷绕 ⇒ 这里通常
    // 已是连续值，unwrapAzimuths 退化为幂等兜底，见该函数注释）
    double yaw_big_azimuth = 0.0;     // ψ_big（平台 x 轴世界方位角）
    double yaw_small_azimuth = 0.0;   // ψ_small（小 yaw 输出 x 轴世界方位角）

    // IMU 安装构型与原始欧拉角（记录/诊断用）：
    //   imu_location 由**配置**（robot_controller.imu_location）决定，是构造实参；
    //   imu_euler_* 取 IMU 原始包（v2 的 StrictPose 不再返回 IMU 欧拉角）。
    int    imu_location = 1;          // 0 = ON_BIG_YAW；1 = ON_HEAD
    double imu_euler_yaw = 0.0, imu_euler_pitch = 0.0, imu_euler_roll = 0.0;

    // MCU 概览（细节见 mcu 分组）
    double  bullet_velocity = 0.0;
    uint8_t auto_aim_switch = 0;

    // MPC 序列（世界方位角，长度 N；供预测/火控与可视化使用）
    std::vector<double> pred_big_azimuth_seq;    // 大 yaw 预测能达到的方位角序列
    std::vector<double> pred_small_azimuth_seq;  // 小 yaw 预测方位角序列
    std::vector<double> ref_big_azimuth_seq;     // 本拍使用的大 yaw 参考序列
    std::vector<double> ref_small_azimuth_seq;   // 本拍使用的小 yaw 参考序列

    // ════════════════════════════════════════════════════════════════════
    // 子模组原始分组（tcbs::RobotController::State 的四个成员，原样保留；
    // 仅供左侧信息块显示与可视化，不参与解算与下发）
    // ════════════════════════════════════════════════════════════════════
    tcbs::RobotController::McuData  mcu;     // MCU 原始反馈（已按 mcu_linear 映射）
    tcbs::RobotController::ImuData  imu;     // IMU 原始数据
    tcbs::com::FullStrictPoseBuilder::StrictPose strict;  // 严格反解包（始终可读）
    tcbs::RobotController::MpcData  mpc;     // MPC 输出 / 参考与预测序列 / 积分补偿
};

// tcbs 状态 → 本项目状态包（唯一转换点；不改变任何数值语义）
RobotState toRobotState(const tcbs::RobotController::State& st);

// 状态包 → ExtraInputInfo：填底盘位姿 + **big_small 包**（single 包保持 NaN）
ExtraInputInfo toExtraInputInfo(const RobotState& st);

// ============================================================================
// RobotControllerAdapter — tcbs::RobotController 的构造与访问封装
//
//   - 构造时按 config 的 common.big_small_yaw.big_small 分支组装
//     tcbs::RobotController 的构造实参（tf 的小 yaw 偏移 → dm::Params 的 Dx/Dy；
//      robot_controller.model / mpc / dual_yaw_mpc / mcu_linear / imu_location /
//      dt_control / mpc_loop_period / sequence_mode），并启动后台线程；
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
    // ── 姿态来源约定（★ 非可视化消费方一律只用严格反解包）──
    //   toRobotState 把**控制链要用的姿态量**统一取自 tcbs::com::FullStrictPoseBuilder::
    //   StrictPose（= tcbs::RobotController::State::strict）：
    //     底盘欧拉角 ← strict.chassis_euler_*        （ZXY）
    //     θ_big/θ_small/θ_p ← strict.yaw_big_angle / yaw_small_angle / pitch_angle
    //     ψ_big/ψ_small ← strict.big_azimuth / small_azimuth
    //   mcu / imu 原始包只留在 RobotState 的镜像里供**可视化**显示，不参与
    //   ExtraInputInfo、变换树、弹道解算与云台下发。
    //
    //   ⚠ strict 只给 (−π,π] 的 **wrap** 方位角，而保持/扫描模式下发给子模组的
    //     MPC 参考必须与控制器内部的（多圈）方位角同圈 ⇒ 方位角在适配器内做多圈
    //     解卷绕：只以 strict 的 wrap 值为输入。
    void unwrapAzimuths(RobotState& s);

    std::mutex az_mtx_;
    bool   az_unwrap_init_ = false;
    double big_azimuth_corr_ = 0.0, small_azimuth_corr_ = 0.0;
    double big_azimuth_last_ = 0.0, small_azimuth_last_ = 0.0;

    // 从配置读取的完整参数（common.big_small_yaw.big_small 分支）
    RobotConfig::BigSmallYawParams::BigSmallBranch params_;
    int    imu_location_ = 1;   // 0 = ON_BIG_YAW；1 = ON_HEAD（构造实参同值）
    double dt_control_ = 0.0;
    std::unique_ptr<tcbs::RobotController> rc_;
};

} // namespace bsy

#endif // BSY_ROBOT_STATE_FOR_BIG_SMALL_YAW_H
