// RobotStateForBigSmallYaw.cpp — 新控制器（TorqueControllerForBigSmallYaw_v2）适配器实现
#include "common/BigSmallYaw/RobotStateForBigSmallYaw.h"

#include <cmath>
#include <stdexcept>

namespace bsy {

RobotState toRobotState(const tcbs::RobotController::State& st) {
    RobotState s;
    s.valid = st.mcu.valid;
    s.ready = st.mcu.valid && st.imu.valid;   // v2 里“数据就绪”= MCU 与 IMU 都有样本

    // ── 底盘姿态：严格反解包（IMU 为准确值 → 反解底盘）──
    // strict 的角度已 wrap 到 (−π, π]；chassis_euler_* 是 ZXY 欧拉角，与变换树的底盘
    // 欧拉角同一约定（流水线内部就用它同步树与算 item.yaw 的底盘修正）。
    const tcbs::com::FullStrictPoseBuilder::StrictPose& sp = st.strict;
    s.info_chassis_yaw   = sp.chassis_euler_yaw;
    s.info_chassis_pitch = sp.chassis_euler_pitch;
    s.info_chassis_roll  = sp.chassis_euler_roll;

    // ── 关节角：★ 取自 strict（严格反解所用关节角）──
    //   θ_big 用**云台侧**角（strict.yaw_big_angle），与 chassis_euler_* 同源 ——
    //   二者配合才能让变换树复现 IMU 实测头姿态；
    //   θ_small / pitch 同样是反解所用的可信编码器值（strict 内 wrap 到 (−π,π]，
    //   两者行程均 < π，等价）。
    s.yaw_big_joint   = sp.yaw_big_angle;
    s.yaw_small_joint = sp.yaw_small_angle;
    s.pitch_joint     = sp.pitch_angle;

    // ── 世界方位角：★ 取自 strict（多圈连续；子模组解卷绕后即为连续值，
    //    unwrapAzimuths 只作幂等兜底）──
    //   ψ_big = strict.big_azimuth（大 yaw 平台 x 轴）
    //   ψ_small = strict.small_azimuth（小 yaw 输出 x 轴）
    s.yaw_big_azimuth   = sp.big_azimuth;
    s.yaw_small_azimuth = sp.small_azimuth;

    // ── IMU 欧拉角：取 IMU 原始包（v2 的 StrictPose 不再携带 IMU 欧拉角）──
    s.imu_euler_yaw   = st.imu.euler_yaw;
    s.imu_euler_pitch = st.imu.euler_pitch;
    s.imu_euler_roll  = st.imu.euler_roll;

    // ── MCU 概览 ──
    s.bullet_velocity = st.mcu.bullet_velocity;
    s.auto_aim_switch = st.mcu.auto_aim_switch;

    // ── MPC 预测 / 参考序列（世界方位角序列；供预测与火控使用）──
    s.pred_big_azimuth_seq   = st.mpc.pred_psi_b_seq;
    s.pred_small_azimuth_seq = st.mpc.pred_psi_s_seq;
    s.ref_big_azimuth_seq    = st.mpc.ref_psi_b_seq;
    s.ref_small_azimuth_seq  = st.mpc.ref_psi_s_seq;

    // ── 原始分组（左侧信息块 / 可视化用；原样透传）──
    s.mcu    = st.mcu;
    s.imu    = st.imu;
    s.strict = st.strict;
    s.mpc    = st.mpc;
    return s;
}

ExtraInputInfo toExtraInputInfo(const RobotState& st) {
    ExtraInputInfo info;
    // 共用：底盘位姿（相机模式下底盘 xyz 为 0，与单 yaw 构型一致）
    info.chassis_x     = st.chassis_x;
    info.chassis_y     = st.chassis_y;
    info.chassis_z     = st.chassis_z;
    info.chassis_yaw   = st.info_chassis_yaw;
    info.chassis_pitch = st.info_chassis_pitch;
    info.chassis_roll  = st.info_chassis_roll;
    // 构型相关的**大小 yaw 包**（single 包保持 NaN：绝不把两级关节角塞进单 yaw 语义的字段）
    info.big_small.yaw_big_pos     = st.yaw_big_joint;
    info.big_small.yaw_small_pos   = st.yaw_small_joint;
    info.big_small.pitch_angle     = st.pitch_joint;
    info.big_small.imu_euler_yaw   = st.imu_euler_yaw;
    info.big_small.imu_euler_pitch = st.imu_euler_pitch;
    info.big_small.imu_euler_roll  = st.imu_euler_roll;
    return info;
}

// ============================================================================
// RobotControllerAdapter
// ============================================================================
RobotControllerAdapter::RobotControllerAdapter() {
    const RobotConfig& cfg = RobotConfig::instance();
    if (cfg.common.bigSmallYaw.mode != YawMode::BIG_SMALL) {
        throw std::runtime_error(
            "bsy::RobotControllerAdapter: 只在 common.big_small_yaw.mode = big_small 时可用"
            "（单 yaw 构型请使用 tcs::RobotController）");
    }
    params_ = cfg.common.bigSmallYaw.bigSmall;   // tf / robot_controller / joints / splitter
    dt_control_   = params_.robotController.dtControl;
    imu_location_ = params_.robotController.imuLocation;
    const auto& rc = params_.robotController;

    // ── 双连杆动力学（tcbs::dm::Params；★ 无默认构造，逐项显式给出）──
    //   Dx/Dy 取**模型辨识值**（子模组辨识结果的“估计”列），与 tf 的机械小 yaw 偏移
    //   （smallYawOffsetX/Y，供变换树用）不是同一口径，故各自独立配置、互不替代。
    const tcbs::dm::Params model(
        /*mb=*/ rc.model.mb,   /*Ib=*/ rc.model.Ib,
        /*Pbx=*/ rc.model.Pbx, /*Pby=*/ rc.model.Pby,
        /*ms=*/ rc.model.ms,   /*Is=*/ rc.model.Is,
        /*Psx=*/ rc.model.Psx, /*Psy=*/ rc.model.Psy,
        /*Dx=*/ rc.model.Dx,   /*Dy=*/ rc.model.Dy,
        /*gx=*/ rc.model.gx,   /*gy=*/ rc.model.gy,
        /*fbc=*/ rc.model.fbc, /*fbv=*/ rc.model.fbv,
        /*fsc=*/ rc.model.fsc, /*fsv=*/ rc.model.fsv,
        /*lambda=*/ rc.model.lambda,
        /*kb=*/ rc.model.kb,   /*ks=*/ rc.model.ks);

    // ── MPC 求解器（tcbs::mpc::MPCController::Options；dt 取 dt_control）──
    //   control_demo 把全部字段都显式写出；这里同样逐项来自配置（无代码默认值）。
    tcbs::mpc::MPCController::Options mpc;
    mpc.dt           = dt_control_;          // 预测步长 = 控制周期
    mpc.refinement   = rc.mpc.refinement;    // 每控制步 RK4 子步（稳定性关键）
    mpc.N            = rc.mpc.n;             // 预测步数
    mpc.max_torque_b = rc.mpc.maxTorqueB;    // tanh 软限幅
    mpc.max_torque_s = rc.mpc.maxTorqueS;
    mpc.w_psi_b      = rc.mpc.wPsiB;         // 世界方位角跟踪
    mpc.w_psi_s      = rc.mpc.wPsiS;
    mpc.w_dpsi_b     = rc.mpc.wDpsiB;        // 世界角速度跟踪
    mpc.w_dpsi_s     = rc.mpc.wDpsiS;
    mpc.w_tau_b      = rc.mpc.wTauB;         // 力矩幅值
    mpc.w_tau_s      = rc.mpc.wTauS;
    mpc.w_x_b        = rc.mpc.wXB;           // 预 tanh 量 L2（保梯度）
    mpc.w_x_s        = rc.mpc.wXS;
    mpc.w_dx_b       = rc.mpc.wDxB;          // 预 tanh 增量 L2（代替硬限速）
    mpc.w_dx_s       = rc.mpc.wDxS;
    mpc.max_iter     = rc.mpc.maxIter;
    mpc.use_gravity  = rc.mpc.useGravity;    // 实测 gx/gy 是否真正进模型

    // ── 积分补偿（tcbs::mpc::DualYawMpcController::Options）──
    //   两轴增益各自独立；**开关**（integral_enable_b / _s）是 set() 的运行期实参，
    //   由云台输出模式每帧传入（本工程当前两轴传同一个值），不在这里配置。
    tcbs::mpc::DualYawMpcController::Options wrapper;
    wrapper.integral_gain_b = rc.dualYawMpc.integralGainB;
    wrapper.integral_gain_s = rc.dualYawMpc.integralGainS;

    // ── MCU 数据线性映射（tcbs::com::McuDataPreprocessor::LinearParams）──
    //   注意：LinearParams 的默认构造带子模组已标定值；本工程规定“不允许缺省用到的
    //   参数”，因此这里**逐项覆盖**为配置值（Sentry1.yaml 与该默认值一致）。
    tcbs::com::McuDataPreprocessor::LinearParams lin;
    lin.send_pitch_scale  = rc.mcuLinear.sendPitchScale;
    lin.send_pitch_offset = rc.mcuLinear.sendPitchOffset;
    lin.recv_pitch_scale  = rc.mcuLinear.recvPitchScale;
    lin.recv_pitch_offset = rc.mcuLinear.recvPitchOffset;
    lin.recv_big_yaw_scale   = rc.mcuLinear.recvBigYawScale;
    lin.recv_big_yaw_offset  = rc.mcuLinear.recvBigYawOffset;
    lin.recv_big_omega_scale = rc.mcuLinear.recvBigOmegaScale;
    lin.send_big_yaw_scale   = rc.mcuLinear.sendBigYawScale;
    lin.send_big_yaw_offset  = rc.mcuLinear.sendBigYawOffset;
    lin.send_big_velocity_scale = rc.mcuLinear.sendBigVelocityScale;
    lin.send_big_torque_scale   = rc.mcuLinear.sendBigTorqueScale;
    lin.recv_small_yaw_scale   = rc.mcuLinear.recvSmallYawScale;
    lin.recv_small_yaw_offset  = rc.mcuLinear.recvSmallYawOffset;
    lin.recv_small_omega_scale = rc.mcuLinear.recvSmallOmegaScale;
    lin.send_small_yaw_scale   = rc.mcuLinear.sendSmallYawScale;
    lin.send_small_yaw_offset  = rc.mcuLinear.sendSmallYawOffset;
    lin.send_small_velocity_scale = rc.mcuLinear.sendSmallVelocityScale;
    lin.send_small_torque_scale   = rc.mcuLinear.sendSmallTorqueScale;

    // ── IMU 安装构型（决定严格反解的运动学链）──
    const auto imu_loc = (rc.imuLocation == 0)
                             ? tcbs::com::FullStrictPoseBuilder::ImuLocation::ON_BIG_YAW
                             : tcbs::com::FullStrictPoseBuilder::ImuLocation::ON_HEAD;

    // ── 一体化封装：通信 + 严格反解 + 双级 yaw MPC + 后台发送线程 ──
    //   序列模式（sequence_mode = true）由配置固定：大/小 yaw 输出按序列下发。
    rc_ = std::make_unique<tcbs::RobotController>(
        imu_loc, model, mpc, wrapper, rc.mpcLoopPeriod, lin, rc.sequenceMode);
}

RobotControllerAdapter::~RobotControllerAdapter() = default;

RobotState RobotControllerAdapter::state() {
    RobotState s = toRobotState(rc_->getState());
    s.imu_location = imu_location_;   // 构造实参（配置决定），strict 不再携带
    unwrapAzimuths(s);   // 恢复/校验多圈连续（正常已是连续值 ⇒ 幂等直通）
    return s;
}

// 把 strict 的 wrap 方位角解卷绕成**多圈连续**量（原地修改 s.yaw_*_azimuth）。
// 为什么必须做：非可视化消费方（GimbalOutputForBigSmallYaw 的保持/哨兵扫描）会把这两个
// 方位角当作 MPC 参考下发；若下发 wrap 值，大 yaw 转过半圈后参考与状态会差整圈，
// MPC 会去追一个假目标。
// ★ 现状（子模组修复后）：FullStrictPoseBuilder 已把 chassis_azimuth 累计圈数解卷绕，
//   big/small_azimuth 本身就是多圈连续量，本函数因此退化为**幂等直通**（|Δ| < π，
//   corr 恒为 0）。保留它是为了：(1) 万一子模组某处仍给出 wrap 值（如未就绪的全零样本、
//   或将来回退到旧版子模组）；(2) 首个样本的锚定语义不变。**不要**因为"已经连续了"
//   就删掉——它是流水线这一侧的兜底。
// 只以 strict 的 wrap 值为输入：取与上一拍输出最近的同圈值。
// 起点一致性：适配器**拥有**这个 tcbs::RobotController，且 state() 在构造后立刻被采样
// 线程/云台线程调用；首个样本的 |wrap 值| ≤ π，即使首个样本是未就绪的全零，随后第一个
// 真实样本也只会落在同一圈 ⇒ 不会与子模组内部（多圈）方位角差整圈。
void RobotControllerAdapter::unwrapAzimuths(RobotState& s) {
    std::lock_guard<std::mutex> lock(az_mtx_);
    constexpr double kTwoPi = 2.0 * M_PI;
    auto unwrap = [](double wrapped, double& prev, double& corr) {
        double v = wrapped + corr;
        const double d = v - prev;
        if (d > M_PI)       corr -= kTwoPi;   // 跨过 +π：回退一圈
        else if (d < -M_PI) corr += kTwoPi;   // 跨过 −π：前进一圈
        v = wrapped + corr;
        prev = v;
        return v;
    };
    if (!az_unwrap_init_) {
        // 首个样本：多圈零位取该 wrap 值所在圈（与子模组内部解卷绕的起点一致）
        big_azimuth_last_   = s.yaw_big_azimuth;
        small_azimuth_last_ = s.yaw_small_azimuth;
        az_unwrap_init_     = true;
        return;
    }
    s.yaw_big_azimuth   = unwrap(s.yaw_big_azimuth, big_azimuth_last_, big_azimuth_corr_);
    s.yaw_small_azimuth = unwrap(s.yaw_small_azimuth, small_azimuth_last_, small_azimuth_corr_);
}

ExtraInputInfo RobotControllerAdapter::sampleExtraInfo() {
    return toExtraInputInfo(state());
}

} // namespace bsy
