// RobotStateForBigSmallYaw.cpp — 新控制器适配器实现
#include "common/BigSmallYaw/RobotStateForBigSmallYaw.h"

#include <cmath>
#include <stdexcept>

namespace bsy {

RobotState toRobotState(const tcbs::RobotController::State& st) {
    RobotState s;
    s.valid = st.mcu.valid;
    s.estimator_valid = st.est.valid;

    // ── 底盘姿态：优先用严格反解包（IMU 为准确值 → 反解底盘）──
    // 注意：strict_pose 的角度已 wrap 到 (−π, π]；chassis_euler_* 是 ZXY 欧拉角，
    // 与变换树的底盘欧拉角同一约定（流水线内部就用它同步树与算 item.yaw 的底盘修正）。
    s.info_chassis_yaw   = st.strict_pose.chassis_euler_yaw;
    s.info_chassis_pitch = st.strict_pose.chassis_euler_pitch;
    s.info_chassis_roll  = st.strict_pose.chassis_euler_roll;
    s.chassis_azimuth    = st.strict_pose.chassis_azimuth;

    // ── 关节角（θ_big 用延迟补偿后的估计；θ_small / pitch 为可信编码器值）──
    s.yaw_big_joint    = st.est.big_joint_angle;
    s.yaw_small_joint  = st.est.small_joint_angle;
    s.pitch_joint      = st.est.pitch_joint_angle;
    s.yaw_big_rate     = st.est.big_joint_rate;
    s.yaw_small_rate   = st.est.small_joint_rate;

    // ── 世界方位角（估计器语义；多圈连续）──
    s.yaw_big_azimuth   = st.est.platform_azimuth;
    s.yaw_small_azimuth = st.est.small_output_azimuth;
    s.platform_rate     = st.est.platform_rate;

    // ── IMU 原始数据（记录/诊断）──
    s.imu_location     = st.strict_pose.imu_location;
    s.imu_euler_yaw    = st.imu.valid ? st.imu.euler_yaw   : st.strict_pose.imu_euler_yaw;
    s.imu_euler_pitch  = st.imu.valid ? st.imu.euler_pitch : st.strict_pose.imu_euler_pitch;
    s.imu_euler_roll   = st.imu.valid ? st.imu.euler_roll  : st.strict_pose.imu_euler_roll;

    // ── MCU ──
    s.bullet_velocity  = st.mcu.bullet_velocity;
    s.auto_aim_switch  = st.mcu.auto_aim_switch;

    // ── 完整原始输入（左侧信息块用；与单 yaw 覆盖层各块一一对应）──
    // MCU 原始反馈
    s.mcu.valid               = st.mcu.valid;
    s.mcu.bullet_velocity     = st.mcu.bullet_velocity;
    s.mcu.pitch_angle         = st.mcu.pitch_angle;
    s.mcu.yaw_big_angle       = st.mcu.yaw_big_angle;
    s.mcu.yaw_big_omega       = st.mcu.yaw_big_omega;
    s.mcu.yaw_small_angle     = st.mcu.yaw_small_angle;
    s.mcu.yaw_small_omega     = st.mcu.yaw_small_omega;
    s.mcu.chassis_imu_yaw     = st.mcu.chassis_imu_yaw;
    s.mcu.chassis_imu_omega   = st.mcu.chassis_imu_omega;
    s.mcu.mark                = st.mcu.mark;
    s.mcu.color               = st.mcu.color;
    s.mcu.auto_aim_switch     = st.mcu.auto_aim_switch;
    s.mcu.yaw_big_temperature   = st.mcu.yaw_big_temperature;
    s.mcu.yaw_small_temperature = st.mcu.yaw_small_temperature;
    s.mcu.mcu2_seq            = st.mcu.mcu2_seq;
    // IMU 原始数据
    s.imu.valid              = st.imu.valid;
    s.imu.gx = st.imu.gx;   s.imu.gy = st.imu.gy;   s.imu.gz = st.imu.gz;
    s.imu.ax = st.imu.ax;   s.imu.ay = st.imu.ay;   s.imu.az = st.imu.az;
    s.imu.euler_yaw   = st.imu.euler_yaw;
    s.imu.euler_pitch = st.imu.euler_pitch;
    s.imu.euler_roll  = st.imu.euler_roll;
    s.imu.dt_one_tenth_ms = st.imu.dt_one_tenth_ms;
    // 状态估计（可信量 + 延迟补偿 + 反解真实位姿）
    s.est.valid               = st.est.valid;
    s.est.imu_yaw             = st.est.imu_yaw;
    s.est.imu_pitch           = st.est.imu_pitch;
    s.est.imu_roll            = st.est.imu_roll;
    s.est.platform_azimuth    = st.est.platform_azimuth;
    s.est.platform_rate       = st.est.platform_rate;
    s.est.small_joint_angle   = st.est.small_joint_angle;
    s.est.small_joint_rate    = st.est.small_joint_rate;
    s.est.pitch_joint_angle   = st.est.pitch_joint_angle;
    s.est.pitch_joint_rate    = st.est.pitch_joint_rate;
    s.est.big_joint_angle_meas= st.est.big_joint_angle_meas;
    s.est.big_joint_angle     = st.est.big_joint_angle;
    s.est.big_joint_rate      = st.est.big_joint_rate;
    s.est.big_motor_angle     = st.est.big_motor_angle;
    s.est.big_motor_rate      = st.est.big_motor_rate;
    s.est.big_platform_angle  = st.est.big_platform_angle;
    s.est.big_platform_rate   = st.est.big_platform_rate;
    s.est.backlash_center     = st.est.backlash_center;
    s.est.backlash_width_obs  = st.est.backlash_width_obs;
    s.est.big_enc_age         = st.est.big_enc_age;
    s.est.big_sample_interval = st.est.big_sample_interval;
    s.est.chassis_imu_age     = st.est.chassis_imu_age;
    s.est.big_enc_innovation  = st.est.big_enc_innovation;
    s.est.big_has_encoder     = st.est.big_has_encoder;
    s.est.head_world_yaw      = st.est.head_world_yaw;
    s.est.head_world_pitch    = st.est.head_world_pitch;
    s.est.head_world_roll     = st.est.head_world_roll;
    s.est.small_output_azimuth= st.est.small_output_azimuth;
    s.est.los_azimuth         = st.est.los_azimuth;
    s.est.los_elevation       = st.est.los_elevation;
    s.est.chassis_azimuth     = st.est.chassis_azimuth;
    s.est.chassis_yaw_rate    = st.est.chassis_yaw_rate;
    for (int i = 0; i < 3; ++i) {
        s.est.base_omega[i] = st.est.base_omega[i];
        s.est.gravity_a[i]  = st.est.gravity_a[i];
    }
    s.est.pitch_acc           = st.est.pitch_acc;
    // 严格反解包
    s.strict.imu_euler_yaw    = st.strict_pose.imu_euler_yaw;
    s.strict.imu_euler_pitch  = st.strict_pose.imu_euler_pitch;
    s.strict.imu_euler_roll   = st.strict_pose.imu_euler_roll;
    s.strict.imu_location     = st.strict_pose.imu_location;
    s.strict.big_joint_angle  = st.strict_pose.big_joint_angle;
    s.strict.small_joint_angle= st.strict_pose.small_joint_angle;
    s.strict.pitch_joint_angle= st.strict_pose.pitch_joint_angle;
    s.strict.chassis_euler_yaw   = st.strict_pose.chassis_euler_yaw;
    s.strict.chassis_euler_pitch = st.strict_pose.chassis_euler_pitch;
    s.strict.chassis_euler_roll  = st.strict_pose.chassis_euler_roll;
    s.strict.platform_azimuth = st.strict_pose.platform_azimuth;
    s.strict.chassis_azimuth  = st.strict_pose.chassis_azimuth;
    s.strict.head_azimuth     = st.strict_pose.head_azimuth;
    s.strict.recon_err_rot    = st.strict_pose.recon_err_rot;
    s.strict.big_joint_angle_age = st.strict_pose.big_joint_angle_age;

    // ── MPC：控制输出、参考与预测序列（世界方位角序列，{0}=大 yaw，{1}=小 yaw）──
    s.torque_big          = st.mpc.torque[0];
    s.torque_small        = st.mpc.torque[1];
    s.torque_mpc_big      = st.mpc.torque_mpc[0];
    s.torque_mpc_small    = st.mpc.torque_mpc[1];
    s.integral_big        = st.mpc.integral[0];
    s.integral_small      = st.mpc.integral[1];
    s.target_joint_big    = st.mpc.target_joint[0];
    s.target_joint_small  = st.mpc.target_joint[1];
    s.target_joint_rate_big   = st.mpc.target_joint_rate[0];
    s.target_joint_rate_small = st.mpc.target_joint_rate[1];
    s.ref_azimuth_big     = st.mpc.ref_azimuth[0];
    s.ref_azimuth_small   = st.mpc.ref_azimuth[1];
    s.delayed_ref_azimuth_big   = st.mpc.delayed_ref_azimuth[0];
    s.delayed_ref_azimuth_small = st.mpc.delayed_ref_azimuth[1];
    s.pred_big_azimuth_seq   = st.mpc.pred_azimuth_seq[0];
    s.pred_small_azimuth_seq = st.mpc.pred_azimuth_seq[1];
    s.ref_big_azimuth_seq    = st.mpc.ref_azimuth_seq[0];
    s.ref_small_azimuth_seq  = st.mpc.ref_azimuth_seq[1];
    s.small_ref_over_limit   = st.mpc.small_ref_over_limit;
    s.big_torque_only        = st.mpc.big_torque_only;
    s.small_torque_only      = st.mpc.small_torque_only;
    s.solve_ms               = st.mpc.solve_ms;
    s.loop_fps               = st.mpc.loop_fps;
    s.solve_count            = st.mpc.solve_count;
    s.solve_fail_count       = st.mpc.solve_fail_count;
    s.ticks_since_set        = st.mpc.ticks_since_set;
    s.sent_ok                = st.mpc.sent_ok;
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
    params_ = cfg.common.bigSmallYaw.bigSmall;   // big_small 分支（tf / robot_controller / joints / splitter）
    dt_control_ = params_.robotController.dtControl;

    tcbs::RobotController::Config c;
    const auto& rc = params_.robotController;
    // ── 双级 yaw 平面模型（dx/dy 与 tf 的小 yaw 轴偏移共用一份配置）──
    c.model.dx             = params_.tf.smallYawOffsetX;
    c.model.dy             = params_.tf.smallYawOffsetY;
    c.model.gravity        = rc.model.gravity;
    c.model.m_u_known      = rc.model.mUKnown;
    c.model.Jbig_eff       = rc.model.JbigEff;
    c.model.Js             = rc.model.Js;
    c.model.Px             = rc.model.Px;
    c.model.Py             = rc.model.Py;
    c.model.Pbx            = rc.model.Pbx;
    c.model.Pby            = rc.model.Pby;
    c.model.fcBig          = rc.model.fcBig;
    c.model.fvBig          = rc.model.fvBig;
    c.model.fcSmall        = rc.model.fcSmall;
    c.model.fvSmall        = rc.model.fvSmall;
    c.model.frictionLambda = rc.model.frictionLambda;
    c.model.tau_offset_big   = rc.model.tauOffsetBig;
    c.model.tau_offset_small = rc.model.tauOffsetSmall;
    // 大 yaw 传动背隙（3-DOF 模型；β 由子模组估计器在线给出，不在此配置）
    c.model.backlash_delta      = rc.model.backlashDelta;
    c.model.backlash_k          = rc.model.backlashK;
    c.model.backlash_c          = rc.model.backlashC;
    c.model.backlash_smooth_eps = rc.model.backlashSmoothEps;
    c.model.backlash_through    = rc.model.backlashThrough;
    c.model.Jmotor              = rc.model.Jmotor;
    c.model.fcMotor             = rc.model.fcMotor;
    c.model.fvMotor             = rc.model.fvMotor;
    c.model.tau_offset_motor    = rc.model.tauOffsetMotor;

    // ── MPC（控制周期 = 本构型分支的 robot_controller.dt_control，同时是流水线序列间隔）──
    c.mpc.dt_control            = dt_control_;
    c.mpc.N                     = rc.mpc.n;
    c.mpc.substeps              = rc.mpc.substeps;
    c.mpc.use_rk4               = rc.mpc.useRk4;
    c.mpc.max_iter              = rc.mpc.maxIter;
    c.mpc.w_big_azimuth         = rc.mpc.wBigAzimuth;
    c.mpc.w_small_azimuth       = rc.mpc.wSmallAzimuth;
    c.mpc.w_small_center        = rc.mpc.wSmallCenter;
    c.mpc.w_small_limit         = rc.mpc.wSmallLimit;
    c.mpc.small_limit_soft_ratio = rc.mpc.smallLimitSoftRatio;
    c.mpc.r_big_torque          = rc.mpc.rBigTorque;
    c.mpc.r_small_torque        = rc.mpc.rSmallTorque;
    c.mpc.rd_big_rate           = rc.mpc.rdBigRate;
    c.mpc.rd_small_rate         = rc.mpc.rdSmallRate;
    c.mpc.smooth_eps            = rc.mpc.smoothEps;
    c.mpc.ref_delay_steps       = rc.mpc.refDelaySteps;
    c.mpc.small_center_angle    = params_.joints.smallCenterAngle;
    c.mpc.big.max_torque        = rc.mpc.bigMaxTorque;
    c.mpc.big.max_torque_rate   = rc.mpc.bigMaxTorqueRate;
    c.mpc.big.min_angle         = params_.joints.bigMinAngle;
    c.mpc.big.max_angle         = params_.joints.bigMaxAngle;
    c.mpc.small.max_torque      = rc.mpc.smallMaxTorque;
    c.mpc.small.max_torque_rate = rc.mpc.smallMaxTorqueRate;
    c.mpc.small.min_angle       = params_.joints.smallMinAngle;
    c.mpc.small.max_angle       = params_.joints.smallMaxAngle;

    // ── 状态估计 ──
    c.estimator.imu_location = (rc.estimator.imuLocation == 0)
                                   ? tcbs::YawStateEstimator::Config::ImuLocation::ON_BIG_YAW
                                   : tcbs::YawStateEstimator::Config::ImuLocation::ON_HEAD;
    c.estimator.mount_yaw        = rc.estimator.mountYaw;
    c.estimator.mount_pitch      = rc.estimator.mountPitch;
    c.estimator.mount_roll       = rc.estimator.mountRoll;
    c.estimator.head_mount_yaw   = rc.estimator.headMountYaw;
    c.estimator.head_mount_pitch = rc.estimator.headMountPitch;
    c.estimator.head_mount_roll  = rc.estimator.headMountRoll;
    c.estimator.transport_delay_s    = rc.estimator.transportDelayS;
    c.estimator.big_enc_max_jump     = rc.estimator.bigEncMaxJump;
    c.estimator.stale_age_s          = rc.estimator.staleAgeS;
    c.estimator.chassis_imu_timeout_s= rc.estimator.chassisImuTimeoutS;
    c.estimator.max_extrap_s         = rc.estimator.maxExtrapS;
    c.estimator.small_rate_lpf_alpha = rc.estimator.smallRateLpfAlpha;
    c.estimator.big_rate_lpf_alpha   = rc.estimator.bigRateLpfAlpha;
    c.estimator.big_motor_rate_tau_s = rc.estimator.bigMotorRateTauS;
    c.estimator.big_motor_rate_alpha = rc.estimator.bigMotorRateAlpha;
    c.estimator.backlash_center_tau_s= rc.estimator.backlashCenterTauS;
    c.estimator.pitch_rate_lpf_alpha = rc.estimator.pitchRateLpfAlpha;
    c.estimator.pitch_acc_lpf_alpha  = rc.estimator.pitchAccLpfAlpha;
    c.estimator.bore[0] = rc.estimator.boreX;
    c.estimator.bore[1] = rc.estimator.boreY;
    c.estimator.bore[2] = rc.estimator.boreZ;
    c.estimator.gravity          = rc.estimator.gravity;
    c.estimator.use_chassis_imu  = rc.estimator.useChassisImu;
    c.estimator.source_timeout_s = rc.estimator.sourceTimeoutS;

    // ── MCU 数据线性映射 ──
    c.mcu_linear.send_pitch_scale  = rc.mcuLinear.sendPitchScale;
    c.mcu_linear.send_pitch_offset = rc.mcuLinear.sendPitchOffset;
    c.mcu_linear.recv_pitch_scale  = rc.mcuLinear.recvPitchScale;
    c.mcu_linear.recv_pitch_offset = rc.mcuLinear.recvPitchOffset;
    c.mcu_linear.recv_big_yaw_scale   = rc.mcuLinear.recvBigYawScale;
    c.mcu_linear.recv_big_yaw_offset  = rc.mcuLinear.recvBigYawOffset;
    c.mcu_linear.recv_big_omega_scale = rc.mcuLinear.recvBigOmegaScale;
    c.mcu_linear.send_big_yaw_scale   = rc.mcuLinear.sendBigYawScale;
    c.mcu_linear.send_big_yaw_offset  = rc.mcuLinear.sendBigYawOffset;
    c.mcu_linear.send_big_velocity_scale = rc.mcuLinear.sendBigVelocityScale;
    c.mcu_linear.send_big_torque_scale   = rc.mcuLinear.sendBigTorqueScale;
    c.mcu_linear.recv_small_yaw_scale   = rc.mcuLinear.recvSmallYawScale;
    c.mcu_linear.recv_small_yaw_offset  = rc.mcuLinear.recvSmallYawOffset;
    c.mcu_linear.recv_small_omega_scale = rc.mcuLinear.recvSmallOmegaScale;
    c.mcu_linear.send_small_yaw_scale   = rc.mcuLinear.sendSmallYawScale;
    c.mcu_linear.send_small_yaw_offset  = rc.mcuLinear.sendSmallYawOffset;
    c.mcu_linear.send_small_velocity_scale = rc.mcuLinear.sendSmallVelocityScale;
    c.mcu_linear.send_small_torque_scale   = rc.mcuLinear.sendSmallTorqueScale;

    // ── 控制器（后台 loop 周期 = 控制周期；模式位 / 积分补偿）──
    c.controller.loop_period       = dt_control_;
    c.controller.big_torque_only   = rc.controller.bigTorqueOnly;
    c.controller.small_torque_only = rc.controller.smallTorqueOnly;
    c.controller.ref_delay_steps   = rc.mpc.refDelaySteps;
    c.controller.integral_gain[0]  = rc.controller.integralGainBig;
    c.controller.integral_gain[1]  = rc.controller.integralGainSmall;
    c.controller.integral_limit[0] = rc.controller.integralLimitBig;
    c.controller.integral_limit[1] = rc.controller.integralLimitSmall;
    c.controller.integral_on_big   = rc.controller.integralOnBig;

    // 序列模式：输出模式一次下发大/小 yaw 方位角序列 + pitch/fire 序列（构造时选定）
    c.sequence_mode = true;

    rc_ = std::make_unique<tcbs::RobotController>(c);
}

RobotControllerAdapter::~RobotControllerAdapter() = default;

RobotState RobotControllerAdapter::state() {
    return toRobotState(rc_->getState());
}

ExtraInputInfo RobotControllerAdapter::sampleExtraInfo() {
    return toExtraInputInfo(state());
}

} // namespace bsy
