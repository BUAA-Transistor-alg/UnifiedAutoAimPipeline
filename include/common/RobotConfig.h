// RobotConfig.h — 全局参数配置（机器配置文件，config/robots/*.yaml）
//
// config 读取规则（v2）：
//   config/selector.yaml          —— 机器配置选择器（仅含 active_config 一个条目）
//   config/robots/<active_config>.yaml —— 当前机器的完整参数（如 Infantry1.yaml）
// RobotConfig::instance() 先读选择器再加载对应机器配置文件；切换机器只需修改
// selector.yaml 的 active_config，无需改代码或重新编译。
//
// ⚠ 重要约定（给后续修改者）：机器配置文件中的所有参数均为必填，本文件及
//   RobotConfig.cpp 中不设任何默认值/回退值——缺段或缺字段时 RobotConfig::load
//   直接抛异常退出，绝不静默采用默认值。
//   **唯一例外**：common.big_small_yaw 下按 yaw 构型分为 single / big_small 两支，
//   只需填写当前 mode 对应的那一支（另一支可整段省略——两种构型的参数互不通用，
//   写没用到的那一份纯属冗余）；当前构型那一支内部的字段依旧全部必填。
//   新增配置项时必须同步：
//   1) 在 config/robots/<active_config>.yaml 对应段中添加字段并写明含义；
//   2) 在 RobotConfig.h 对应结构体中添加成员（无默认初始化）；
//   3) 在 src/common/RobotConfig.cpp 中通过 requireScalar 等读取。
//
// 机器配置文件顶层分为三个大类：
//   - common      ：两个流水线共用的参数（tf 偏移 / 相机内参 / 弹道 / MPC / 输入控制器 /
//                    min_delay_seconds 等；其中 common.big_small_yaw 决定 yaw 构型
//                    （single = 旧单 yaw / big_small = 大/小双 yaw）并提供新构型的
//                    tcbs::RobotController 参数与大小 yaw 拆分器参数，见 YawMode）
//   - armor     ：Armor 流水线独占参数（推理模型 / 批量 / 观测丢失超时 /
//                   OutpostESEKF 与 SuperPower EKF 滤波参数）
//   - power_rune  ：PowerRune 流水线独占参数（推理模型 / NMS / 阈值 / 批量）
//
// 相机内参与畸变系数两流水线共用，但分为两套按输入模式自动选择
// （common.input_mode 下）：
//   - input_mode.camera_mode ：--input camera（实机相机，含 IP/曝光/增益/extra_info_delay）
//   - input_mode.video_mode  ：--input video / interactive（录制视频 / 交互图片）
#ifndef ROBOT_CONFIG_H
#define ROBOT_CONFIG_H

#include <cstdint>
#include <string>
#include <array>

#include <opencv2/opencv.hpp>

// yaw 构型（config: common.big_small_yaw.mode；**初始化时定型，运行中不可切换**）：
//   SINGLE    ：单 yaw（旧，TorqueController 子模组）：chassis -> yaw -> pitch -> ...
//   BIG_SMALL ：大/小双 yaw（新，TorqueControllerForBigSmallYaw 子模组）：
//               chassis -> yaw_big -> yaw_small -> pitch -> ...
// 构型决定变换树节点结构、ExtraInputInfo 使用哪一包关节角、瞄准解算几何与输出模式，
// 因此两边（树 / 输入信息 / 状态包）都按本构型只填自己那一包，另一包整体置 NaN，
// 越界调用（如单 yaw 下调 setYawBig）直接抛异常，绝不静默混用。
enum class YawMode {
    SINGLE    = 0,
    BIG_SMALL = 1,
};

class RobotConfig {
public:
    // 变换树各节点相对其父节点的固定偏移（单位：米）。
    // ⚠ 两个 yaw 构型的 tf 分支**内容不同**（见 BigSmallYawParams）：
    //   single 分支：只需 yawJointZOffset（yaw 关节）+ 其余公共偏移，
    //                smallYawOffset* 不存在（该构型没有小 yaw 轴）；
    //   big_small 分支：yawJointZOffset 表示**大 yaw**关节沿 z 的偏移，
    //                并额外需要 smallYawOffset*（小 yaw 轴相对大 yaw 轴的偏移，大 yaw 系）。
    struct TfOffsets {
        float yawJointZOffset;  // yaw 关节沿 z 轴偏移（相对 chassis；big_small 下为大 yaw 关节）
        float pitchJointYOffset;  // pitch 关节沿 y 轴偏移（相对 yaw 旋转中心）
        float imuOffsetX, imuOffsetY, imuOffsetZ;          // imu 相对 head
        float cameraOffsetX, cameraOffsetY, cameraOffsetZ;  // camera 相对 head
        float muzzleOffsetX, muzzleOffsetY, muzzleOffsetZ;  // muzzle 相对 head
        // ── 仅 big_small 分支必填：小 yaw 轴相对大 yaw 轴的偏移（**大 yaw 系**，米）──
        // 同时作为变换树里 yaw_small 节点相对 yaw_big 节点的位置与双级 yaw 平面模型的
        // 平面偏置 dx/dy（z 分量只进变换树，不进平面模型）。改机械后必须同步修改。
        float smallYawOffsetX = 0.0f;
        float smallYawOffsetY = 0.0f;
        float smallYawOffsetZ = 0.0f;
    };

    // 相机参数（分辨率 + 内参 + 畸变；相机模式额外含 IP/曝光/增益/extra_info_delay）
    // ⚠ 给后续修改者：本结构体所有字段均无默认值，必须由机器配置文件提供：
    //   - 相机模式（common.input_mode.camera_mode）必填 device_ip / net_ip /
    //     exposure / gain / extra_info_delay；
    //   - 视频/交互模式（common.input_mode.video_mode）必填 test_max_fps；
    //   - 两模式均必填 resolution / camera_matrix / dist_coeffs。
    //   缺任一字段 RobotConfig::load 直接抛异常退出，解析见 src/common/RobotConfig.cpp。
    struct CameraParams {
        std::string deviceIp;   // 相机设备 IP（相机模式必填）
        std::string netIp;      // 本机网口 IP（相机模式必填）
        float exposure;         // 曝光时间（微秒，相机模式必填）
        float gain;             // 增益（相机模式必填）
        int width, height;      // 图像分辨率（像素）
        cv::Mat cameraMatrix;       // 3x3 CV_64F 内参矩阵
        cv::Mat distCoeffs;         // Nx1 CV_64F 畸变系数
        // 相机输入模式（CameraInputMode）extra_info 延迟（秒）：
        // 后台线程持续采样 tcs::RobotController::getState()，返回给流水线的 extra_info
        // 为相对当前时刻 extra_info_delay 前的队头数据（0.0 = 最新状态）。
        // 相机模式必填（config: extra_info_delay）。
        double extraInfoDelay;
        // 测试最大帧率（视频/交互模式必填，config: test_max_fps）：开启后
        // VideoInputMode 的 getFrameDelay() 返回 0（不做按视频帧率的节流），
        // 用于测量视频输入 + 流水线的最大帧数/FPS。
        bool testMaxFps;
    };

    // 云台角度解算参数
    struct GimbalParams {
        double bulletDiameter;      // 弹丸直径（米）
        double bulletMass;          // 弹丸质量（kg）
        double bulletVelocity;      // 默认弹丸初速（m/s）
        double integrationStep;     // 弹道积分步长（秒）
        double distanceThreshold;   // 弹道最近点距离阈值（米）
        double distanceIterateThreshold;  // 迭代触发阈值（米）
        double stopZ;               // 弹道计算截止高度（world 系，米）
        float  pitchMin;            // pitch 搜索下界（弧度）
        float  pitchMax;            // pitch 搜索上界（弧度）
        float  pitchSearchStep;     // pitch 粗搜索步长（弧度）
    };

    // 预测弹道解算参数
    struct PredictedBallisticParams {
        double extraPredictTime;      // 额外预测时间（秒）
        int    maxIterations;         // 飞行时间迭代上限
        double timeErrorTolerance;    // 迭代提前停止的时间误差容差（秒）
    };

    // tcs::RobotController（TorqueController 子模组）构造参数
    struct RobotControllerParams {
        bool   sequenceMode;     // 序列输入模式
        bool   yawTorqueOnlyMode; // 仅力矩控制模式（yaw_torque_only_mode 开关）：
                                  // true 时 MCU 仅接收 yaw 力矩控制（不再发送角度/速度目标），
                                  // false 时正常发送角度/速度目标（见 McuMpcController::set）
        double dtControl;        // 控制周期（秒）
        int    mpcPredN;         // MPC 预测步数
        double J;                // yaw 轴转动惯量
        double tauC;             // 库仑摩擦
        double b;                // 粘滞摩擦系数
        double tauD;             // 常数扰动
        double maxTorque;        // 最大力矩（N·m）
        double maxTorqueRate;    // 最大力矩变化率（N·m/s）
        double Q;                // MPC 状态代价
        double R;                // MPC 控制代价
        double Rd;               // MPC 控制变化率代价
        int    maxIter;          // MPC 迭代上限
        double integralGain;     // yaw 力矩积分补偿比例系数
        double smoothEps;        // 位置跟踪误差平滑绝对值常数 a（代价 = Q*sqrt(err²+a)，
                                 // 很小的正数，子模组默认 1e-6；|err| ≫ √a 时 ≈ |err|，
                                 // err = 0 附近连续可导）
        // ── MCU 数据线性映射标定参数（tcs::McuDataPreprocessor::LinearParams，当前标定默认值）──
        double sendPitchScale;   // imu_euler_pitch → pitch_target_angle（发送）
        double sendPitchOffset;  // 发送偏移
        double recvPitchScale;   // mcu_pitch_angle → imu_euler_pitch（接收）
        double recvPitchOffset;  // 接收偏移
    };

    // 预测序列参数（序列输入模式：预测序列生成与消费）
    struct PredictSequenceParams {
        int    predictionPoints;    // 预测点数：实际精确弹道解算的点数（M）
        int    interpolationRefine; // 插值细化倍数：相邻实际计算点之间细分的返回点间隔数（K）
                                    // 原划分序列点数 = (M-1)*K + 1，间隔恰为 dt_control
        int    exactLeadPoints;     // 前导精确点数（n）：在序列最前面拼接 n 个逐点精确弹道
                                    // 解算的前导点（不使用插值；0 表示关闭，行为与原版一致）。
                                    // 返回序列 = [前导精确点] + [原划分序列]，
                                    // 总返回点数 = (M-1)*K+1+n，精确解算点共 M+n 个。
                                    // 原划分首段（紧邻窗口 n+1..n+K-1）需要外推时参考点改用
                                    // items[n-1]（须与段左端点同目标，否则复制左端点），
                                    // 其余段仍用原规则。取值范围 n >= 0
        int    pitchSeqLead;        // m：pitch 序列提前数（必须小于总返回点数 (M-1)*K+1+n）
        int    fireSeqLead;         // o：fire 序列提前数
        double pitchBias;           // pitch 轴偏置（弧度）
        double yawBias;             // yaw 轴偏置（弧度）
        double fireAngleLowerLimit; // fire 判定角度阈值下限（弧度）
        double fireAngleLength;     // fire 判定弧长（米）
        double aimStickRatio;       // 瞄准点滞回幅度系数（无单位，>=0）：SequencePredictor
                                    // 在“慢目标”帧启用瞄准点滞回时，滞回量 =
                                    // aim_stick_ratio × (t=0 全部瞄准点到预测车体中心的
                                    // 平均距离)。0 = 关闭瞄准点滞回。
        double minRotationToleranceAngle;  // 旋转容差角下限（弧度，>=0）：fast_target 帧下
                                    // 每个预测瞄准点的旋转容差角 =
                                    // max(本值, fire_angle_length / 该点的旋转半径)
                                    // （旋转半径 = 中心位置−瞄准点 的 xy 投影长度）。
                                    // 半径很小时容差角趋于无穷，用本值兜底（同时给出
                                    // “最小可打角度窗口”）。
    };

    // ══════════════════════════════════════════════════════════════════════
    // yaw 构型与**构型相关参数**（config: common.big_small_yaw）
    //
    // ⚠ 配置约定（本工程唯一的例外，见 config/robots/*.yaml 文件头）：
    //   common.big_small_yaw 下按构型分为 single / big_small **两支**，
    //   **只需填写当前 mode 那一支**（另一支可以整段省略、不必写）；
    //   当前构型那一支内的字段依旧**全部必填**（缺字段直接抛异常，不设代码默认值）；
    //   另一支若写了则同样严格校验（防止半截残留的自相矛盾配置）。
    //
    //   mode = single    → 只填 single 支：tf（单 yaw 链）+ robot_controller（tcs 子模组）
    //   mode = big_small → 只填 big_small 支：tf（含小 yaw 偏移）+ robot_controller
    //                      （tcbs 子模组：model / mpc / estimator / mcu_linear / controller）
    //                      + joints（行程/回中）+ splitter（大 yaw 平滑轨迹规划器）
    //
    // 参数来源约定（Sentry1.yaml 即按此填写）：
    //   - big_small.robot_controller.model / mpc / estimator / mcu_linear / controller
    //     与子模组的 tcbs::dual_yaw::ModelParams、tcbs::dual_yaw::DualYawMpcConfig、
    //     tcbs::YawStateEstimator::Config、tcbs::McuDataPreprocessor::LinearParams、
    //     tcbs::McuMpcController::Config 一一对应（子模组默认值见
    //     sub_module/TorqueControllerForBigSmallYaw/include/tcbs/mpc/planar_yaw_params.h 等）；
    //   - 小 yaw 轴相对大 yaw 轴的偏移只配置一处（big_small.tf.small_yaw_offset_*），
    //     同时作为变换树 yaw_small 节点位置与模型平面偏置 dx/dy；
    //   - big_small.joints 的行程 / 回中目标是 MPC 与拆分器**共用**的同一份配置，
    //     mpc.small_limit_soft_ratio 亦为二者共用的软限位比例（避免“拆分器以为能瞄准、
    //     MPC 却已进软限位”这类不一致）。
    // ══════════════════════════════════════════════════════════════════════
    struct BigSmallYawParams {
        YawMode mode;   // 构型开关（config: mode；single | big_small）

        // ── 单 yaw 构型分支（mode = single 时必填；big_small 下可整段省略）──
        struct SingleBranch {
            TfOffsets tf;                             // 变换树偏移（chassis -> yaw -> pitch -> head）
            RobotControllerParams robotController;    // tcs::RobotController 构造参数
        };
        SingleBranch single;
        bool singlePresent = false;   // 配置文件中是否写了 single 支（写了就严格校验）

        // ── 大小 yaw 构型分支（mode = big_small 时必填；single 下可整段省略）──
        struct BigSmallBranch {
            TfOffsets tf;   // 含 smallYawOffsetX/Y/Z（小 yaw 轴相对大 yaw 轴的偏移）

            // tcbs::RobotController 构造参数（**仅本构型使用**）
            struct RobotControllerParamsBS {
                bool   sequenceMode;  // 必须为 true（大/小 yaw 输出按序列下发）
                double dtControl;     // 控制周期（秒）：序列间隔 / MPC 步长 / 后台 loop 周期

                // 双级 yaw 平面模型（tcbs::dual_yaw::ModelParams）
                struct ModelParams {
                    double gravity;            // 重力加速度
                    double mUKnown;            // 上装质量（未知则 0）
                    double JbigEff;            // 大 yaw 侧惯量（含 m_u|d|²）
                    double Js;                 // 上装绕小 yaw 轴总惯量
                    double Px, Py;             // 上装一阶矩（kg·m）
                    double fcBig, fvBig;       // 大 yaw 库仑/粘滞摩擦
                    double fcSmall, fvSmall;   // 小 yaw 库仑/粘滞摩擦
                    double frictionLambda;     // 库仑摩擦软符号系数 λ
                    double tauOffsetBig;       // 可选常数负载
                    double tauOffsetSmall;
                    // 大 yaw 传动背隙（tcbs 3-DOF 模型 eomBacklash 用；2-DOF 的 eom 不受影响）
                    //   τ_t = k·[dz(Δ) + γ·Δ] + c·Δ̇,  Δ = θ_motor − θ_platform − β
                    //   ★ β（死区中心）由子模组估计器在线给出，不在此配置。
                    double backlashDelta;      // 背隙总宽度 δ（rad）
                    double backlashK;          // 接触刚度 k（N·m/rad）
                    double backlashC;          // 接触阻尼 c（N·m·s/rad）
                    double backlashSmoothEps;  // 平滑死区的过渡半宽 ε（rad）
                    double backlashThrough;    // 直通线性项 γ（死区内梯度引导；0 = 严格物理）
                    double Jmotor;             // 电机侧惯量（折算到关节侧，kg·m²）
                    double fcMotor;            // 电机侧库仑摩擦
                    double fvMotor;            // 电机侧粘滞摩擦
                    double tauOffsetMotor;     // 电机侧可选常数负载（0 = 关闭）
                };
                ModelParams model;

                // MPC（tcbs::dual_yaw::DualYawMpcConfig；dt_control 取本结构的 dtControl）
                struct MpcParams {
                    int    n;                  // 预测步数 N
                    int    substeps;           // 每控制步 RK4 子步
                    bool   useRk4;             // true: RK4；false: 半隐式欧拉
                    int    maxIter;            // 求解迭代上限
                    double wBigAzimuth;        // 大 yaw 世界方位角跟踪权重
                    double wSmallAzimuth;      // 小 yaw 世界方位角跟踪权重
                    double wSmallCenter;       // 小 yaw 回中权重
                    double wSmallLimit;        // 小 yaw 软限位权重
                    double smallLimitSoftRatio;  // 软限位比例（拆分器判界与 MPC 代价共用）
                    double rBigTorque, rSmallTorque;    // 力矩惩罚
                    double rdBigRate, rdSmallRate;      // 力矩变化率惩罚
                    double smoothEps;          // 位置误差平滑常数
                    int    refDelaySteps;      // 参考延迟步数（0 = 不延迟）
                    double bigMaxTorque;       // 大 yaw 力矩上限（N·m）
                    double bigMaxTorqueRate;   // 大 yaw 力矩变化率上限（N·m/s）
                    double smallMaxTorque;     // 小 yaw 力矩上限（N·m）
                    double smallMaxTorqueRate; // 小 yaw 力矩变化率上限（N·m/s）
                };
                MpcParams mpc;

                // 状态估计（tcbs::YawStateEstimator::Config）
                struct EstimatorParams {
                    int    imuLocation;         // 0 = ON_BIG_YAW（IMU 在大 yaw 转子上）；1 = ON_HEAD
                    double mountYaw, mountPitch, mountRoll;            // R_A_IMU（ZXY）
                    double headMountYaw, headMountPitch, headMountRoll; // R_H_IMU（ZXY）
                    double transportDelayS;     // 链路传输时延（s）
                    double bigEncMaxJump;       // 大 yaw 单次测量最大修正幅度（rad）
                    double staleAgeS;           // 过旧判定阈值（s）
                    double chassisImuTimeoutS;  // 底盘 IMU 可用超时（s）
                    double maxExtrapS;          // 可信量外推上限（s）
                    // 角速度低通（分轴；α = 1 表示直通不滤波）
                    double smallRateLpfAlpha;   // 小 yaw 关节角速度低通系数
                    double bigRateLpfAlpha;     // 大 yaw 平台/关节角速度低通系数
                    // 大 yaw 电机侧角速度低通（来源 = MCU 编码器角速度）
                    double bigMotorRateTauS;    // 低通时间常数（s）
                    double bigMotorRateAlpha;   // 拿不到采样间隔时的兜底系数
                    double backlashCenterTauS;  // 背隙中心 β 在线估计的遗忘时间常数（s；<=0 关闭）
                    double pitchRateLpfAlpha;   // pitch 角速度低通系数
                    double pitchAccLpfAlpha;    // pitch 角加速度低通系数（0 = 不用）
                    double boreX, boreY, boreZ; // 视轴方向（head 系单位矢量）
                    double gravity;             // 估计器内重力
                    bool   useChassisImu;       // 是否用底盘 IMU 分离大 yaw 关节角速度
                    double sourceTimeoutS;      // 数据源超时（s）
                };
                EstimatorParams estimator;

                // MCU 数据线性映射（tcbs::McuDataPreprocessor::LinearParams）
                struct McuLinearParams {
                    double sendPitchScale, sendPitchOffset;   // 关节角 → 电控 pitch 目标值
                    double recvPitchScale, recvPitchOffset;   // 电控原始 pitch → 关节角
                    double recvBigYawScale, recvBigYawOffset;
                    double recvBigOmegaScale;
                    double sendBigYawScale, sendBigYawOffset;
                    double sendBigVelocityScale, sendBigTorqueScale;
                    double recvSmallYawScale, recvSmallYawOffset;
                    double recvSmallOmegaScale;
                    double sendSmallYawScale, sendSmallYawOffset;
                    double sendSmallVelocityScale, sendSmallTorqueScale;
                };
                McuLinearParams mcuLinear;

                // 控制器（tcbs::McuMpcController::Config；loop 周期取 dtControl）
                struct ControllerParams {
                    bool   bigTorqueOnly;      // 大 yaw 仅力矩模式位（发送给电控）
                    bool   smallTorqueOnly;    // 小 yaw 仅力矩模式位
                    double integralGainBig, integralGainSmall;    // 逐关节积分补偿增益
                    double integralLimitBig, integralLimitSmall;  // 逐关节积分限幅
                    bool   integralOnBig;      // 是否允许大 yaw 积分补偿
                };
                ControllerParams controller;
            };
            RobotControllerParamsBS robotController;

            // 关节行程与回中目标（rad；MPC JointLimits / small_center_angle 与拆分器共用）
            struct JointParams {
                double bigMinAngle;       // 大 yaw 行程下界（多圈，通常 -1e9 = 不限位）
                double bigMaxAngle;       // 大 yaw 行程上界
                double smallMinAngle;     // 小 yaw 行程下界（非对称，例：-25°）
                double smallMaxAngle;     // 小 yaw 行程上界（例：+20°）
                double smallCenterAngle;  // 小 yaw 回中目标关节角（0 = 关节零位；拆分器与
                                          // 小 yaw MPC 代价项共用，非对称行程下 0 不是行程中心）
            };
            JointParams joints;

            // 大小 yaw 拆分器（common/BigSmallYaw/BigSmallYawSplitter）的大 yaw 平滑轨迹规划器
            // （同时满足最大速度/加速度/加加速度限制；移植自子模组
            //   python/scripts/trajectory_planner.py）
            struct SplitterParams {
                double plannerMaxVelocity;      // 大 yaw 平滑轨迹最大角速度（rad/s）
                double plannerMaxAcceleration;  // 最大角加速度（rad/s²）
                double plannerMaxJerk;          // 最大角加加速度（rad/s³）
                int    plannerSubsteps;         // 单步细化倍数（>= 1；1 = 与 Python 一致）
            };
            SplitterParams splitter;
        };
        BigSmallBranch bigSmall;
        bool bigSmallPresent = false;   // 配置文件中是否写了 big_small 支（写了就严格校验）
    };

    // 流水线缓冲队列与批量参数（config 各流水线段的 pipeline 子段）
    // 两条流水线结构相同（5 阶段 / 6 队列），各用各的一份配置。
    struct PipelineParams {
        // 6 个缓冲队列最大长度（[输入, 阶段间×4, 输出]）；队列满时新帧直接丢弃。
        // 须与流水线 NUM_QUEUES（=6）一致。
        std::array<int, 6> queueMaxSizes;
        int preprocessBatch;    // 阶段1 预处理最大批量
        int inferenceBatch;     // 阶段2 推理最大批量（须 ≤ inference.max_batch 与共享内存容量）
        int postprocessBatch;   // 阶段3 后处理最大批量
    };

    // Armor 流水线独占参数
    struct ArmorParams {
        std::string modelName = "0526";     // 0526 / 0726
        std::string modelPath;              // 所选模型的推理路径
        std::string device;                 // 推理设备
        int inputWidth;                     // YOLO 推理输入宽度（像素，须与模型输入一致）
        int inputHeight;                    // YOLO 推理输入高度（像素，须与模型输入一致）
        int maxBatch;                       // 推理最大批量（动态 batch 1..max_batch）
        int shmKey;                         // 共享内存 Key（推理进程通信，见 InferShm.h）
        double observationLostTimeoutSec;   // 连续观测丢失多久后重置滤波器（秒）
        PipelineParams pipeline;            // 缓冲队列长度 + 可批处理阶段批量

        // OutpostESEKF 误差状态扩展卡尔曼滤波参数
        struct EsekfParams {
            double positionNoise;           // 位置过程噪声（Q 位置块）
            double rotationNoise;           // 姿态过程噪声（Q 姿态块）
            double measurementNoise;        // 观测噪声（重投影误差 R）
            double orientationZRegNoise;    // 姿态 z 轴正则化观测噪声
            double dzNoise;                 // dz 偏移过程噪声（Q(7,7)/Q(8,8)）
            double dzSearchRange;           // dz 黄金分割搜索范围（米）
            double dzLimit;                 // dz 偏移限幅（米，|dz| 上限）
            double initPositionNoise;       // 初始化位置噪声系数（P 位置块）
            double initOrientationNoise;    // 初始化姿态噪声系数（P 姿态块）
            double initYawRateNoise;        // 初始化旋转速度不确定性（P(6,6)）
            double initDz2Noise;            // 初始化 dz2 不确定性（P(7,7)）
            double initDz3Noise;            // 初始化 dz3 不确定性（P(8,8)）
        };
        EsekfParams esekf;

        // SuperPower EKF（sp_ekf::ClassEKF，label 0~5 每类一个）滤波参数
        // （config: armor.super_power_ekf；字段与 ClassEKF::Params 一一对应，
        //  经 buildConfig 组装成原接口 YAML 配置节点下发）
        struct SuperPowerEkfParams {
            double observationLostTimeoutSec;     // 连续无观测多久后自动销毁内部目标（秒）
            double initialRadiusM;                // SP 普通四装甲初始半径（米）
            bool   jointUpdateEnabled;            // 双板（联合）观测更新开关
            int    minDetectCount;                // 最小识别帧数（DETECTING→TRACKING）
            int    maxTempLostCount;              // 临时失检最大帧数（超过转 LOST 重建）
            double maxDtSec;                      // 最大帧间隔（秒，超过视为时间不连续）
            int    armorNum;                      // 装甲板数量（普通四装甲=4）
            double angularVelocityFitWindowSec;   // 角速度拟合滑窗（秒）
            int    angularVelocityFitMinSamples;  // 角速度拟合最少样本数
            // 双板联合更新门控参数（config: armor.super_power_ekf.joint_update）
            double jointMaxNis;                     // 联合 NIS 门控阈值
            double jointMaxSecondaryPositionErrorM; // 副板最大位置误差门控（米）
            double jointMaxSecondaryAngleErrorRad;  // 副板最大角度误差门控（弧度）
            double jointMeasurementVarianceScale;   // 联合更新位置方差放大系数
            double jointAngleVarianceScale;         // 联合更新角度方差放大系数
        };
        SuperPowerEkfParams superPowerEkf;

        // Armor 目标选取滞回参数（config: armor.target_selection）
        //   stage5 目标级滞回：上一帧选中的目标（filter/label）在再次选取时获得
        //   固定优先度，避免距底盘原点相近的多个候选来回切换（与角速度无关）；
        //   慢目标（施密特触发器）阈值：目标自身角速度 |ω| 低于下阈值视为慢目标
        //   （瞄准点滞回允许），高于上阈值视为快目标，介于两阈值之间保持上一帧
        //   判定（施密特触发器防抖，无连续帧计数）。判定本身由
        //   SequencePredictor::predict 依据预测器下发的目标角速度（带正负）与其
        //   可用标志（Predictor::omega_valid）完成，本段只提供阈值；
        //   角速度不可用的目标（PowerRune / 基地 label 7/8 / EKF 未就绪：
        //   omega_valid = false，角速度填 0）不判定、不启用瞄准点滞回。
        struct TargetSelectionParams {
            double stage5StickPriorityM;      // stage5 目标级滞回固定优先度（米，>=0）
            double slowAngularVelocityLower;  // 慢目标施密特触发下阈值（rad/s，>=0）
            double slowAngularVelocityUpper;  // 慢目标施密特触发上阈值（rad/s，> lower）
            // ── 快目标（fast_target）施密特触发阈值（rad/s，两者均 > slow_angular_velocity_upper）──
            // 目标自身角速度 |ω| 高于 upper 判定为 fast_target（整条预测序列改为对
            // “即将与枪线对齐的那块板”做单点解算，火控数组额外做枪线判定），低于
            // lower 取消 fast_target，介于两者之间保持上一帧判定（与慢目标同一套
            // 施密特防抖逻辑，判定同样在 SequencePredictor::predict 内完成）；
            // 角速度不可用的目标（PowerRune / 基地 label 7/8 / EKF 未就绪）不判定、
            // 不进入 fast_target 分支。
            double fastAngularVelocityLower;  // 快目标施密特触发下阈值（rad/s，> slow upper）
            double fastAngularVelocityUpper;  // 快目标施密特触发上阈值（rad/s，> lower）
        };
        TargetSelectionParams targetSelection;
    };

    // PowerRune 流水线独占参数
    struct PowerRuneParams {
        std::string modelPath;      // 推理模型路径
        std::string device;         // 推理设备
        int         inputWidth;     // YOLO 推理输入宽度（像素，须与模型输入一致）
        int         inputHeight;    // YOLO 推理输入高度（像素，须与模型输入一致）
        bool        manualNms;      // true: 无 NMS 原始输出，需手动 NMS
        float       confThreshold;  // 置信度阈值
        int         maxBatch;       // 推理最大批量
        int         shmKey;         // 共享内存 Key（推理进程通信，见 InferShm.h）
        PipelineParams pipeline;    // 缓冲队列长度 + 可批处理阶段批量

        // RollPredictor（阶段5）拟合参数（config: power_rune.roll_predictor）
        // 控制 RollPredictor 的模型拟合是否启用“宽松”参数：
        //   true : SMALL 模型同时最小二乘拟合斜率（不再固定 π/3）；
        //          BIG 模型放宽参数范围：a ∈ [0.1, 1.045]、ω ∈ [1.826, 2.058]；
        //   false: 保持原有逻辑：SMALL 固定斜率 π/3，
        //          BIG 模型 a ∈ [0.780, 1.045]、ω ∈ [1.884, 2.000]。
        struct RollPredictorParams {
            bool looseFit;   // config: loose_fit
        };
        RollPredictorParams rollPredictor;
    };

    // 共用参数（两个流水线共享）
    struct CommonParams {
        // yaw 构型与**构型相关参数**（tf / 控制器参数 / 新构型模型与拆分器参数）。
        // ⚠ config 里只需填写当前 mode 那一支（见 BigSmallYawParams 与 yaml 文件头约定）。
        BigSmallYawParams bigSmallYaw;

        /// 当前构型（bigSmallYaw.mode）对应的 tf 偏移（另一支可能不存在，不读取）
        const TfOffsets& tf() const {
            return (bigSmallYaw.mode == YawMode::SINGLE) ? bigSmallYaw.single.tf
                                                         : bigSmallYaw.bigSmall.tf;
        }
        /// 当前构型的控制周期（秒）：两构型语义相同（预测序列间隔 / MPC 步长 /
        /// 后台 loop 周期），取自该构型分支的 robot_controller.dt_control
        double dtControl() const {
            return (bigSmallYaw.mode == YawMode::SINGLE)
                       ? bigSmallYaw.single.robotController.dtControl
                       : bigSmallYaw.bigSmall.robotController.dtControl;
        }
        /// 单 yaw 构型（tcs::RobotController）参数；构型不符时内部为空分支（不要读取）
        const RobotControllerParams& singleYawRobotController() const {
            return bigSmallYaw.single.robotController;
        }

        // 输入模式相机参数（common.input_mode）：两流水线共用，按输入模式自动选择
        struct InputModeParams {
            CameraParams cameraMode;    // 相机输入模式（--input camera，含 IP/曝光/增益/extra_info_delay）
            CameraParams videoMode;     // 视频/互动输入模式（--input video/interactive）
        };
        InputModeParams inputMode;

        // 推理进程启动策略（必填，config: common.infer_process_lazy）：
        // true: 仅启动当前流水线所需推理进程（按需启停，见 InferProcessManager）；
        // false: 启动时启动全部推理进程并后台闲置（launch_all.py 预启动）
        bool inferProcessLazy;

        // 推理挂死强制重启时限（秒，必填，config: common.infer_force_restart_timeout_sec）：
        // 推理客户端（InferShmClient）每次推理响应超时（2s）返回时，检测当前时间
        // 与上一次推理正常返回时间的间隔，超过该时限则判定推理进程挂死，调用
        // InferProcessManager::forceRestart 强制重启对应的推理进程（见 InferShmClient
        // 文件头注释）。0 = 关闭该功能。
        double inferForceRestartTimeoutSec;

        // 队列积压自适应额外延迟（可选功能，见 BacklogAdaptiveDelay，v6 PID 式 PI）：
        // 开启后处理线程每次 tryPopFrame() 后统计除输出缓冲队列外各缓冲队列
        // 积压总数，以 targetBacklog 为目标做 PI 直接输出
        // （extra_delay = gainP*e + ∫gainI*e·dt，抗饱和，无 EMA、无速率上限），
        // 得到取帧线程的额外延迟。小量级增益下稳态无极限环震荡。
        // ⚠ 给后续修改者：本结构体所有字段均无默认值，必须由机器配置文件的
        //   common.backlog_adaptive_delay 段提供（缺段/缺字段时 RobotConfig::load
        //   直接抛异常退出），解析见 src/common/RobotConfig.cpp，取值校验也在那里。
        struct BacklogAdaptiveDelayParams {
            bool   enabled;             // 功能总开关（config: enabled）
            double targetBacklog;       // 目标积压（设定点，>= 1，config: target_backlog）
            double maxExtraDelaySeconds; // 额外延迟上限（秒，config: max_extra_delay_seconds）
            double gainP;               // 比例增益，秒/单位积压 >= 0（config: gain_p）
            double gainI;               // 积分增益，秒/单位积压/秒 >= 0（config: gain_i）
        };
        BacklogAdaptiveDelayParams backlogAdaptiveDelay;

        GimbalParams gimbal;                    // 云台解算参数
        PredictedBallisticParams predictedBallistic;  // 预测弹道解算参数
        RobotControllerParams robotController;         // tcs::RobotController 构造参数
        PredictSequenceParams predictSequence;         // 预测序列参数
        double       minDelaySeconds;           // 两个流水线共用的提取帧最小延迟（秒）

        // 录制参数（必填段，config: common.recording；--record 开启录制时生效）
        // ⚠ 无默认值：output_dir / min_free_space_mb 必须由机器配置文件提供。
        struct RecordingParams {
            std::string outputDir;      // 录制输出目录（相对项目根目录或以 / 开头为绝对路径）
            int64_t minFreeSpaceBytes;  // 剩余空间阈值（字节，低于该值停止写入；
                                        // config 中为 MB，加载时换算）
        };
        RecordingParams recording;
    };

    CommonParams    common;      // 共用参数
    ArmorParams   armor;     // Armor 独占参数
    PowerRuneParams powerRune;   // PowerRune 独占参数

    // 从指定 yaml 文件加载配置；文件缺失或格式错误抛出 std::runtime_error。
    static RobotConfig load(const std::string& yamlPath);

    // 懒加载单例：首次调用时经机器配置选择器 config/selector.yaml 定位并读取
    // 当前机器配置文件（config/robots/<active_config>.yaml）。
    static RobotConfig& instance();
};

#endif // ROBOT_CONFIG_H
