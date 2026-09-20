// RobotConfig.cpp — 从 yaml 配置文件加载全局参数（common / armor / power_rune 三结构）
#include "common/RobotConfig.h"

#include <stdexcept>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "common/PathResolver.h"
#include "common/Infer/InferShm.h"

namespace {

// 读取 section[key] 为标量 T；缺失或类型错误抛出 std::runtime_error。
template <typename T>
T requireScalar(const YAML::Node& section, const std::string& key, const std::string& sectionName) {
    const YAML::Node& n = section[key];
    if (!n || !n.IsDefined()) {
        throw std::runtime_error("RobotConfig: 配置段 '" + sectionName + "' 缺少配置项 '" + key + "'");
    }
    try {
        return n.as<T>();
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("RobotConfig: 配置项 '" + sectionName + "." + key + "' 类型错误: " + e.what());
    }
}

// 读取 section[key] 为长度恰为 expectSize 的数值列表。
std::vector<double> requireList(const YAML::Node& section, const std::string& key,
                                const std::string& sectionName, size_t expectSize) {
    const YAML::Node& n = section[key];
    if (!n || !n.IsDefined()) {
        throw std::runtime_error("RobotConfig: 配置段 '" + sectionName + "' 缺少配置项 '" + key + "'");
    }
    std::vector<double> v;
    try {
        v = n.as<std::vector<double>>();
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("RobotConfig: 配置项 '" + sectionName + "." + key + "' 类型错误: " + e.what());
    }
    if (v.size() != expectSize) {
        throw std::runtime_error("RobotConfig: 配置项 '" + sectionName + "." + key +
                                 "' 需要 " + std::to_string(expectSize) + " 个元素，实际 " +
                                 std::to_string(v.size()));
    }
    return v;
}

// 解析一套相机参数（common.input_mode.camera_mode / video_mode）。
// ⚠ 不设任何默认值：每个模式段中的所有参数都必须显式出现在机器配置文件
//   （config/robots/*.yaml），缺字段即抛异常。
void parseCameraParams(const YAML::Node& camNode, const std::string& name,
                       RobotConfig::CameraParams& out, bool cameraMode) {
    const YAML::Node& res = camNode["resolution"];
    if (!res || !res.IsMap()) throw std::runtime_error("RobotConfig: 缺少 '" + name + ".resolution' 配置段");
    out.width  = requireScalar<int>(res, "width", name + ".resolution");
    out.height = requireScalar<int>(res, "height", name + ".resolution");
    std::vector<double> cm = requireList(camNode, "camera_matrix", name, 9);
    std::vector<double> dc = requireList(camNode, "dist_coeffs", name, 5);
    out.cameraMatrix = (cv::Mat_<double>(3, 3) <<
        cm[0], cm[1], cm[2], cm[3], cm[4], cm[5], cm[6], cm[7], cm[8]);
    out.distCoeffs.create((int)dc.size(), 1, CV_64F);
    for (size_t i = 0; i < dc.size(); ++i) {
        out.distCoeffs.at<double>((int)i, 0) = dc[i];
    }
    if (cameraMode) {
        // 相机模式（实机相机：IP / 曝光 / 增益）
        out.deviceIp  = requireScalar<std::string>(camNode, "device_ip", name);
        out.netIp     = requireScalar<std::string>(camNode, "net_ip", name);
        out.exposure  = requireScalar<float>(camNode, "exposure", name);
        out.gain      = requireScalar<float>(camNode, "gain", name);
    } else {
        // 视频/交互模式（测试最大帧率开关）
        out.testMaxFps = requireScalar<bool>(camNode, "test_max_fps", name);
    }
}

// 解析流水线缓冲队列与批量参数段（armor.pipeline / power_rune.pipeline）：
// queue_max_sizes 必为 6 个正整数（[输入, 阶段间×4, 输出]）；preprocess_batch /
// inference_batch / postprocess_batch 必为正整数，且 inference_batch 不得超过
// 该流水线 inference.max_batch（模型编译的动态批量上限，maxBatch 参数传入）。
void parsePipelineParams(const YAML::Node& node, const std::string& name,
                         int maxBatch, RobotConfig::PipelineParams& out) {
    const YAML::Node& qs = node["queue_max_sizes"];
    if (!qs || !qs.IsDefined())
        throw std::runtime_error("RobotConfig: 配置段 '" + name + "' 缺少配置项 'queue_max_sizes'");
    std::vector<int> qv;
    try {
        qv = qs.as<std::vector<int>>();
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("RobotConfig: 配置项 '" + name + ".queue_max_sizes' 类型错误: " + e.what());
    }
    if (qv.size() != 6) {
        throw std::runtime_error("RobotConfig: 配置项 '" + name + ".queue_max_sizes' 需要 6 个元素"
                                 "（输入 + 阶段间×4 + 输出），实际 " + std::to_string(qv.size()));
    }
    for (size_t i = 0; i < 6; ++i) {
        if (qv[i] < 1) {
            throw std::runtime_error("RobotConfig: 配置项 '" + name + ".queue_max_sizes[" +
                                     std::to_string(i) + "]' 必须 >= 1");
        }
        out.queueMaxSizes[i] = qv[i];
    }
    out.preprocessBatch  = requireScalar<int>(node, "preprocess_batch", name);
    out.inferenceBatch   = requireScalar<int>(node, "inference_batch", name);
    out.postprocessBatch = requireScalar<int>(node, "postprocess_batch", name);
    if (out.preprocessBatch < 1)
        throw std::runtime_error("RobotConfig: 配置项 '" + name + ".preprocess_batch' 必须 >= 1");
    if (out.inferenceBatch < 1)
        throw std::runtime_error("RobotConfig: 配置项 '" + name + ".inference_batch' 必须 >= 1");
    if (out.postprocessBatch < 1)
        throw std::runtime_error("RobotConfig: 配置项 '" + name + ".postprocess_batch' 必须 >= 1");
    if (out.inferenceBatch > maxBatch) {
        throw std::runtime_error("RobotConfig: 配置项 '" + name + ".inference_batch' (" +
                                 std::to_string(out.inferenceBatch) + ") 不能超过该流水线 "
                                 "inference.max_batch (" + std::to_string(maxBatch) + ")");
    }
}

// ── 解析 tf 偏移段（两构型共用同一套字段名；big_small 额外要求小 yaw 偏移）──
void parseTfOffsets(const YAML::Node& tf, const std::string& S,
                    RobotConfig::TfOffsets& out, bool needSmallYawOffset) {
    if (!tf || !tf.IsMap()) throw std::runtime_error("RobotConfig: 缺少 '" + S + "' 配置段");
    out.yawJointZOffset   = requireScalar<float>(tf, "yaw_joint_z_offset", S);
    out.pitchJointYOffset = requireScalar<float>(tf, "pitch_joint_y_offset", S);
    out.imuOffsetX        = requireScalar<float>(tf, "imu_offset_x", S);
    out.imuOffsetY        = requireScalar<float>(tf, "imu_offset_y", S);
    out.imuOffsetZ        = requireScalar<float>(tf, "imu_offset_z", S);
    out.cameraOffsetX     = requireScalar<float>(tf, "camera_offset_x", S);
    out.cameraOffsetY     = requireScalar<float>(tf, "camera_offset_y", S);
    out.cameraOffsetZ     = requireScalar<float>(tf, "camera_offset_z", S);
    out.muzzleOffsetX     = requireScalar<float>(tf, "muzzle_offset_x", S);
    out.muzzleOffsetY     = requireScalar<float>(tf, "muzzle_offset_y", S);
    out.muzzleOffsetZ     = requireScalar<float>(tf, "muzzle_offset_z", S);
    if (needSmallYawOffset) {
        // 小 yaw 轴相对大 yaw 轴的偏移（大 yaw 系，米）：变换树 yaw_small 节点位置 +
        // 平面模型 dx/dy（只配置这一处）
        out.smallYawOffsetX = requireScalar<float>(tf, "small_yaw_offset_x", S);
        out.smallYawOffsetY = requireScalar<float>(tf, "small_yaw_offset_y", S);
        out.smallYawOffsetZ = requireScalar<float>(tf, "small_yaw_offset_z", S);
    }
}

// ── 解析 single 支：tf + tcs::RobotController 构造参数（原 common.tf / common.robot_controller）──
void parseSingleYawBranch(const YAML::Node& node, RobotConfig::BigSmallYawParams::SingleBranch& out) {
    const std::string S = "common.big_small_yaw.single";
    if (!node || !node.IsMap()) throw std::runtime_error("RobotConfig: 缺少 '" + S + "' 配置段");
    parseTfOffsets(node["tf"], S + ".tf", out.tf, /*needSmallYawOffset=*/false);

    const YAML::Node& rc = node["robot_controller"];
    if (!rc || !rc.IsMap())
        throw std::runtime_error("RobotConfig: 缺少 '" + S + ".robot_controller' 配置段");
    const std::string R = S + ".robot_controller";
    auto& k = out.robotController;
    k.sequenceMode        = requireScalar<bool>(rc, "sequence_mode", R);
    k.yawTorqueOnlyMode   = requireScalar<bool>(rc, "yaw_torque_only_mode", R);
    k.dtControl           = requireScalar<double>(rc, "dt_control", R);
    k.mpcPredN            = requireScalar<int>(rc, "mpc_pred_n", R);
    k.J                   = requireScalar<double>(rc, "J", R);
    k.tauC                = requireScalar<double>(rc, "tau_c", R);
    k.b                   = requireScalar<double>(rc, "b", R);
    k.tauD                = requireScalar<double>(rc, "tau_d", R);
    k.maxTorque           = requireScalar<double>(rc, "max_torque", R);
    k.maxTorqueRate       = requireScalar<double>(rc, "max_torque_rate", R);
    k.Q                   = requireScalar<double>(rc, "Q", R);
    k.R                   = requireScalar<double>(rc, "R", R);
    k.Rd                  = requireScalar<double>(rc, "Rd", R);
    k.maxIter             = requireScalar<int>(rc, "max_iter", R);
    k.integralGain        = requireScalar<double>(rc, "integral_gain", R);
    k.smoothEps           = requireScalar<double>(rc, "smooth_eps", R);
    k.sendPitchScale      = requireScalar<double>(rc, "send_pitch_scale", R);
    k.sendPitchOffset     = requireScalar<double>(rc, "send_pitch_offset", R);
    k.recvPitchScale      = requireScalar<double>(rc, "recv_pitch_scale", R);
    k.recvPitchOffset     = requireScalar<double>(rc, "recv_pitch_offset", R);
    if (k.dtControl <= 0.0)
        throw std::runtime_error("RobotConfig: '" + R + ".dt_control' 必须 > 0");
    if (k.mpcPredN < 1)
        throw std::runtime_error("RobotConfig: '" + R + ".mpc_pred_n' 必须 >= 1");
    if (k.maxIter < 1)
        throw std::runtime_error("RobotConfig: '" + R + ".max_iter' 必须 >= 1");
    if (!(k.maxTorque > 0.0) || !(k.maxTorqueRate > 0.0))
        throw std::runtime_error("RobotConfig: '" + R + "' 的 max_torque / max_torque_rate 必须 > 0");
}

// ── 解析 big_small 支：tf（含小 yaw 偏移）+ tcbs::RobotController 参数 + joints + splitter ──
void parseBigSmallYawBranch(const YAML::Node& node,
                            RobotConfig::BigSmallYawParams::BigSmallBranch& out) {
    const std::string S = "common.big_small_yaw.big_small";
    if (!node || !node.IsMap()) throw std::runtime_error("RobotConfig: 缺少 '" + S + "' 配置段");
    parseTfOffsets(node["tf"], S + ".tf", out.tf, /*needSmallYawOffset=*/true);

    // ── robot_controller（tcbs::RobotController 构造参数）──
    const YAML::Node& rc = node["robot_controller"];
    if (!rc || !rc.IsMap())
        throw std::runtime_error("RobotConfig: 缺少 '" + S + ".robot_controller' 配置段");
    const std::string R = S + ".robot_controller";
    auto& k = out.robotController;
    k.sequenceMode = requireScalar<bool>(rc, "sequence_mode", R);
    k.dtControl    = requireScalar<double>(rc, "dt_control", R);
    if (!k.sequenceMode) {
        throw std::runtime_error("RobotConfig: '" + R + ".sequence_mode' 必须为 true"
                                 "（大/小 yaw 输出按序列下发）");
    }
    if (k.dtControl <= 0.0)
        throw std::runtime_error("RobotConfig: '" + R + ".dt_control' 必须 > 0");

    // model（tcbs::dual_yaw::ModelParams；dx/dy 取 tf 的小 yaw 偏移，不重复配置）
    const YAML::Node& md = rc["model"];
    if (!md || !md.IsMap()) throw std::runtime_error("RobotConfig: 缺少 '" + R + ".model' 配置段");
    const std::string M = R + ".model";
    auto& m = k.model;
    m.gravity        = requireScalar<double>(md, "gravity", M);
    m.mUKnown        = requireScalar<double>(md, "m_u_known", M);
    m.JbigEff        = requireScalar<double>(md, "Jbig_eff", M);
    m.Js             = requireScalar<double>(md, "Js", M);
    m.Px             = requireScalar<double>(md, "Px", M);
    m.Py             = requireScalar<double>(md, "Py", M);
    m.fcBig          = requireScalar<double>(md, "fc_big", M);
    m.fvBig          = requireScalar<double>(md, "fv_big", M);
    m.fcSmall        = requireScalar<double>(md, "fc_small", M);
    m.fvSmall        = requireScalar<double>(md, "fv_small", M);
    m.frictionLambda = requireScalar<double>(md, "friction_lambda", M);
    m.tauOffsetBig   = requireScalar<double>(md, "tau_offset_big", M);
    m.tauOffsetSmall = requireScalar<double>(md, "tau_offset_small", M);
    // 大 yaw 传动背隙（tcbs 3-DOF 模型）
    m.backlashDelta     = requireScalar<double>(md, "backlash_delta", M);
    m.backlashK         = requireScalar<double>(md, "backlash_k", M);
    m.backlashC         = requireScalar<double>(md, "backlash_c", M);
    m.backlashSmoothEps = requireScalar<double>(md, "backlash_smooth_eps", M);
    m.backlashThrough   = requireScalar<double>(md, "backlash_through", M);
    m.Jmotor            = requireScalar<double>(md, "Jmotor", M);
    m.fcMotor           = requireScalar<double>(md, "fc_motor", M);
    m.fvMotor           = requireScalar<double>(md, "fv_motor", M);
    m.tauOffsetMotor    = requireScalar<double>(md, "tau_offset_motor", M);
    if (!(m.JbigEff > 0.0) || !(m.Js > 0.0))
        throw std::runtime_error("RobotConfig: '" + M + "' 的 Jbig_eff / Js 必须 > 0");
    if (!(m.frictionLambda >= 0.0))
        throw std::runtime_error("RobotConfig: '" + M + ".friction_lambda' 必须 >= 0");
    if (!(m.backlashDelta >= 0.0) || !(m.backlashK >= 0.0) || !(m.backlashC >= 0.0) ||
        !(m.backlashSmoothEps >= 0.0) || !(m.backlashThrough >= 0.0))
        throw std::runtime_error("RobotConfig: '" + M + "' 的 backlash_delta / backlash_k / "
                                 "backlash_c / backlash_smooth_eps / backlash_through 必须 >= 0");
    if (!(m.Jmotor >= 0.0) || !(m.fcMotor >= 0.0) || !(m.fvMotor >= 0.0))
        throw std::runtime_error("RobotConfig: '" + M + "' 的 Jmotor / fc_motor / fv_motor 必须 >= 0");

    // mpc（tcbs::dual_yaw::DualYawMpcConfig；dt_control 取 robot_controller.dt_control）
    const YAML::Node& mp = rc["mpc"];
    if (!mp || !mp.IsMap()) throw std::runtime_error("RobotConfig: 缺少 '" + R + ".mpc' 配置段");
    const std::string P = R + ".mpc";
    auto& c = k.mpc;
    c.n                   = requireScalar<int>(mp, "pred_n", P);
    c.substeps            = requireScalar<int>(mp, "substeps", P);
    c.useRk4              = requireScalar<bool>(mp, "use_rk4", P);
    c.maxIter             = requireScalar<int>(mp, "max_iter", P);
    c.wBigAzimuth         = requireScalar<double>(mp, "w_big_azimuth", P);
    c.wSmallAzimuth       = requireScalar<double>(mp, "w_small_azimuth", P);
    c.wSmallCenter        = requireScalar<double>(mp, "w_small_center", P);
    c.wSmallLimit         = requireScalar<double>(mp, "w_small_limit", P);
    c.smallLimitSoftRatio = requireScalar<double>(mp, "small_limit_soft_ratio", P);
    c.rBigTorque          = requireScalar<double>(mp, "r_big_torque", P);
    c.rSmallTorque        = requireScalar<double>(mp, "r_small_torque", P);
    c.rdBigRate           = requireScalar<double>(mp, "rd_big_rate", P);
    c.rdSmallRate         = requireScalar<double>(mp, "rd_small_rate", P);
    c.smoothEps           = requireScalar<double>(mp, "smooth_eps", P);
    c.refDelaySteps       = requireScalar<int>(mp, "ref_delay_steps", P);
    c.bigMaxTorque        = requireScalar<double>(mp, "big_max_torque", P);
    c.bigMaxTorqueRate    = requireScalar<double>(mp, "big_max_torque_rate", P);
    c.smallMaxTorque      = requireScalar<double>(mp, "small_max_torque", P);
    c.smallMaxTorqueRate  = requireScalar<double>(mp, "small_max_torque_rate", P);
    if (c.n < 1)        throw std::runtime_error("RobotConfig: '" + P + ".pred_n' 必须 >= 1");
    if (c.substeps < 1) throw std::runtime_error("RobotConfig: '" + P + ".substeps' 必须 >= 1");
    if (c.maxIter < 1)  throw std::runtime_error("RobotConfig: '" + P + ".max_iter' 必须 >= 1");
    if (!(c.smallLimitSoftRatio > 0.0 && c.smallLimitSoftRatio <= 1.0))
        throw std::runtime_error("RobotConfig: '" + P + ".small_limit_soft_ratio' 必须落在 (0, 1]");
    if (c.refDelaySteps < 0)
        throw std::runtime_error("RobotConfig: '" + P + ".ref_delay_steps' 必须 >= 0");
    if (!(c.bigMaxTorque > 0.0) || !(c.bigMaxTorqueRate > 0.0) ||
        !(c.smallMaxTorque > 0.0) || !(c.smallMaxTorqueRate > 0.0))
        throw std::runtime_error("RobotConfig: '" + P + "' 的力矩上限与力矩变化率上限必须 > 0");

    // estimator（tcbs::YawStateEstimator::Config）
    const YAML::Node& es = rc["estimator"];
    if (!es || !es.IsMap()) throw std::runtime_error("RobotConfig: 缺少 '" + R + ".estimator' 配置段");
    const std::string E = R + ".estimator";
    auto& e = k.estimator;
    e.imuLocation        = requireScalar<int>(es, "imu_location", E);
    e.mountYaw           = requireScalar<double>(es, "mount_yaw", E);
    e.mountPitch         = requireScalar<double>(es, "mount_pitch", E);
    e.mountRoll          = requireScalar<double>(es, "mount_roll", E);
    e.headMountYaw       = requireScalar<double>(es, "head_mount_yaw", E);
    e.headMountPitch     = requireScalar<double>(es, "head_mount_pitch", E);
    e.headMountRoll      = requireScalar<double>(es, "head_mount_roll", E);
    e.transportDelayS    = requireScalar<double>(es, "transport_delay_s", E);
    e.bigEncMaxJump      = requireScalar<double>(es, "big_enc_max_jump", E);
    e.staleAgeS          = requireScalar<double>(es, "stale_age_s", E);
    e.chassisImuTimeoutS = requireScalar<double>(es, "chassis_imu_timeout_s", E);
    e.maxExtrapS         = requireScalar<double>(es, "max_extrap_s", E);
    e.smallRateLpfAlpha  = requireScalar<double>(es, "small_rate_lpf_alpha", E);
    e.bigRateLpfAlpha    = requireScalar<double>(es, "big_rate_lpf_alpha", E);
    e.bigMotorRateTauS   = requireScalar<double>(es, "big_motor_rate_tau_s", E);
    e.bigMotorRateAlpha  = requireScalar<double>(es, "big_motor_rate_alpha", E);
    e.backlashCenterTauS = requireScalar<double>(es, "backlash_center_tau_s", E);
    e.pitchRateLpfAlpha  = requireScalar<double>(es, "pitch_rate_lpf_alpha", E);
    e.pitchAccLpfAlpha   = requireScalar<double>(es, "pitch_acc_lpf_alpha", E);
    e.boreX              = requireScalar<double>(es, "bore_x", E);
    e.boreY              = requireScalar<double>(es, "bore_y", E);
    e.boreZ              = requireScalar<double>(es, "bore_z", E);
    e.gravity            = requireScalar<double>(es, "gravity", E);
    e.useChassisImu      = requireScalar<bool>(es, "use_chassis_imu", E);
    e.sourceTimeoutS     = requireScalar<double>(es, "source_timeout_s", E);
    if (e.imuLocation != 0 && e.imuLocation != 1)
        throw std::runtime_error("RobotConfig: '" + E + ".imu_location' 只能是 0（ON_BIG_YAW）"
                                 " 或 1（ON_HEAD）");
    if (!(e.smallRateLpfAlpha > 0.0 && e.smallRateLpfAlpha <= 1.0))
        throw std::runtime_error("RobotConfig: '" + E + ".small_rate_lpf_alpha' 必须落在 (0, 1]");
    if (!(e.bigRateLpfAlpha > 0.0 && e.bigRateLpfAlpha <= 1.0))
        throw std::runtime_error("RobotConfig: '" + E + ".big_rate_lpf_alpha' 必须落在 (0, 1]");
    if (!(e.bigMotorRateTauS >= 0.0))
        throw std::runtime_error("RobotConfig: '" + E + ".big_motor_rate_tau_s' 必须 >= 0");
    if (!(e.bigMotorRateAlpha > 0.0 && e.bigMotorRateAlpha <= 1.0))
        throw std::runtime_error("RobotConfig: '" + E + ".big_motor_rate_alpha' 必须落在 (0, 1]");
    if (!(e.pitchRateLpfAlpha > 0.0 && e.pitchRateLpfAlpha <= 1.0))
        throw std::runtime_error("RobotConfig: '" + E + ".pitch_rate_lpf_alpha' 必须落在 (0, 1]");
    if (!(e.pitchAccLpfAlpha >= 0.0 && e.pitchAccLpfAlpha <= 1.0))
        throw std::runtime_error("RobotConfig: '" + E + ".pitch_acc_lpf_alpha' 必须落在 [0, 1]");

    // mcu_linear（tcbs::McuDataPreprocessor::LinearParams）
    const YAML::Node& ml = rc["mcu_linear"];
    if (!ml || !ml.IsMap()) throw std::runtime_error("RobotConfig: 缺少 '" + R + ".mcu_linear' 配置段");
    const std::string L = R + ".mcu_linear";
    auto& l = k.mcuLinear;
    l.sendPitchScale         = requireScalar<double>(ml, "send_pitch_scale", L);
    l.sendPitchOffset        = requireScalar<double>(ml, "send_pitch_offset", L);
    l.recvPitchScale         = requireScalar<double>(ml, "recv_pitch_scale", L);
    l.recvPitchOffset        = requireScalar<double>(ml, "recv_pitch_offset", L);
    l.recvBigYawScale        = requireScalar<double>(ml, "recv_big_yaw_scale", L);
    l.recvBigYawOffset       = requireScalar<double>(ml, "recv_big_yaw_offset", L);
    l.recvBigOmegaScale      = requireScalar<double>(ml, "recv_big_omega_scale", L);
    l.sendBigYawScale        = requireScalar<double>(ml, "send_big_yaw_scale", L);
    l.sendBigYawOffset       = requireScalar<double>(ml, "send_big_yaw_offset", L);
    l.sendBigVelocityScale   = requireScalar<double>(ml, "send_big_velocity_scale", L);
    l.sendBigTorqueScale     = requireScalar<double>(ml, "send_big_torque_scale", L);
    l.recvSmallYawScale      = requireScalar<double>(ml, "recv_small_yaw_scale", L);
    l.recvSmallYawOffset     = requireScalar<double>(ml, "recv_small_yaw_offset", L);
    l.recvSmallOmegaScale    = requireScalar<double>(ml, "recv_small_omega_scale", L);
    l.sendSmallYawScale      = requireScalar<double>(ml, "send_small_yaw_scale", L);
    l.sendSmallYawOffset     = requireScalar<double>(ml, "send_small_yaw_offset", L);
    l.sendSmallVelocityScale = requireScalar<double>(ml, "send_small_velocity_scale", L);
    l.sendSmallTorqueScale   = requireScalar<double>(ml, "send_small_torque_scale", L);

    // controller（tcbs::McuMpcController::Config；loop_period 取 dt_control）
    const YAML::Node& ct = rc["controller"];
    if (!ct || !ct.IsMap()) throw std::runtime_error("RobotConfig: 缺少 '" + R + ".controller' 配置段");
    const std::string C = R + ".controller";
    auto& q = k.controller;
    q.bigTorqueOnly      = requireScalar<bool>(ct, "big_torque_only", C);
    q.smallTorqueOnly    = requireScalar<bool>(ct, "small_torque_only", C);
    q.integralGainBig    = requireScalar<double>(ct, "integral_gain_big", C);
    q.integralGainSmall  = requireScalar<double>(ct, "integral_gain_small", C);
    q.integralLimitBig   = requireScalar<double>(ct, "integral_limit_big", C);
    q.integralLimitSmall = requireScalar<double>(ct, "integral_limit_small", C);
    q.integralOnBig      = requireScalar<bool>(ct, "integral_on_big", C);
    if (q.integralGainBig < 0.0 || q.integralGainSmall < 0.0)
        throw std::runtime_error("RobotConfig: '" + C + "' 的 integral_gain_big / "
                                 "integral_gain_small 必须 >= 0");

    // ── joints（行程与回中；MPC JointLimits / small_center_angle 与拆分器共用）──
    const YAML::Node& jn = node["joints"];
    if (!jn || !jn.IsMap()) throw std::runtime_error("RobotConfig: 缺少 '" + S + ".joints' 配置段");
    const std::string J = S + ".joints";
    auto& j = out.joints;
    j.bigMinAngle      = requireScalar<double>(jn, "big_min_angle", J);
    j.bigMaxAngle      = requireScalar<double>(jn, "big_max_angle", J);
    j.smallMinAngle    = requireScalar<double>(jn, "small_min_angle", J);
    j.smallMaxAngle    = requireScalar<double>(jn, "small_max_angle", J);
    j.smallCenterAngle = requireScalar<double>(jn, "small_center_angle", J);
    if (!(j.bigMinAngle < j.bigMaxAngle))
        throw std::runtime_error("RobotConfig: '" + J + "' 的 big_min_angle 必须小于 big_max_angle");
    if (!(j.smallMinAngle < j.smallMaxAngle))
        throw std::runtime_error("RobotConfig: '" + J + "' 的 small_min_angle 必须小于 small_max_angle");
    if (j.smallCenterAngle < j.smallMinAngle || j.smallCenterAngle > j.smallMaxAngle)
        throw std::runtime_error("RobotConfig: '" + J + ".small_center_angle' 必须落在小 yaw 行程内");

    // ── splitter（大 yaw 平滑轨迹规划器）──
    const YAML::Node& sp = node["splitter"];
    if (!sp || !sp.IsMap()) throw std::runtime_error("RobotConfig: 缺少 '" + S + ".splitter' 配置段");
    const std::string SP = S + ".splitter";
    auto& s = out.splitter;
    s.plannerMaxVelocity     = requireScalar<double>(sp, "planner_max_velocity", SP);
    s.plannerMaxAcceleration = requireScalar<double>(sp, "planner_max_acceleration", SP);
    s.plannerMaxJerk         = requireScalar<double>(sp, "planner_max_jerk", SP);
    s.plannerSubsteps        = requireScalar<int>(sp, "planner_substeps", SP);
    if (!(s.plannerMaxVelocity > 0.0) || !(s.plannerMaxAcceleration > 0.0) ||
        !(s.plannerMaxJerk > 0.0))
        throw std::runtime_error("RobotConfig: '" + SP + "' 的 planner_max_velocity / "
                                 "planner_max_acceleration / planner_max_jerk 必须 > 0");
    if (s.plannerSubsteps < 1)
        throw std::runtime_error("RobotConfig: '" + SP + ".planner_substeps' 必须 >= 1");
}

// 解析 common.big_small_yaw：构型开关 + 两支。
// ⚠ 约定：**只需填写当前 mode 那一支**；另一支可以整段省略，若写了则同样严格校验
//   （缺字段报错），当前构型那一支缺失或字段不全一律抛异常。
void parseBigSmallYawParams(const YAML::Node& bs, RobotConfig::BigSmallYawParams& out) {
    const std::string S = "common.big_small_yaw";
    if (!bs || !bs.IsMap()) throw std::runtime_error("RobotConfig: 缺少 '" + S + "' 配置段");

    // 构型开关（字符串，仅接受 single / big_small）
    const std::string modeStr = requireScalar<std::string>(bs, "mode", S);
    if (modeStr == "single")         out.mode = YawMode::SINGLE;
    else if (modeStr == "big_small") out.mode = YawMode::BIG_SMALL;
    else throw std::runtime_error("RobotConfig: " + S + ".mode 只能是 'single' 或 "
                                  "'big_small'，实际为 '" + modeStr + "'");

    // 两支：存在即严格校验
    out.singlePresent   = (bool)bs["single"];
    out.bigSmallPresent = (bool)bs["big_small"];
    if (out.singlePresent)   parseSingleYawBranch(bs["single"], out.single);
    if (out.bigSmallPresent) parseBigSmallYawBranch(bs["big_small"], out.bigSmall);

    // 当前构型那一支必须存在（另一支缺失是允许的）
    if (out.mode == YawMode::SINGLE && !out.singlePresent) {
        throw std::runtime_error("RobotConfig: " + S + ".mode = 'single' 时必须提供 "
                                 "'" + S + ".single' 配置段（本构型的 tf / robot_controller）");
    }
    if (out.mode == YawMode::BIG_SMALL && !out.bigSmallPresent) {
        throw std::runtime_error("RobotConfig: " + S + ".mode = 'big_small' 时必须提供 "
                                 "'" + S + ".big_small' 配置段"
                                 "（本构型的 tf / robot_controller / joints / splitter）");
    }
}

} // namespace


// ⚠ 给后续修改者：所有参数均无代码默认值，必须由机器配置文件
//   （config/robots/<active_config>.yaml）提供；
//   缺段或缺字段直接抛异常（不静默采用默认值）。
RobotConfig RobotConfig::load(const std::string& yamlPath) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(yamlPath);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("RobotConfig: 读取/解析配置文件失败 '" + yamlPath + "': " + e.what());
    }
    if (!root || !root.IsMap()) {
        throw std::runtime_error("RobotConfig: 配置文件 '" + yamlPath + "' 为空或不是映射结构");
    }

    RobotConfig cfg;

    // ══════════════ common（共用参数） ══════════════
    const YAML::Node& cm = root["common"];
    if (!cm || !cm.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'common' 配置段");

    // ── common.big_small_yaw（yaw 构型开关 + **构型相关参数**）──
    // 本段内按构型分为 single / big_small 两支，**只需填写当前 mode 那一支**；
    // 当前构型那一支内的字段全部必填（缺字段抛异常，绝不静默采用默认值），
    // 变换树偏移（tf）与控制器参数（robot_controller）都在此段内，见文件头约定。
    parseBigSmallYawParams(cm["big_small_yaw"], cfg.common.bigSmallYaw);

    // ── common.input_mode（按输入模式自动选择：camera_mode / video_mode）──
    const YAML::Node& im = cm["input_mode"];
    if (!im || !im.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'common.input_mode' 配置段");
    const YAML::Node& cam = im["camera_mode"];
    if (!cam || !cam.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'common.input_mode.camera_mode' 配置段");
    parseCameraParams(cam, "common.input_mode.camera_mode", cfg.common.inputMode.cameraMode,
                      /*cameraMode=*/true);
    const YAML::Node& vid = im["video_mode"];
    if (!vid || !vid.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'common.input_mode.video_mode' 配置段");
    parseCameraParams(vid, "common.input_mode.video_mode", cfg.common.inputMode.videoMode,
                      /*cameraMode=*/false);

    // ── common.input_mode.camera_mode.extra_info_delay（相机输入模式 extra_info
    //    延迟；相机模式，无默认值）──
    cfg.common.inputMode.cameraMode.extraInfoDelay =
        requireScalar<double>(cam, "extra_info_delay", "common.input_mode.camera_mode");

    // ── common.min_delay_seconds（两个流水线共用）──
    cfg.common.minDelaySeconds = requireScalar<double>(cm, "min_delay_seconds", "common");

    // ── common.infer_process_lazy──
    //   false：启动时启动全部推理进程并后台闲置（launch_all.py 预启动）；
    //   true：仅启动当前流水线所需推理进程，切换时立即关闭不需要的进程（主程序管理）
    cfg.common.inferProcessLazy = requireScalar<bool>(cm, "infer_process_lazy", "common");

    // ── common.infer_force_restart_timeout_sec（推理挂死强制重启时限）──
    // 推理客户端（InferShmClient）每次推理响应超时（2s）返回时，检测距上一次
    // 推理正常返回的间隔，超过该时限则判定推理进程挂死，经 InferProcessManager::
    // forceRestart 强制重启对应推理进程（见 InferShmClient 文件头注释）。
    // 0 = 关闭该功能。
    cfg.common.inferForceRestartTimeoutSec =
        requireScalar<double>(cm, "infer_force_restart_timeout_sec", "common");
    if (cfg.common.inferForceRestartTimeoutSec < 0.0) {
        throw std::runtime_error("RobotConfig: common.infer_force_restart_timeout_sec "
                                 "必须 >= 0（0 = 关闭推理挂死强制重启）");
    }

    // ── common.backlog_adaptive_delay（队列积压自适应额外延迟，v6 PID 式 PI）──
    // 开启后（见 BacklogAdaptiveDelay）：
    //   - 以 target_backlog 为目标做 PI 直接输出：extra_delay = gain_p*e +
    //     ∫gain_i*e·dt（增益单位分别为 秒/单位积压、秒/单位积压/秒，内部换算
    //     为 µs；e 为积压总数与目标的差，无 EMA 平滑），钳位
    //     [0, max_extra_delay_seconds]（抗饱和，无速率上限）。稳态时积压
    //     收敛到 target_backlog，误差归零。
    //   若要在机器配置文件中新增本功能的参数，需同步修改
    //   RobotConfig::BacklogAdaptiveDelayParams 与本段解析。
    const YAML::Node& bad = cm["backlog_adaptive_delay"];
    if (!bad || !bad.IsMap())
        throw std::runtime_error("RobotConfig: 缺少 'common.backlog_adaptive_delay' 配置段");
    cfg.common.backlogAdaptiveDelay.enabled =
        requireScalar<bool>(bad, "enabled", "common.backlog_adaptive_delay");
    cfg.common.backlogAdaptiveDelay.targetBacklog =
        requireScalar<double>(bad, "target_backlog", "common.backlog_adaptive_delay");
    cfg.common.backlogAdaptiveDelay.maxExtraDelaySeconds =
        requireScalar<double>(bad, "max_extra_delay_seconds", "common.backlog_adaptive_delay");
    cfg.common.backlogAdaptiveDelay.gainP =
        requireScalar<double>(bad, "gain_p", "common.backlog_adaptive_delay");
    cfg.common.backlogAdaptiveDelay.gainI =
        requireScalar<double>(bad, "gain_i", "common.backlog_adaptive_delay");
    // 取值校验：非法值同样在此处报错（不静默修正）
    if (cfg.common.backlogAdaptiveDelay.targetBacklog < 1.0 ||
        cfg.common.backlogAdaptiveDelay.maxExtraDelaySeconds < 0.0 ||
        cfg.common.backlogAdaptiveDelay.gainP < 0.0 ||
        cfg.common.backlogAdaptiveDelay.gainI < 0.0) {
        throw std::runtime_error("RobotConfig: common.backlog_adaptive_delay."
                                 "target_backlog 必须 >= 1，"
                                 "max_extra_delay_seconds 必须 >= 0，"
                                 "gain_p / gain_i 必须 >= 0");
    }

    // ── common.recording ──
    //   output_dir        ：录制输出目录（相对项目根目录或以 / 开头为绝对路径）
    //   min_free_space_mb ：剩余空间低于该值（MB）时停止写入（0 = 不检查；
    //                       加载时换算为字节存入 minFreeSpaceBytes）
    const YAML::Node& rec = cm["recording"];
    if (!rec || !rec.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'common.recording' 配置段");
    cfg.common.recording.outputDir =
        requireScalar<std::string>(rec, "output_dir", "common.recording");
    cfg.common.recording.minFreeSpaceBytes = static_cast<int64_t>(
        requireScalar<double>(rec, "min_free_space_mb", "common.recording") * 1024.0 * 1024.0);

    // ── common.gimbal ──
    const YAML::Node& gim = cm["gimbal"];
    if (!gim || !gim.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'common.gimbal' 配置段");
    cfg.common.gimbal.bulletDiameter    = requireScalar<double>(gim, "bullet_diameter", "common.gimbal");
    cfg.common.gimbal.bulletMass        = requireScalar<double>(gim, "bullet_mass", "common.gimbal");
    cfg.common.gimbal.bulletVelocity    = requireScalar<double>(gim, "bullet_velocity", "common.gimbal");
    cfg.common.gimbal.integrationStep   = requireScalar<double>(gim, "integration_step", "common.gimbal");
    cfg.common.gimbal.distanceThreshold = requireScalar<double>(gim, "distance_threshold", "common.gimbal");
    cfg.common.gimbal.distanceIterateThreshold = requireScalar<double>(gim, "distance_iterate_threshold", "common.gimbal");
    cfg.common.gimbal.stopZ             = requireScalar<double>(gim, "stop_z", "common.gimbal");
    cfg.common.gimbal.pitchMin          = requireScalar<float>(gim, "pitch_min", "common.gimbal");
    cfg.common.gimbal.pitchMax          = requireScalar<float>(gim, "pitch_max", "common.gimbal");
    cfg.common.gimbal.pitchSearchStep   = requireScalar<float>(gim, "pitch_search_step", "common.gimbal");

    // ── common.predicted_ballistic ──
    const YAML::Node& pb = cm["predicted_ballistic"];
    if (!pb || !pb.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'common.predicted_ballistic' 配置段");
    cfg.common.predictedBallistic.extraPredictTime   = requireScalar<double>(pb, "extra_predict_time", "common.predicted_ballistic");
    cfg.common.predictedBallistic.maxIterations      = requireScalar<int>(pb, "max_iterations", "common.predicted_ballistic");
    cfg.common.predictedBallistic.timeErrorTolerance = requireScalar<double>(pb, "time_error_tolerance", "common.predicted_ballistic");

    // ── common.robot_controller ──
    // 注：原本的 common.tf / common.robot_controller 两段已并入
    //     common.big_small_yaw 的构型分支（single / big_small），此处不再解析。

    // ── common.predict_sequence ──
    const YAML::Node& ic = cm["predict_sequence"];
    if (!ic || !ic.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'common.predict_sequence' 配置段");
    cfg.common.predictSequence.predictionPoints    = requireScalar<int>(ic, "prediction_points", "common.predict_sequence");
    cfg.common.predictSequence.interpolationRefine = requireScalar<int>(ic, "interpolation_refine", "common.predict_sequence");
    cfg.common.predictSequence.exactLeadPoints         = requireScalar<int>(ic, "exact_lead_points", "common.predict_sequence");
    cfg.common.predictSequence.pitchSeqLead        = requireScalar<int>(ic, "pitch_seq_lead", "common.predict_sequence");
    cfg.common.predictSequence.fireSeqLead         = requireScalar<int>(ic, "fire_seq_lead", "common.predict_sequence");
    cfg.common.predictSequence.pitchBias           = requireScalar<double>(ic, "pitch_bias", "common.predict_sequence");
    cfg.common.predictSequence.yawBias             = requireScalar<double>(ic, "yaw_bias", "common.predict_sequence");
    cfg.common.predictSequence.fireAngleLowerLimit = requireScalar<double>(ic, "fire_angle_lower_limit", "common.predict_sequence");
    cfg.common.predictSequence.fireAngleLength     = requireScalar<double>(ic, "fire_angle_length", "common.predict_sequence");
    cfg.common.predictSequence.aimStickRatio       = requireScalar<double>(ic, "aim_stick_ratio", "common.predict_sequence");
    cfg.common.predictSequence.minRotationToleranceAngle = requireScalar<double>(ic, "min_rotation_tolerance_angle", "common.predict_sequence");

    // 交叉校验：预测点数/插值倍数 >= 1；前导精确点数 >= 0；
    // pitch 序列提前数 m 必须小于总返回点数 (M-1)*K+1+n
    if (cfg.common.predictSequence.predictionPoints < 1) {
        throw std::runtime_error("RobotConfig: common.predict_sequence.prediction_points 必须 >= 1");
    }
    if (cfg.common.predictSequence.interpolationRefine < 1) {
        throw std::runtime_error("RobotConfig: common.predict_sequence.interpolation_refine 必须 >= 1");
    }
    if (cfg.common.predictSequence.exactLeadPoints < 0) {
        throw std::runtime_error("RobotConfig: common.predict_sequence.exact_lead_points 必须 >= 0");
    }
    const int total_points = (cfg.common.predictSequence.predictionPoints - 1)
                             * cfg.common.predictSequence.interpolationRefine + 1
                             + cfg.common.predictSequence.exactLeadPoints;
    if (cfg.common.predictSequence.pitchSeqLead < 0 ||
        cfg.common.predictSequence.pitchSeqLead >= total_points) {
        throw std::runtime_error("RobotConfig: common.predict_sequence.pitch_seq_lead 必须小于总返回点数 "
                                 "(prediction_points-1)*interpolation_refine+1+exact_lead_points = " +
                                 std::to_string(total_points));
    }
    if (cfg.common.predictSequence.fireSeqLead < 0) {
        throw std::runtime_error("RobotConfig: common.predict_sequence.fire_seq_lead 必须 >= 0");
    }
    if (cfg.common.predictSequence.aimStickRatio < 0.0) {
        throw std::runtime_error("RobotConfig: common.predict_sequence.aim_stick_ratio 必须 >= 0");
    }
    if (cfg.common.predictSequence.minRotationToleranceAngle < 0.0) {
        throw std::runtime_error("RobotConfig: common.predict_sequence.min_rotation_tolerance_angle 必须 >= 0");
    }

    // ══════════════ armor（独占参数） ══════════════
    const YAML::Node& op = root["armor"];
    if (!op || !op.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'armor' 配置段");
    const YAML::Node& oinf = op["inference"];
    if (!oinf || !oinf.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'armor.inference' 配置段");
    // Older configuration files without a selector keep the 0526 contract.
    cfg.armor.modelName = oinf["model"]
        ? requireScalar<std::string>(oinf, "model", "armor.inference") : "0526";
    if (cfg.armor.modelName != "0526" && cfg.armor.modelName != "0726")
        throw std::runtime_error("RobotConfig: armor.inference.model must be 0526 or 0726");
    const YAML::Node model_cfg = oinf["model"]
        ? oinf["models"][cfg.armor.modelName] : oinf;
    if (!model_cfg || !model_cfg.IsMap())
        throw std::runtime_error("RobotConfig: missing armor.inference.models." + cfg.armor.modelName);
    cfg.armor.modelPath = requireScalar<std::string>(model_cfg, "model_path", "armor.inference selected model");
    cfg.armor.device    = requireScalar<std::string>(oinf, "device", "armor.inference");

    // ── armor.inference.resolution（YOLO 推理输入分辨率）──
    const YAML::Node& ores = model_cfg["resolution"];
    if (!ores || !ores.IsMap())
        throw std::runtime_error("RobotConfig: 缺少 'armor.inference.resolution' 配置段");
    cfg.armor.inputWidth  = requireScalar<int>(ores, "width", "armor.inference.resolution");
    cfg.armor.inputHeight = requireScalar<int>(ores, "height", "armor.inference.resolution");
    if (cfg.armor.inputWidth <= 0 || cfg.armor.inputHeight <= 0) {
        throw std::runtime_error("RobotConfig: armor.inference.resolution 宽高必须为正整数");
    }
    cfg.armor.maxBatch = requireScalar<int>(model_cfg, "max_batch", "armor.inference selected model");
    if (cfg.armor.maxBatch < 1) {
        throw std::runtime_error("RobotConfig: armor.inference.max_batch 必须 >= 1");
    }
    if (cfg.armor.modelName == "0726") {
        if (cfg.armor.inputWidth % 32 != 0 || cfg.armor.inputHeight % 32 != 0)
            throw std::runtime_error("RobotConfig: 0726 width/height must be positive multiples of 32");
        // Reserve enough space for the shared-memory transport's maximum batch.
        const auto input_bytes = 1LL * cfg.armor.inputWidth * cfg.armor.inputHeight * 3 * InferShm::MAX_IMAGES;
        if (input_bytes > static_cast<long long>(InferShm::MAX_INPUT_BYTES))
            throw std::runtime_error("RobotConfig: 0726 resolution exceeds shared-memory input capacity (W*H <= 640*640)");
    }
    cfg.armor.shmKey   = requireScalar<int>(oinf, "shm_key", "armor.inference");
    cfg.armor.observationLostTimeoutSec = requireScalar<double>(op, "observation_lost_timeout", "armor");

    // ── armor.pipeline（缓冲队列长度 + 可批处理阶段批量）──
    const YAML::Node& opipe = op["pipeline"];
    if (!opipe || !opipe.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'armor.pipeline' 配置段");
    parsePipelineParams(opipe, "armor.pipeline", cfg.armor.maxBatch, cfg.armor.pipeline);

    // ── armor.outpost_esekf（OutpostESEKF 滤波参数）──
    const YAML::Node& ek = op["outpost_esekf"];
    if (!ek || !ek.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'armor.outpost_esekf' 配置段");
    cfg.armor.esekf.positionNoise        = requireScalar<double>(ek, "position_noise", "armor.outpost_esekf");
    cfg.armor.esekf.rotationNoise        = requireScalar<double>(ek, "rotation_noise", "armor.outpost_esekf");
    cfg.armor.esekf.measurementNoise     = requireScalar<double>(ek, "measurement_noise", "armor.outpost_esekf");
    cfg.armor.esekf.orientationZRegNoise = requireScalar<double>(ek, "orientation_z_reg_noise", "armor.outpost_esekf");
    cfg.armor.esekf.dzNoise              = requireScalar<double>(ek, "dz_noise", "armor.outpost_esekf");
    cfg.armor.esekf.dzSearchRange        = requireScalar<double>(ek, "dz_search_range", "armor.outpost_esekf");
    cfg.armor.esekf.dzLimit              = requireScalar<double>(ek, "dz_limit", "armor.outpost_esekf");
    cfg.armor.esekf.initPositionNoise    = requireScalar<double>(ek, "init_position_noise", "armor.outpost_esekf");
    cfg.armor.esekf.initOrientationNoise = requireScalar<double>(ek, "init_orientation_noise", "armor.outpost_esekf");
    cfg.armor.esekf.initYawRateNoise     = requireScalar<double>(ek, "init_yaw_rate_noise", "armor.outpost_esekf");
    cfg.armor.esekf.initDz2Noise         = requireScalar<double>(ek, "init_dz2_noise", "armor.outpost_esekf");
    cfg.armor.esekf.initDz3Noise         = requireScalar<double>(ek, "init_dz3_noise", "armor.outpost_esekf");

    // ── armor.super_power_ekf（SuperPower EKF 滤波参数，label 0~5 每类一个 ClassEKF）──
    // 字段与 ClassEKF::Params 一一对应（含原接口 TrackerConfig / PairUpdateConfig /
    // 角速度拟合参数），经 ClassEKF::buildConfig 组装成原接口 YAML 配置节点下发。
    const YAML::Node& spekf = op["super_power_ekf"];
    if (!spekf || !spekf.IsMap())
        throw std::runtime_error("RobotConfig: 缺少 'armor.super_power_ekf' 配置段");
    cfg.armor.superPowerEkf.observationLostTimeoutSec =
        requireScalar<double>(spekf, "observation_lost_timeout_sec", "armor.super_power_ekf");
    cfg.armor.superPowerEkf.initialRadiusM =
        requireScalar<double>(spekf, "initial_radius_m", "armor.super_power_ekf");
    cfg.armor.superPowerEkf.jointUpdateEnabled =
        requireScalar<bool>(spekf, "joint_update_enabled", "armor.super_power_ekf");
    cfg.armor.superPowerEkf.minDetectCount =
        requireScalar<int>(spekf, "min_detect_count", "armor.super_power_ekf");
    cfg.armor.superPowerEkf.maxTempLostCount =
        requireScalar<int>(spekf, "max_temp_lost_count", "armor.super_power_ekf");
    cfg.armor.superPowerEkf.maxDtSec =
        requireScalar<double>(spekf, "max_dt_s", "armor.super_power_ekf");
    cfg.armor.superPowerEkf.armorNum =
        requireScalar<int>(spekf, "armor_num", "armor.super_power_ekf");
    const YAML::Node& ekf_fit = spekf["angular_velocity_fit"];
    if (!ekf_fit || !ekf_fit.IsMap())
        throw std::runtime_error("RobotConfig: 缺少 'armor.super_power_ekf.angular_velocity_fit' 配置段");
    cfg.armor.superPowerEkf.angularVelocityFitWindowSec =
        requireScalar<double>(ekf_fit, "window_s", "armor.super_power_ekf.angular_velocity_fit");
    cfg.armor.superPowerEkf.angularVelocityFitMinSamples =
        requireScalar<int>(ekf_fit, "min_samples", "armor.super_power_ekf.angular_velocity_fit");
    const YAML::Node& ekf_joint = spekf["joint_update"];
    if (!ekf_joint || !ekf_joint.IsMap())
        throw std::runtime_error("RobotConfig: 缺少 'armor.super_power_ekf.joint_update' 配置段");
    cfg.armor.superPowerEkf.jointMaxNis =
        requireScalar<double>(ekf_joint, "max_joint_nis", "armor.super_power_ekf.joint_update");
    cfg.armor.superPowerEkf.jointMaxSecondaryPositionErrorM =
        requireScalar<double>(ekf_joint, "max_secondary_position_error_m", "armor.super_power_ekf.joint_update");
    cfg.armor.superPowerEkf.jointMaxSecondaryAngleErrorRad =
        requireScalar<double>(ekf_joint, "max_secondary_angle_error_rad", "armor.super_power_ekf.joint_update");
    cfg.armor.superPowerEkf.jointMeasurementVarianceScale =
        requireScalar<double>(ekf_joint, "measurement_variance_scale", "armor.super_power_ekf.joint_update");
    cfg.armor.superPowerEkf.jointAngleVarianceScale =
        requireScalar<double>(ekf_joint, "angle_variance_scale", "armor.super_power_ekf.joint_update");
    // 取值校验：非法值同样在此处报错（不静默修正）；下界与原接口
    // SuperPowerPredictor 的内置钳位（window_s ≥ 0.02、min_samples ≥ 2）保持一致。
    if (cfg.armor.superPowerEkf.observationLostTimeoutSec <= 0.0 ||
        cfg.armor.superPowerEkf.initialRadiusM <= 0.0 ||
        cfg.armor.superPowerEkf.minDetectCount < 1 ||
        cfg.armor.superPowerEkf.maxTempLostCount < 0 ||
        cfg.armor.superPowerEkf.maxDtSec <= 0.0 ||
        cfg.armor.superPowerEkf.armorNum < 1 ||
        cfg.armor.superPowerEkf.angularVelocityFitWindowSec < 0.02 ||
        cfg.armor.superPowerEkf.angularVelocityFitMinSamples < 2 ||
        cfg.armor.superPowerEkf.jointMaxNis <= 0.0 ||
        cfg.armor.superPowerEkf.jointMaxSecondaryPositionErrorM <= 0.0 ||
        cfg.armor.superPowerEkf.jointMaxSecondaryAngleErrorRad <= 0.0 ||
        cfg.armor.superPowerEkf.jointMeasurementVarianceScale <= 0.0 ||
        cfg.armor.superPowerEkf.jointAngleVarianceScale <= 0.0) {
        throw std::runtime_error("RobotConfig: armor.super_power_ekf 取值非法："
                                 "observation_lost_timeout_sec / initial_radius_m / max_dt_s 必须 > 0，"
                                 "min_detect_count 必须 >= 1，max_temp_lost_count 必须 >= 0，"
                                 "armor_num 必须 >= 1，angular_velocity_fit.window_s 必须 >= 0.02，"
                                 "angular_velocity_fit.min_samples 必须 >= 2，"
                                 "joint_update 各门控阈值必须 > 0");
    }

    // ── armor.target_selection（目标选取滞回参数）──
    const YAML::Node& sel = op["target_selection"];
    if (!sel || !sel.IsMap())
        throw std::runtime_error("RobotConfig: 缺少 'armor.target_selection' 配置段");
    cfg.armor.targetSelection.stage5StickPriorityM =
        requireScalar<double>(sel, "stage5_stick_priority_m", "armor.target_selection");
    cfg.armor.targetSelection.slowAngularVelocityLower =
        requireScalar<double>(sel, "slow_angular_velocity_lower", "armor.target_selection");
    cfg.armor.targetSelection.slowAngularVelocityUpper =
        requireScalar<double>(sel, "slow_angular_velocity_upper", "armor.target_selection");
    cfg.armor.targetSelection.fastAngularVelocityLower =
        requireScalar<double>(sel, "fast_angular_velocity_lower", "armor.target_selection");
    cfg.armor.targetSelection.fastAngularVelocityUpper =
        requireScalar<double>(sel, "fast_angular_velocity_upper", "armor.target_selection");
    if (cfg.armor.targetSelection.stage5StickPriorityM < 0.0 ||
        cfg.armor.targetSelection.slowAngularVelocityLower < 0.0 ||
        cfg.armor.targetSelection.slowAngularVelocityUpper <=
            cfg.armor.targetSelection.slowAngularVelocityLower ||
        cfg.armor.targetSelection.fastAngularVelocityLower <=
            cfg.armor.targetSelection.slowAngularVelocityUpper ||
        cfg.armor.targetSelection.fastAngularVelocityUpper <=
            cfg.armor.targetSelection.fastAngularVelocityLower) {
        throw std::runtime_error("RobotConfig: armor.target_selection 取值非法："
                                 "stage5_stick_priority_m / slow_angular_velocity_lower 必须 >= 0，"
                                 "slow_angular_velocity_upper 必须 > slow_angular_velocity_lower，"
                                 "fast_angular_velocity_lower 必须 > slow_angular_velocity_upper，"
                                 "fast_angular_velocity_upper 必须 > fast_angular_velocity_lower");
    }

    // ══════════════ power_rune（独占参数） ══════════════
    const YAML::Node& pr = root["power_rune"];
    if (!pr || !pr.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'power_rune' 配置段");
    const YAML::Node& prinf = pr["inference"];
    if (!prinf || !prinf.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'power_rune.inference' 配置段");
    cfg.powerRune.modelPath     = requireScalar<std::string>(prinf, "model_path", "power_rune.inference");
    cfg.powerRune.device        = requireScalar<std::string>(prinf, "device", "power_rune.inference");

    // ── power_rune.inference.resolution（YOLO 推理输入分辨率）──
    const YAML::Node& prres = prinf["resolution"];
    if (!prres || !prres.IsMap())
        throw std::runtime_error("RobotConfig: 缺少 'power_rune.inference.resolution' 配置段");
    cfg.powerRune.inputWidth  = requireScalar<int>(prres, "width", "power_rune.inference.resolution");
    cfg.powerRune.inputHeight = requireScalar<int>(prres, "height", "power_rune.inference.resolution");
    if (cfg.powerRune.inputWidth <= 0 || cfg.powerRune.inputHeight <= 0) {
        throw std::runtime_error("RobotConfig: power_rune.inference.resolution 宽高必须为正整数");
    }
    cfg.powerRune.manualNms     = requireScalar<bool>(prinf, "manual_nms", "power_rune.inference");
    cfg.powerRune.confThreshold = requireScalar<float>(prinf, "conf_threshold", "power_rune.inference");
    cfg.powerRune.maxBatch      = requireScalar<int>(prinf, "max_batch", "power_rune.inference");
    cfg.powerRune.shmKey        = requireScalar<int>(prinf, "shm_key", "power_rune.inference");

    // ── power_rune.pipeline（缓冲队列长度 + 可批处理阶段批量）──
    const YAML::Node& prpipe = pr["pipeline"];
    if (!prpipe || !prpipe.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'power_rune.pipeline' 配置段");
    parsePipelineParams(prpipe, "power_rune.pipeline", cfg.powerRune.maxBatch, cfg.powerRune.pipeline);

    return cfg;
}

// ── 机器配置路径解析（config 读取规则 v2）──
//   1) 读取机器配置选择器 config/selector.yaml 的 active_config 条目；
//   2) 返回 config/robots/<active_config>.yaml 的绝对路径。
// 选择器缺失 / 条目缺失或非法时抛出 std::runtime_error。
std::string resolveMachineConfigPath() {
    const std::string selectorPath = PathResolver::resolvePath("config/selector.yaml");
    YAML::Node selector;
    try {
        selector = YAML::LoadFile(selectorPath);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("RobotConfig: 无法解析机器配置选择器 '" + selectorPath +
                                 "': " + e.what());
    }
    const YAML::Node& nameNode = selector["active_config"];
    if (!nameNode || !nameNode.IsDefined()) {
        throw std::runtime_error("RobotConfig: 机器配置选择器 '" + selectorPath +
                                 "' 缺少 'active_config' 条目（应填写 config/robots/ 下"
                                 "的配置文件名，不含 .yaml 后缀）");
    }
    std::string name;
    try {
        name = nameNode.as<std::string>();
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("RobotConfig: 机器配置选择器 'active_config' 类型错误: " +
                                 std::string(e.what()));
    }
    // 仅允许纯文件名（不含路径分隔符 / ..），避免越出 config/robots/ 目录
    if (name.empty() || name.find('/') != std::string::npos ||
        name.find('\\') != std::string::npos || name.find("..") != std::string::npos) {
        throw std::runtime_error("RobotConfig: 机器配置选择器 'active_config' 非法: '" + name + "'");
    }
    return PathResolver::resolvePath("config/robots/" + name + ".yaml");
}

RobotConfig& RobotConfig::instance() {
    // 懒加载：首次调用时经机器配置选择器 config/selector.yaml 定位当前机器配置文件
    // （config/robots/<active_config>.yaml），规则见 resolveMachineConfigPath()。
    static RobotConfig cfg = load(resolveMachineConfigPath());
    return cfg;
}
