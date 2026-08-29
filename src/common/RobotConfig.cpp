// RobotConfig.cpp — 从 yaml 配置文件加载全局参数（common / armor / power_rune 三结构）
#include "common/RobotConfig.h"

#include <stdexcept>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "common/PathResolver.h"

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

    // ── common.tf 偏移 ──
    const YAML::Node& tf = cm["tf"];
    if (!tf || !tf.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'common.tf' 配置段");
    cfg.common.tf.yawJointZOffset   = requireScalar<float>(tf, "yaw_joint_z_offset", "common.tf");
    cfg.common.tf.pitchJointYOffset = requireScalar<float>(tf, "pitch_joint_y_offset", "common.tf");
    cfg.common.tf.imuOffsetX        = requireScalar<float>(tf, "imu_offset_x", "common.tf");
    cfg.common.tf.imuOffsetY        = requireScalar<float>(tf, "imu_offset_y", "common.tf");
    cfg.common.tf.imuOffsetZ        = requireScalar<float>(tf, "imu_offset_z", "common.tf");
    cfg.common.tf.cameraOffsetX     = requireScalar<float>(tf, "camera_offset_x", "common.tf");
    cfg.common.tf.cameraOffsetY     = requireScalar<float>(tf, "camera_offset_y", "common.tf");
    cfg.common.tf.cameraOffsetZ     = requireScalar<float>(tf, "camera_offset_z", "common.tf");
    cfg.common.tf.muzzleOffsetX     = requireScalar<float>(tf, "muzzle_offset_x", "common.tf");
    cfg.common.tf.muzzleOffsetY     = requireScalar<float>(tf, "muzzle_offset_y", "common.tf");
    cfg.common.tf.muzzleOffsetZ     = requireScalar<float>(tf, "muzzle_offset_z", "common.tf");

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
    const YAML::Node& rc = cm["robot_controller"];
    if (!rc || !rc.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'common.robot_controller' 配置段");
    cfg.common.robotController.sequenceMode  = requireScalar<bool>(rc, "sequence_mode", "common.robot_controller");
    cfg.common.robotController.dtControl     = requireScalar<double>(rc, "dt_control", "common.robot_controller");
    cfg.common.robotController.mpcPredN      = requireScalar<int>(rc, "mpc_pred_n", "common.robot_controller");
    cfg.common.robotController.J             = requireScalar<double>(rc, "J", "common.robot_controller");
    cfg.common.robotController.tauC          = requireScalar<double>(rc, "tau_c", "common.robot_controller");
    cfg.common.robotController.b             = requireScalar<double>(rc, "b", "common.robot_controller");
    cfg.common.robotController.tauD          = requireScalar<double>(rc, "tau_d", "common.robot_controller");
    cfg.common.robotController.maxTorque     = requireScalar<double>(rc, "max_torque", "common.robot_controller");
    cfg.common.robotController.maxTorqueRate = requireScalar<double>(rc, "max_torque_rate", "common.robot_controller");
    cfg.common.robotController.Q             = requireScalar<double>(rc, "Q", "common.robot_controller");
    cfg.common.robotController.R             = requireScalar<double>(rc, "R", "common.robot_controller");
    cfg.common.robotController.Rd            = requireScalar<double>(rc, "Rd", "common.robot_controller");
    cfg.common.robotController.maxIter       = requireScalar<int>(rc, "max_iter", "common.robot_controller");
    cfg.common.robotController.integralGain  = requireScalar<double>(rc, "integral_gain", "common.robot_controller");
    cfg.common.robotController.sendPitchScale  = requireScalar<double>(rc, "send_pitch_scale", "common.robot_controller");
    cfg.common.robotController.sendPitchOffset = requireScalar<double>(rc, "send_pitch_offset", "common.robot_controller");
    cfg.common.robotController.recvPitchScale  = requireScalar<double>(rc, "recv_pitch_scale", "common.robot_controller");
    cfg.common.robotController.recvPitchOffset = requireScalar<double>(rc, "recv_pitch_offset", "common.robot_controller");

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

    // ══════════════ armor（独占参数） ══════════════
    const YAML::Node& op = root["armor"];
    if (!op || !op.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'armor' 配置段");
    const YAML::Node& oinf = op["inference"];
    if (!oinf || !oinf.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'armor.inference' 配置段");
    cfg.armor.modelPath = requireScalar<std::string>(oinf, "model_path", "armor.inference");
    cfg.armor.device    = requireScalar<std::string>(oinf, "device", "armor.inference");

    // ── armor.inference.resolution（YOLO 推理输入分辨率）──
    const YAML::Node& ores = oinf["resolution"];
    if (!ores || !ores.IsMap())
        throw std::runtime_error("RobotConfig: 缺少 'armor.inference.resolution' 配置段");
    cfg.armor.inputWidth  = requireScalar<int>(ores, "width", "armor.inference.resolution");
    cfg.armor.inputHeight = requireScalar<int>(ores, "height", "armor.inference.resolution");
    if (cfg.armor.inputWidth <= 0 || cfg.armor.inputHeight <= 0) {
        throw std::runtime_error("RobotConfig: armor.inference.resolution 宽高必须为正整数");
    }
    cfg.armor.maxBatch = requireScalar<int>(oinf, "max_batch", "armor.inference");
    if (cfg.armor.maxBatch < 1) {
        throw std::runtime_error("RobotConfig: armor.inference.max_batch 必须 >= 1");
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
