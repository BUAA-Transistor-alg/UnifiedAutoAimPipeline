// RobotConfig.cpp — 从 yaml 配置文件加载全局参数（common / outpost / power_rune 三结构）
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
// ⚠ 不设任何默认值：每个模式段中的所有参数都必须显式出现在 config.yaml，
//   缺字段即抛异常。
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

// 解析流水线缓冲队列与批量参数段（outpost.pipeline / power_rune.pipeline）：
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


// ⚠ 给后续修改者：所有参数均无代码默认值，必须由 config.yaml 提供；
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

    // ── common.backlog_adaptive_delay（队列积压自适应额外延迟，v6 PID 式 PI）──
    // 开启后（见 BacklogAdaptiveDelay）：
    //   - 以 target_backlog 为目标做 PI 直接输出：extra_delay = gain_p*e +
    //     ∫gain_i*e·dt（增益单位分别为 秒/单位积压、秒/单位积压/秒，内部换算
    //     为 µs；e 为积压总数与目标的差，无 EMA 平滑），钳位
    //     [0, max_extra_delay_seconds]（抗饱和，无速率上限）。稳态时积压
    //     收敛到 target_backlog，误差归零。
    //   若要在 config.yaml 中新增本功能的参数，需同步修改
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

    // ══════════════ outpost（独占参数） ══════════════
    const YAML::Node& op = root["outpost"];
    if (!op || !op.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'outpost' 配置段");
    const YAML::Node& oinf = op["inference"];
    if (!oinf || !oinf.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'outpost.inference' 配置段");
    cfg.outpost.modelPath = requireScalar<std::string>(oinf, "model_path", "outpost.inference");
    cfg.outpost.device    = requireScalar<std::string>(oinf, "device", "outpost.inference");

    // ── outpost.inference.resolution（YOLO 推理输入分辨率）──
    const YAML::Node& ores = oinf["resolution"];
    if (!ores || !ores.IsMap())
        throw std::runtime_error("RobotConfig: 缺少 'outpost.inference.resolution' 配置段");
    cfg.outpost.inputWidth  = requireScalar<int>(ores, "width", "outpost.inference.resolution");
    cfg.outpost.inputHeight = requireScalar<int>(ores, "height", "outpost.inference.resolution");
    if (cfg.outpost.inputWidth <= 0 || cfg.outpost.inputHeight <= 0) {
        throw std::runtime_error("RobotConfig: outpost.inference.resolution 宽高必须为正整数");
    }
    cfg.outpost.maxBatch = requireScalar<int>(oinf, "max_batch", "outpost.inference");
    if (cfg.outpost.maxBatch < 1) {
        throw std::runtime_error("RobotConfig: outpost.inference.max_batch 必须 >= 1");
    }
    cfg.outpost.shmKey   = requireScalar<int>(oinf, "shm_key", "outpost.inference");
    cfg.outpost.observationLostTimeoutSec = requireScalar<double>(op, "observation_lost_timeout", "outpost");

    // ── outpost.pipeline（缓冲队列长度 + 可批处理阶段批量）──
    const YAML::Node& opipe = op["pipeline"];
    if (!opipe || !opipe.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'outpost.pipeline' 配置段");
    parsePipelineParams(opipe, "outpost.pipeline", cfg.outpost.maxBatch, cfg.outpost.pipeline);

    // ── outpost.esekf（ESEKF 滤波参数）──
    const YAML::Node& ek = op["esekf"];
    if (!ek || !ek.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'outpost.esekf' 配置段");
    cfg.outpost.esekf.positionNoise        = requireScalar<double>(ek, "position_noise", "outpost.esekf");
    cfg.outpost.esekf.rotationNoise        = requireScalar<double>(ek, "rotation_noise", "outpost.esekf");
    cfg.outpost.esekf.measurementNoise     = requireScalar<double>(ek, "measurement_noise", "outpost.esekf");
    cfg.outpost.esekf.orientationZRegNoise = requireScalar<double>(ek, "orientation_z_reg_noise", "outpost.esekf");
    cfg.outpost.esekf.dzNoise              = requireScalar<double>(ek, "dz_noise", "outpost.esekf");
    cfg.outpost.esekf.dzSearchRange        = requireScalar<double>(ek, "dz_search_range", "outpost.esekf");
    cfg.outpost.esekf.dzLimit              = requireScalar<double>(ek, "dz_limit", "outpost.esekf");
    cfg.outpost.esekf.initPositionNoise    = requireScalar<double>(ek, "init_position_noise", "outpost.esekf");
    cfg.outpost.esekf.initOrientationNoise = requireScalar<double>(ek, "init_orientation_noise", "outpost.esekf");
    cfg.outpost.esekf.initYawRateNoise     = requireScalar<double>(ek, "init_yaw_rate_noise", "outpost.esekf");
    cfg.outpost.esekf.initDz2Noise         = requireScalar<double>(ek, "init_dz2_noise", "outpost.esekf");
    cfg.outpost.esekf.initDz3Noise         = requireScalar<double>(ek, "init_dz3_noise", "outpost.esekf");

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

RobotConfig& RobotConfig::instance() {
    // 懒加载：首次调用时自动读取项目根目录 config/config.yaml
    static RobotConfig cfg = load(PathResolver::resolvePath("config/config.yaml"));
    return cfg;
}
