// RobotConfig.cpp — 从 yaml 配置文件加载全局参数（common / outpost / power_rune 三结构）
#include "RobotConfig.h"

#include <stdexcept>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "PathResolver.h"

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

// 解析一套相机参数：resolution / camera_matrix / dist_coeffs 必填；
// device_ip / net_ip / exposure / gain 可选（视频模式无）。
void parseCameraParams(const YAML::Node& camNode, const std::string& name,
                       RobotConfig::CameraParams& out) {
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
    if (camNode["device_ip"] && camNode["device_ip"].IsDefined())
        out.deviceIp = camNode["device_ip"].as<std::string>();
    if (camNode["net_ip"] && camNode["net_ip"].IsDefined())
        out.netIp = camNode["net_ip"].as<std::string>();
    if (camNode["exposure"] && camNode["exposure"].IsDefined())
        out.exposure = camNode["exposure"].as<float>();
    if (camNode["gain"] && camNode["gain"].IsDefined())
        out.gain = camNode["gain"].as<float>();
}

} // namespace

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

    // ── common.camera_mode / common.video_mode（按输入模式自动选择）──
    const YAML::Node& cam = cm["camera_mode"];
    if (!cam || !cam.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'common.camera_mode' 配置段");
    parseCameraParams(cam, "common.camera_mode", cfg.common.cameraMode);
    const YAML::Node& vid = cm["video_mode"];
    if (!vid || !vid.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'common.video_mode' 配置段");
    parseCameraParams(vid, "common.video_mode", cfg.common.videoMode);

    // ── common.max_delay_seconds（两个流水线共用）──
    cfg.common.maxDelaySeconds = requireScalar<double>(cm, "max_delay_seconds", "common");

    // ── common.recording（可选：录制参数；缺失时使用默认值）──
    //   output_dir        ：录制输出目录（相对项目根目录或以 / 开头为绝对路径）
    //   min_free_space_mb ：剩余空间低于该值（MB）时停止写入
    cfg.common.recording.outputDir = "recordings";
    cfg.common.recording.minFreeSpaceBytes = 1024LL * 1024 * 1024;   // 默认 1 GiB
    const YAML::Node& rec = cm["recording"];
    if (rec && rec.IsMap()) {
        if (rec["output_dir"] && rec["output_dir"].IsDefined())
            cfg.common.recording.outputDir = rec["output_dir"].as<std::string>();
        if (rec["min_free_space_mb"] && rec["min_free_space_mb"].IsDefined())
            cfg.common.recording.minFreeSpaceBytes = static_cast<int64_t>(
                rec["min_free_space_mb"].as<double>() * 1024.0 * 1024.0);
    }

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
    cfg.common.robotController.sendPitchScale  = requireScalar<double>(rc, "send_pitch_scale", "common.robot_controller");
    cfg.common.robotController.sendPitchOffset = requireScalar<double>(rc, "send_pitch_offset", "common.robot_controller");
    cfg.common.robotController.recvPitchScale  = requireScalar<double>(rc, "recv_pitch_scale", "common.robot_controller");
    cfg.common.robotController.recvPitchOffset = requireScalar<double>(rc, "recv_pitch_offset", "common.robot_controller");

    // ── common.input_controller ──
    const YAML::Node& ic = cm["input_controller"];
    if (!ic || !ic.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'common.input_controller' 配置段");
    cfg.common.inputController.predictionPoints    = requireScalar<int>(ic, "prediction_points", "common.input_controller");
    cfg.common.inputController.interpolationRefine = requireScalar<int>(ic, "interpolation_refine", "common.input_controller");
    cfg.common.inputController.pitchSeqLead        = requireScalar<int>(ic, "pitch_seq_lead", "common.input_controller");
    cfg.common.inputController.fireSeqLead         = requireScalar<int>(ic, "fire_seq_lead", "common.input_controller");
    cfg.common.inputController.pitchBias           = requireScalar<double>(ic, "pitch_bias", "common.input_controller");
    cfg.common.inputController.yawBias             = requireScalar<double>(ic, "yaw_bias", "common.input_controller");
    cfg.common.inputController.fireAngleLowerLimit = requireScalar<double>(ic, "fire_angle_lower_limit", "common.input_controller");
    cfg.common.inputController.fireAngleLength     = requireScalar<double>(ic, "fire_angle_length", "common.input_controller");

    // 交叉校验：预测点数/插值倍数 >= 1；pitch 序列提前数 m 必须小于总返回点数 (M-1)*K+1
    if (cfg.common.inputController.predictionPoints < 1) {
        throw std::runtime_error("RobotConfig: common.input_controller.prediction_points 必须 >= 1");
    }
    if (cfg.common.inputController.interpolationRefine < 1) {
        throw std::runtime_error("RobotConfig: common.input_controller.interpolation_refine 必须 >= 1");
    }
    const int total_points = (cfg.common.inputController.predictionPoints - 1)
                             * cfg.common.inputController.interpolationRefine + 1;
    if (cfg.common.inputController.pitchSeqLead < 0 ||
        cfg.common.inputController.pitchSeqLead >= total_points) {
        throw std::runtime_error("RobotConfig: common.input_controller.pitch_seq_lead 必须小于总返回点数 "
                                 "(prediction_points-1)*interpolation_refine+1 = " +
                                 std::to_string(total_points));
    }
    if (cfg.common.inputController.fireSeqLead < 0) {
        throw std::runtime_error("RobotConfig: common.input_controller.fire_seq_lead 必须 >= 0");
    }

    // ══════════════ outpost（独占参数） ══════════════
    const YAML::Node& op = root["outpost"];
    if (!op || !op.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'outpost' 配置段");
    const YAML::Node& oinf = op["inference"];
    if (!oinf || !oinf.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'outpost.inference' 配置段");
    cfg.outpost.modelPath = requireScalar<std::string>(oinf, "model_path", "outpost.inference");
    cfg.outpost.device    = requireScalar<std::string>(oinf, "device", "outpost.inference");
    cfg.outpost.observationLostTimeoutSec = requireScalar<double>(op, "observation_lost_timeout", "outpost");

    // ══════════════ power_rune（独占参数） ══════════════
    const YAML::Node& pr = root["power_rune"];
    if (!pr || !pr.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'power_rune' 配置段");
    const YAML::Node& prinf = pr["inference"];
    if (!prinf || !prinf.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'power_rune.inference' 配置段");
    cfg.powerRune.modelPath     = requireScalar<std::string>(prinf, "model_path", "power_rune.inference");
    cfg.powerRune.device        = requireScalar<std::string>(prinf, "device", "power_rune.inference");
    cfg.powerRune.manualNms     = requireScalar<bool>(prinf, "manual_nms", "power_rune.inference");
    cfg.powerRune.confThreshold = requireScalar<float>(prinf, "conf_threshold", "power_rune.inference");
    cfg.powerRune.maxBatch      = requireScalar<int>(prinf, "max_batch", "power_rune.inference");

    return cfg;
}

RobotConfig& RobotConfig::instance() {
    // 懒加载：首次调用时自动读取项目根目录 config/config.yaml
    static RobotConfig cfg = load(PathResolver::resolvePath("config/config.yaml"));
    return cfg;
}
