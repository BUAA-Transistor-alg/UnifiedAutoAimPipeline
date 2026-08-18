// RobotConfig.cpp — 从 yaml 配置文件加载全局参数
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

    // ── tf 偏移 ──
    const YAML::Node& tf = root["tf"];
    if (!tf || !tf.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'tf' 配置段");
    cfg.tf.yawJointZOffset   = requireScalar<float>(tf, "yaw_joint_z_offset", "tf");
    cfg.tf.pitchJointYOffset = requireScalar<float>(tf, "pitch_joint_y_offset", "tf");
    cfg.tf.imuOffsetX        = requireScalar<float>(tf, "imu_offset_x", "tf");
    cfg.tf.imuOffsetY        = requireScalar<float>(tf, "imu_offset_y", "tf");
    cfg.tf.imuOffsetZ        = requireScalar<float>(tf, "imu_offset_z", "tf");
    cfg.tf.cameraOffsetX     = requireScalar<float>(tf, "camera_offset_x", "tf");
    cfg.tf.cameraOffsetY     = requireScalar<float>(tf, "camera_offset_y", "tf");
    cfg.tf.cameraOffsetZ     = requireScalar<float>(tf, "camera_offset_z", "tf");
    cfg.tf.muzzleOffsetX     = requireScalar<float>(tf, "muzzle_offset_x", "tf");
    cfg.tf.muzzleOffsetY     = requireScalar<float>(tf, "muzzle_offset_y", "tf");
    cfg.tf.muzzleOffsetZ     = requireScalar<float>(tf, "muzzle_offset_z", "tf");

    // ── camera ──
    const YAML::Node& cam = root["camera"];
    if (!cam || !cam.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'camera' 配置段");
    cfg.camera.deviceIp = requireScalar<std::string>(cam, "device_ip", "camera");
    cfg.camera.netIp    = requireScalar<std::string>(cam, "net_ip", "camera");
    cfg.camera.exposure = requireScalar<float>(cam, "exposure", "camera");
    cfg.camera.gain     = requireScalar<float>(cam, "gain", "camera");
    const YAML::Node& res = cam["resolution"];
    if (!res || !res.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'camera.resolution' 配置段");
    cfg.camera.width  = requireScalar<int>(res, "width", "camera.resolution");
    cfg.camera.height = requireScalar<int>(res, "height", "camera.resolution");
    std::vector<double> cm = requireList(cam, "camera_matrix", "camera", 9);
    std::vector<double> dc = requireList(cam, "dist_coeffs", "camera", 5);
    cfg.camera.cameraMatrix = (cv::Mat_<double>(3, 3) <<
        cm[0], cm[1], cm[2], cm[3], cm[4], cm[5], cm[6], cm[7], cm[8]);
    cfg.camera.distCoeffs.create((int)dc.size(), 1, CV_64F);
    for (size_t i = 0; i < dc.size(); ++i) {
        cfg.camera.distCoeffs.at<double>((int)i, 0) = dc[i];
    }

    // ── inference ──
    const YAML::Node& inf = root["inference"];
    if (!inf || !inf.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'inference' 配置段");
    cfg.modelPath       = requireScalar<std::string>(inf, "model_path", "inference");
    cfg.inferenceDevice = requireScalar<std::string>(inf, "device", "inference");

    // ── gimbal ──
    const YAML::Node& gim = root["gimbal"];
    if (!gim || !gim.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'gimbal' 配置段");
    cfg.gimbal.bulletDiameter    = requireScalar<double>(gim, "bullet_diameter", "gimbal");
    cfg.gimbal.bulletMass        = requireScalar<double>(gim, "bullet_mass", "gimbal");
    cfg.gimbal.bulletVelocity    = requireScalar<double>(gim, "bullet_velocity", "gimbal");
    cfg.gimbal.integrationStep   = requireScalar<double>(gim, "integration_step", "gimbal");
    cfg.gimbal.distanceThreshold       = requireScalar<double>(gim, "distance_threshold", "gimbal");
    cfg.gimbal.distanceIterateThreshold = requireScalar<double>(gim, "distance_iterate_threshold", "gimbal");
    cfg.gimbal.stopZ                   = requireScalar<double>(gim, "stop_z", "gimbal");
    cfg.gimbal.pitchMin          = requireScalar<float>(gim, "pitch_min", "gimbal");
    cfg.gimbal.pitchMax          = requireScalar<float>(gim, "pitch_max", "gimbal");
    cfg.gimbal.pitchSearchStep   = requireScalar<float>(gim, "pitch_search_step", "gimbal");

    // ── predicted_ballistic ──
    const YAML::Node& pb = root["predicted_ballistic"];
    if (!pb || !pb.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'predicted_ballistic' 配置段");
    cfg.predictedBallistic.extraPredictTime   = requireScalar<double>(pb, "extra_predict_time", "predicted_ballistic");
    cfg.predictedBallistic.maxIterations      = requireScalar<int>(pb, "max_iterations", "predicted_ballistic");
    cfg.predictedBallistic.timeErrorTolerance = requireScalar<double>(pb, "time_error_tolerance", "predicted_ballistic");

    // ── robot_controller ──
    const YAML::Node& rc = root["robot_controller"];
    if (!rc || !rc.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'robot_controller' 配置段");
    cfg.robotController.sequenceMode  = requireScalar<bool>(rc, "sequence_mode", "robot_controller");
    cfg.robotController.dtControl     = requireScalar<double>(rc, "dt_control", "robot_controller");
    cfg.robotController.mpcPredN      = requireScalar<int>(rc, "mpc_pred_n", "robot_controller");
    cfg.robotController.J             = requireScalar<double>(rc, "J", "robot_controller");
    cfg.robotController.tauC          = requireScalar<double>(rc, "tau_c", "robot_controller");
    cfg.robotController.b             = requireScalar<double>(rc, "b", "robot_controller");
    cfg.robotController.tauD          = requireScalar<double>(rc, "tau_d", "robot_controller");
    cfg.robotController.maxTorque     = requireScalar<double>(rc, "max_torque", "robot_controller");
    cfg.robotController.maxTorqueRate = requireScalar<double>(rc, "max_torque_rate", "robot_controller");
    cfg.robotController.Q             = requireScalar<double>(rc, "Q", "robot_controller");
    cfg.robotController.R             = requireScalar<double>(rc, "R", "robot_controller");
    cfg.robotController.Rd            = requireScalar<double>(rc, "Rd", "robot_controller");
    cfg.robotController.maxIter       = requireScalar<int>(rc, "max_iter", "robot_controller");

    // ── input_controller ──
    const YAML::Node& ic = root["input_controller"];
    if (!ic || !ic.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'input_controller' 配置段");
    cfg.inputController.predictionSeqLen   = requireScalar<int>(ic, "prediction_seq_len", "input_controller");
    cfg.inputController.pitchSeqLead       = requireScalar<int>(ic, "pitch_seq_lead", "input_controller");
    cfg.inputController.fireSeqLead        = requireScalar<int>(ic, "fire_seq_lead", "input_controller");
    cfg.inputController.pitchBias          = requireScalar<double>(ic, "pitch_bias", "input_controller");
    cfg.inputController.fireAngleLowerLimit = requireScalar<double>(ic, "fire_angle_lower_limit", "input_controller");
    cfg.inputController.fireAngleLength    = requireScalar<double>(ic, "fire_angle_length", "input_controller");

    // 交叉校验：pitch 序列提前数 m 必须小于预测序列长度 n
    if (cfg.inputController.predictionSeqLen < 1) {
        throw std::runtime_error("RobotConfig: input_controller.prediction_seq_len 必须 >= 1");
    }
    if (cfg.inputController.pitchSeqLead < 0 ||
        cfg.inputController.pitchSeqLead >= cfg.inputController.predictionSeqLen) {
        throw std::runtime_error("RobotConfig: input_controller.pitch_seq_lead 必须小于 prediction_seq_len");
    }
    if (cfg.inputController.fireSeqLead < 0) {
        throw std::runtime_error("RobotConfig: input_controller.fire_seq_lead 必须 >= 0");
    }

    // ── power_rune（能量机关）推理与相机参数 ──
    const YAML::Node& pr = root["power_rune"];
    if (!pr || !pr.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'power_rune' 配置段");
    cfg.powerRune.modelPath     = requireScalar<std::string>(pr, "model_path", "power_rune");
    cfg.powerRune.device        = requireScalar<std::string>(pr, "device", "power_rune");
    cfg.powerRune.manualNms     = requireScalar<bool>(pr, "manual_nms", "power_rune");
    cfg.powerRune.confThreshold = requireScalar<float>(pr, "conf_threshold", "power_rune");
    cfg.powerRune.maxBatch      = requireScalar<int>(pr, "max_batch", "power_rune");
    const YAML::Node& pres = pr["resolution"];
    if (!pres || !pres.IsMap()) throw std::runtime_error("RobotConfig: 缺少 'power_rune.resolution' 配置段");
    cfg.powerRune.width  = requireScalar<int>(pres, "width", "power_rune.resolution");
    cfg.powerRune.height = requireScalar<int>(pres, "height", "power_rune.resolution");
    std::vector<double> pcm = requireList(pr, "camera_matrix", "power_rune", 9);
    std::vector<double> pdc = requireList(pr, "dist_coeffs", "power_rune", 5);
    cfg.powerRune.cameraMatrix = (cv::Mat_<double>(3, 3) <<
        pcm[0], pcm[1], pcm[2], pcm[3], pcm[4], pcm[5], pcm[6], pcm[7], pcm[8]);
    cfg.powerRune.distCoeffs.create((int)pdc.size(), 1, CV_64F);
    for (size_t i = 0; i < pdc.size(); ++i) {
        cfg.powerRune.distCoeffs.at<double>((int)i, 0) = pdc[i];
    }

    // ── 其他 ──
    cfg.observationLostTimeoutSec = requireScalar<double>(root, "observation_lost_timeout", "");

    return cfg;
}

RobotConfig& RobotConfig::instance() {
    // 懒加载：首次调用时自动读取项目根目录 config/config.yaml
    static RobotConfig cfg = load(PathResolver::resolvePath("config/config.yaml"));
    return cfg;
}
