#include "common/Debug/Log.h"
#include "common/Input/IInputMode.h"
#include "common/PathResolver.h"
#include "common/PipelineResult.h"
#include "common/Output/OutputContext.h"

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <thread>

namespace {
using Json = nlohmann::json;
namespace d = diagnostics;
namespace fs = std::filesystem;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Error, typename Function>
void expectError(Function function) {
    bool reported = false;
    try { function(); }
    catch (const Error&) { reported = true; }
    check(reported, "没有报告预期的错误");
}

fs::path sessionDirectory(const fs::path& root) {
    std::vector<fs::path> directories;
    for (const auto& entry : fs::directory_iterator(root))
        if (entry.is_directory()) directories.push_back(entry.path());
    check(directories.size() == 1, "会话目录数量不符");
    return directories.front();
}

Json readJson(const fs::path& path) {
    std::ifstream input(path);
    input.exceptions(std::ios::badbit);
    check(input.is_open(), "无法打开 JSON 文件");
    Json value;
    input >> value;
    return value;
}

std::vector<Json> readRecords(const fs::path& path) {
    std::ifstream input(path);
    input.exceptions(std::ios::badbit);
    check(input.is_open(), "无法打开 JSONL 文件");
    std::vector<Json> rows;
    std::string line;
    while (std::getline(input, line)) rows.push_back(Json::parse(line));
    return rows;
}

void checkStageAdapters(const fs::path& root, Log& log, d::TimePoint now, d::SessionInfo info) {
    if (!log.enabled(d::Stage::Perception) || !log.enabled(d::Stage::Tracking) ||
        !log.enabled(d::Stage::Predict)) return;
    info.output_directory = root / "stage_adapters";
    log.startSession(info);
    PipelineResult result;
    result.valid = true;
    result.armor.valid = true;
    result.frame_timestamp = now;
    result.extra_info.fillCurrentPackZeros(YawMode::SINGLE);
    ExtraInputInfo discarded = result.extra_info;
    const auto discarded_stamp = log.recordInputData(now, discarded);
    const auto stamp = log.recordInputData(now, result.extra_info);
    check(stamp.frame_id != discarded_stamp.frame_id, "相同时间戳的不同图像共用了编号");
    check(result.extra_info.diagnostic_frame_id == stamp.frame_id, "输入编号没有随帧传递");
    ArmorDetect::Object object{};
    object.label = 2;
    object.prob = 0.75f;
    for (std::size_t k = 0; k < 4; ++k) {
        object.landmarks[2 * k] = 10.0f + k;
        object.landmarks[2 * k + 1] = 20.0f + k;
    }
    result.armor.objects = {object, object};
    result.armor.pose_valid = {false, true};
    result.armor.world_positions = {cv::Vec3f(0, 0, 0), cv::Vec3f(1, 2, 3)};
    std::vector<cv::Point2f> projected;
    for (std::size_t k = 0; k < 4; ++k)
        projected.emplace_back(object.landmarks[2 * k] + 3, object.landmarks[2 * k + 1] + 4);
    result.armor.reprojected_points = {projected};
    result.armor.target_valid = true;
    result.armor.target_label = 2;
    result.armor.target_pos = cv::Vec3d(4, 5, 6);
    result.predictor_valid = true;
    result.predictor.timestamp = now - std::chrono::milliseconds(20);
    result.predictor.omega_valid = true;
    result.predictor.target_omega = -1.25;
    log.recordPerceptionData(result);
    log.recordTrackingData(result);
    OutputContext context;
    context.predict_result.valid = true;
    SequencePredictor::Item item{};
    item.success = true;
    item.target_index = SequencePredictor::kFastAimTargetIndex;
    item.predicted_point = cv::Vec3f(7, 8, 9);
    item.predict_time = 0.25;
    item.flight_time = 0.1;
    item.gimbal_yaw = 0.5f;
    item.gimbal_pitch = -0.25f;
    item.yaw = 0.75f;
    item.pitch = -0.125f;
    context.predict_result.items = {item, SequencePredictor::Item{}};
    log.recordPredictData(result, context, now);
    const auto plan_id = context.diagnostic_plan_id;
    check(plan_id.has_value(), "预测编号没有传入输出上下文");
    result.armor.objects[1].landmarks[0] = 999;
    result.armor.target_pos = cv::Vec3d(999, 999, 999);
    context.predict_result.items.clear();

    PipelineResult empty;
    empty.valid = true;
    empty.armor.valid = true;
    empty.extra_info.fillCurrentPackZeros(YawMode::SINGLE);
    empty.frame_timestamp = now;
    log.recordInputData(now, empty.extra_info);
    log.recordPerceptionData(empty);
    log.recordTrackingData(empty);
    OutputContext no_plan;
    log.recordPredictData(empty, no_plan, now);
    check(no_plan.diagnostic_plan_id == *plan_id + 1, "无目标帧没有产生独立预测记录编号");

    PipelineResult rune;
    rune.valid = true;
    rune.power_rune.valid = true;
    rune.extra_info.fillCurrentPackZeros(YawMode::SINGLE);
    rune.frame_timestamp = now;
    log.recordInputData(now, rune.extra_info);
    PoseDetection detection{};
    detection.class_id = 3;
    detection.confidence = 0.5f;
    detection.keypoints = {{10, 20, 1}};
    rune.power_rune.detections.push_back(detection);
    rune.power_rune.pose_valid = true;
    rune.power_rune.pr_world_posi = cv::Vec3f(2, 3, 4);
    rune.power_rune.filtered_R = cv::Mat::eye(3, 3, CV_32F);
    rune.power_rune.filtered_pos = cv::Vec3f(2.5f, 3.5f, 4.5f);
    rune.power_rune.filtered_omega = 8;
    log.recordPerceptionData(rune);
    log.recordTrackingData(rune);
    auto invalid = PipelineResult{};
    invalid.valid = true;
    invalid.armor.valid = true;
    invalid.extra_info.fillCurrentPackZeros(YawMode::SINGLE);
    expectError<std::invalid_argument>([&] { log.recordPerceptionData(invalid); });
    log.finishSession();

    const auto session = sessionDirectory(info.output_directory);
    const auto perceptions = readRecords(session / "perception.jsonl");
    const auto tracks = readRecords(session / "tracking.jsonl");
    const auto plans = readRecords(session / "predict.jsonl");
    check(perceptions.size() == 3 && tracks.size() == 3 && plans.size() == 2,
          "阶段接口写入数量不符");
    for (const auto& row : {perceptions[0], tracks[0], plans[0]})
        check(row.at("stamp").at("frame_id") == *stamp.frame_id, "阶段记录关联了错误图像");
    check(!perceptions[0].contains("started_at_ns") && !plans[0].contains("started_at_ns"),
          "日志填写了业务结果没有提供的阶段开始时间");
    const auto& objects = perceptions[0].at("objects");
    check(objects[0].at("pose_valid") == false && !objects[0].contains("position_world_m"),
          "PnP 失败仍然记录了有效世界位置");
    check(objects[1].at("reprojection_rms_px") == 5.0 && objects[1].at("image_points")[0][0] == 10,
          "重投影计算或检测副本错误");
    check(tracks[0].at("filtered_position_world_m")[0] == 4 && !tracks[0].contains("track_id") &&
          !tracks[0].contains("detection_index"), "滤波状态副本或缺省关联信息错误");
    check(perceptions[1].at("objects").empty() && tracks[1].at("state_valid") == false &&
          plans[1].at("valid") == false && plans[1].at("points").empty(), "无目标帧记录不完整");
    const auto& points = plans[0].at("points");
    check(points.size() == 2 && points[0].at("target_index") == -2 &&
          points[0].at("aim_position_world_m")[0] == 7 && points[1].at("valid") == false &&
          !points[1].contains("aim_position_world_m"), "成功及失败的计划点提取错误");
    check(points[0].at("stamp").at("plan_id") == *plan_id &&
          points[1].at("stamp").at("reference_index") == 1, "计划点关联编号错误");
    const auto& common = RobotConfig::instance().common;
    check(plans[0].at("extra_predict_time_s") == common.predictedBallistic.extraPredictTime + 0.02,
          "实际额外预测时间没有包含预测器快照年龄");
    const auto execution = now + std::chrono::duration_cast<d::TimePoint::duration>(
        std::chrono::duration<double>(common.predictedBallistic.extraPredictTime + common.dtControl()));
    check(points[0].at("execution_timestamp_ns") ==
          std::chrono::duration_cast<std::chrono::nanoseconds>(execution.time_since_epoch()).count(),
          "计划执行时间计算错误");
    check(perceptions[2].at("pipeline") == "power_rune" &&
          tracks[2].at("observed_position_world_m")[0] == 2 &&
          !tracks[2].contains("target_angular_velocity_rad_s"), "能量机关数据提取或角速度语义错误");
}
} // namespace

int main(int argc, char** argv) {
    check(argc == 2 || argc == 3, "需要指定 cache 下的检查输出目录");
    const fs::path root = argv[1];
    fs::create_directories(root);
    auto& log = Log::instance();
    check(!log.sessionActive(), "首次访问日志时不应启动会话");
    const auto now = std::chrono::steady_clock::now();
    d::RecordContext context{};
    context.stamp.frame_id = 0;
    context.pipeline = d::Pipeline::Armor;
    d::InputData input{context, now, 0.125, -0.25};
    d::SessionInfo info{root / "first", "common:\n  diagnostic_test: true\n", "diagnostic-log-test", false};

    if (argc == 3 && (std::string(argv[2]) == "auto-input" || std::string(argv[2]) == "auto-perception")) {
        const fs::path logs = PathResolver::resolvePath("logs");
        std::set<fs::path> previous;
        if (fs::exists(logs))
            for (const auto& entry : fs::directory_iterator(logs)) previous.insert(entry.path());
        ExtraInputInfo pose;
        pose.fillCurrentPackZeros(YawMode::SINGLE);
        pose.single.yaw_pos = 0.375;
        pose.single.pitch_angle = -0.125;
        log.recordInputData(now, pose);
        if (std::string(argv[2]) == "auto-perception") {
            PipelineResult result;
            result.valid = true;
            result.armor.valid = true;
            result.extra_info = pose;
            result.frame_timestamp = now;
            log.recordPerceptionData(result);
        }
        check(log.sessionActive(), "输入记录没有自动开启会话");
        log.finishSession();
        std::vector<fs::path> created;
        for (const auto& entry : fs::directory_iterator(logs))
            if (!previous.count(entry.path())) created.push_back(entry.path());
        check(created.size() == 1, "自动建立的会话数量不符");
        if (log.enabled(d::Stage::Input)) {
            const auto rows = readRecords(created.front() / "input.jsonl");
            check(rows.size() == 1 && rows.front().at("yaw_rad") == 0.375,
                  "自动会话没有记录配对姿态");
        }
        if (std::string(argv[2]) == "auto-perception") {
            const auto rows = readRecords(created.front() / "perception.jsonl");
            check(rows.size() == 1 && rows.front().at("stamp").at("frame_id") == *pose.diagnostic_frame_id,
                  "感知接口自动创建会话后丢失输入编号");
        }
        const auto selector = YAML::LoadFile(PathResolver::resolvePath("config/selector.yaml"));
        const auto config_path = PathResolver::resolvePath(
            "config/robots/" + selector["active_config"].as<std::string>() + ".yaml");
        std::ifstream selected(config_path), saved(created.front() / "effective_config.yaml");
        check(selected.is_open() && saved.is_open(), "无法打开配置快照");
        const std::string expected_config{std::istreambuf_iterator<char>(selected), {}};
        const std::string saved_config{std::istreambuf_iterator<char>(saved), {}};
        check(expected_config == saved_config, "自动会话配置快照不符");
        const auto metadata = readJson(created.front() / "session.json");
        check(metadata.at("status") == "complete", "自动会话没有完整关闭");
        check(metadata.at("infantry2_imu_compensation_enabled") == static_cast<bool>(IS_INFANTRY_2),
              "自动会话的构建补偿标记不符");
        std::cout << "输入接口自动创建会话及配置快照检查通过：" << created.front() << '\n';
        return 0;
    }

    // 在单独进程退出时检查析构排空。
    if (argc == 3) {
        log.startSession(info);
        log.recordInputData(input);
        std::cout << sessionDirectory(info.output_directory) << '\n';
        return 0;
    }
    if (log.enabled(d::Stage::Input)) expectError<std::logic_error>([&] { log.recordInputData(input); });
    log.startSession(info);
    check(log.sessionActive(), "会话没有开启");
    expectError<std::logic_error>([&] { log.startSession(info); });
    log.recordInputData(input);

    d::PerceptionData perception{};
    perception.context = context;
    perception.started_at = now;
    perception.finished_at = now;
    d::PerceptionObject detection{};
    detection.detection_index = 0;
    detection.class_id = 2;
    detection.confidence = 0.95;
    detection.image_points = {{{10.5, 20.5}}, {{30.5, 40.5}}};
    detection.pose_valid = false;
    perception.objects.push_back(detection);
    log.recordPerceptionData(perception);
    perception.objects.front().image_points.front()[0] = 999;

    d::TrackingData tracking{};
    tracking.context = context;
    tracking.track_id = std::numeric_limits<std::uint64_t>::max();
    tracking.class_id = 2;
    tracking.detection_index = 0;
    tracking.state_timestamp = now;
    tracking.state_valid = false;
    tracking.target_angular_velocity_rad_s = 0.0;
    log.recordTrackingData(tracking);

    d::PredictData prediction{};
    prediction.context = context;
    prediction.context.stamp.plan_id = 0;
    prediction.predictor_timestamp = now;
    prediction.started_at = now;
    prediction.finished_at = now;
    prediction.valid = true;
    prediction.selection_reason = "选靶,\"原因\"\n换行";
    const auto reason = prediction.selection_reason;
    d::PredictPoint point{};
    point.stamp = prediction.context.stamp;
    point.stamp.reference_index = 0;
    point.execution_timestamp = now;
    point.valid = false;
    point.fire_allowed = false;
    prediction.points.push_back(point);
    log.recordPredictData(prediction);
    prediction.selection_reason = "已修改";
    prediction.points.clear();

    d::ControlData control{};
    control.context = context;
    control.context.stamp.control_tick_id = 0;
    control.context.stamp.plan_id = 0;
    control.context.stamp.reference_index = 0;
    control.joint = d::Joint::Yaw;
    control.reference_source = d::ReferenceSource::Plan;
    control.started_at = now;
    control.integral_enabled = false;
    control.solution_valid = false;
    control.command_torque_nm = 0.0;
    control.command_send_succeeded = false;
    log.recordControlData(control);

    if (log.enabled(d::Stage::Predict)) {
        auto invalid = prediction;
        invalid.context.stamp.plan_id.reset();
        expectError<std::invalid_argument>([&] { log.recordPredictData(invalid); });
        invalid = prediction;
        invalid.selection_reason.assign(17 * 1024 * 1024, 'x');
        expectError<std::length_error>([&] { log.recordPredictData(invalid); });
    }
    if (log.enabled(d::Stage::Control)) {
        auto invalid = control;
        invalid.measured_position_rad = std::numeric_limits<double>::infinity();
        expectError<std::invalid_argument>([&] { log.recordControlData(invalid); });
        invalid = control;
        invalid.context.stamp.control_tick_id.reset();
        expectError<std::invalid_argument>([&] { log.recordControlData(invalid); });
    }

    std::vector<std::thread> producers;
    for (std::uint64_t producer = 0; producer < 4; ++producer) {
        producers.emplace_back([&, producer] {
            auto data = control;
            for (std::uint64_t i = 0; i < 150; ++i) {
                data.context.stamp.control_tick_id = 1 + producer * 150 + i;
                log.recordControlData(data);
            }
        });
    }
    for (auto& producer : producers) producer.join();
    log.finishSession();
    log.finishSession();
    check(!log.sessionActive(), "会话没有关闭");

    const auto session = sessionDirectory(info.output_directory);
    const auto metadata = readJson(session / "session.json");
    check(metadata.at("status") == "complete", "没有记录会话完整结束状态");
    check(metadata.at("infantry2_imu_compensation_enabled") == false, "丢失了可选布尔值 false");
    const std::array<const char*, 5> names{"input", "perception", "tracking", "predict", "control"};
    const std::array<std::uint64_t, 5> expected{1, 1, 1, 1, 601};
    for (std::size_t i = 0; i < names.size(); ++i) {
        const bool enabled = metadata.at("stages").at(names[i]);
        const fs::path path = session / (std::string(names[i]) + ".jsonl");
        check(fs::exists(path) == enabled, "阶段开关与输出文件不符");
        const auto count = enabled ? expected[i] : 0;
        check(metadata.at("records").at(names[i]).at("accepted") == count, "接收计数不符");
        check(metadata.at("records").at(names[i]).at("written") == count, "写入计数不符");
        if (!enabled) continue;
        const auto rows = readRecords(path);
        check(rows.size() == count, "输出记录数量不符");
        check(rows.front().at("stamp").at("frame_id") == 0, "编号 0 未保留");
        check(rows.front().at("written_at_ns").is_number_integer(), "写入时间没有使用整数纳秒");
        if (i == 0) check(rows.front().at("yaw_rad") == 0.125, "输入角度写入错误");
        if (i == 1) check(rows.front().at("objects")[0].at("image_points")[0][0] == 10.5, "未持有感知数据副本");
        if (i == 2) {
            check(rows.front().at("track_id") == std::numeric_limits<std::uint64_t>::max(), "64 位编号精度丢失");
            check(!rows.front().contains("filtered_position_world_m"), "未提供的可选字段仍被写入");
            check(rows.front().at("target_angular_velocity_rad_s") == 0.0, "可选数值 0 丢失");
        }
        if (i == 3) {
            check(rows.front().at("selection_reason") == reason, "字符串转义或数据副本错误");
            check(rows.front().at("points").size() == 1, "未持有计划数组副本");
            check(rows.front().at("points")[0].at("fire_allowed") == false, "可选 false 丢失");
        }
        if (i == 4) {
            std::set<std::uint64_t> ids;
            for (const auto& row : rows)
                ids.insert(row.at("stamp").at("control_tick_id").get<std::uint64_t>());
            check(ids.size() == count, "并发记录发生重复或丢失");
            check(!rows.front().contains("measured_torque_nm"), "缺失的实际力矩被写入");
            check(rows.front().at("command_torque_nm") == 0.0, "零指令力矩丢失");
        }
    }

    info.output_directory = root / "second";
    log.startSession(info);
    log.finishSession();
    check(readJson(sessionDirectory(info.output_directory) / "session.json").at("status") == "complete",
          "第二次会话未正常关闭");
    if (log.enabled(d::Stage::Input)) {
        info.output_directory = root / "input_adapter";
        log.startSession(info);
        ExtraInputInfo pose;
        pose.fillCurrentPackZeros(YawMode::SINGLE);
        pose.single.yaw_pos = 0.375;
        pose.single.pitch_angle = -0.125;
        const auto first_stamp = log.recordInputData(now, pose);
        pose.single.yaw_pos = -0.5;
        pose.single.pitch_angle = 0.25;
        const auto later = now + std::chrono::milliseconds(10);
        const auto second_stamp = log.recordInputData(later, pose);
        pose.single.yaw_pos = std::numeric_limits<double>::quiet_NaN();
        expectError<std::invalid_argument>([&] { log.recordInputData(later, pose); });
        ExtraInputInfo big_small;
        big_small.fillCurrentPackZeros(YawMode::BIG_SMALL);
        log.recordInputData(later, big_small);
        log.finishSession();
        const auto rows = readRecords(sessionDirectory(info.output_directory) / "input.jsonl");
        check(rows.size() == 2, "输入接口记录数量不符");
        check(first_stamp.frame_id == 0 && second_stamp.frame_id == 1, "输入编号没有递增");
        check(rows[0].at("yaw_rad") == 0.375 && rows[0].at("pitch_rad") == -0.125,
              "首幅图像的配对姿态被后续修改影响");
        check(rows[1].at("yaw_rad") == -0.5 && rows[1].at("pitch_rad") == 0.25,
              "第二幅图像的配对姿态不符");
        check(rows[0].at("frame_timestamp_ns") ==
                  std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() &&
              rows[1].at("frame_timestamp_ns") ==
                  std::chrono::duration_cast<std::chrono::nanoseconds>(later.time_since_epoch()).count(),
              "输入接口改变了帧时间戳");
    }
    checkStageAdapters(root, log, now, info);
    info.output_directory = root / "regular_file";
    { std::ofstream file(info.output_directory); file << "file"; }
    expectError<fs::filesystem_error>([&] { log.startSession(info); });
    std::cout << "五阶段输出、可选字段、并发记录、会话排空及错误检查通过：" << session << '\n';
}
