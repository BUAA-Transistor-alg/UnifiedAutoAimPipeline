#include "common/Debug/Log.h"
#include "common/Input/IInputMode.h"
#include "common/PathResolver.h"
#include "common/PipelineResult.h"
#include "common/Output/OutputContext.h"

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>

namespace {
namespace d = diagnostics;
using Json = nlohmann::json;

// 分阶段记录开关，修改后重新编译生效。
constexpr bool kLogInput = true;
constexpr bool kLogPerception = true;
constexpr bool kLogTracking = true;
constexpr bool kLogPredict = true;
constexpr bool kLogControl = true;
constexpr std::array<bool, 5> kStages{
    kLogInput, kLogPerception, kLogTracking, kLogPredict, kLogControl};
constexpr std::array<const char*, 5> kStageNames{
    "input", "perception", "tracking", "predict", "control"};
constexpr std::size_t kQueueCapacity = 4096;
constexpr std::size_t kQueueBytes = 16 * 1024 * 1024;
using Payload = std::variant<d::InputData, d::PerceptionData, d::TrackingData,
                             d::PredictData, d::ControlData>;

std::size_t stageIndex(d::Stage stage) {
    switch (stage) {
        case d::Stage::Input: return 0;
        case d::Stage::Perception: return 1;
        case d::Stage::Tracking: return 2;
        case d::Stage::Predict: return 3;
        case d::Stage::Control: return 4;
    }
    throw std::invalid_argument("Log: 无效诊断阶段");
}

const char* pipelineName(d::Pipeline pipeline) {
    switch (pipeline) {
        case d::Pipeline::Armor: return "armor";
        case d::Pipeline::PowerRune: return "power_rune";
    }
    throw std::invalid_argument("Log: 无效流水线");
}

const char* jointName(d::Joint joint) {
    switch (joint) {
        case d::Joint::Yaw: return "yaw";
        case d::Joint::Pitch: return "pitch";
    }
    throw std::invalid_argument("Log: 无效关节");
}

const char* referenceName(d::ReferenceSource source) {
    switch (source) {
        case d::ReferenceSource::Plan: return "plan";
        case d::ReferenceSource::Hold: return "hold";
        case d::ReferenceSource::Scan: return "scan";
        case d::ReferenceSource::External: return "external";
    }
    throw std::invalid_argument("Log: 无效参考来源");
}

std::int64_t timeNs(d::TimePoint time) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
}

template <typename T>
Json valueJson(const T& value) { return Json(value); }
Json valueJson(d::TimePoint value) { return timeNs(value); }

template <typename T>
void putOptional(Json& object, const char* key, const std::optional<T>& value) {
    if (value) object[key] = valueJson(*value);
}

Json stampJson(const d::DiagnosticStamp& stamp) {
    Json object = Json::object();
    putOptional(object, "frame_id", stamp.frame_id);
    putOptional(object, "plan_id", stamp.plan_id);
    putOptional(object, "control_tick_id", stamp.control_tick_id);
    putOptional(object, "reference_index", stamp.reference_index);
    return object;
}

Json contextJson(const d::RecordContext& context) {
    Json object{{"stamp", stampJson(context.stamp)}, {"recorded_at_ns", timeNs(context.recorded_at)}};
    if (context.pipeline) object["pipeline"] = pipelineName(*context.pipeline);
    return object;
}

void finite(double value) {
    if (!std::isfinite(value)) throw std::invalid_argument("Log: 诊断数据包含 NaN 或 Inf");
}

template <typename T>
void finite(const std::optional<T>& value) {
    if (!value) return;
    if constexpr (std::is_arithmetic_v<T>) finite(static_cast<double>(*value));
    else for (double component : *value) finite(component);
}

void nonnegative(const std::optional<double>& value) {
    finite(value);
    if (value && *value < 0) throw std::invalid_argument("Log: 时间间隔不能为负数");
}

void validateContext(const d::RecordContext& context, bool needs_frame, bool needs_pipeline) {
    if (needs_frame && !context.stamp.frame_id) throw std::invalid_argument("Log: 缺少 frame_id");
    if (needs_pipeline && !context.pipeline) throw std::invalid_argument("Log: 缺少 pipeline");
    if (context.pipeline) pipelineName(*context.pipeline);
    if (context.stamp.reference_index && !context.stamp.plan_id)
        throw std::invalid_argument("Log: reference_index 必须关联 plan_id");
}

void validateInterval(d::TimePoint start, d::TimePoint finish) {
    if (finish < start) throw std::invalid_argument("Log: 阶段结束时间早于开始时间");
}

void validateInterval(const std::optional<d::TimePoint>& start,
                      const std::optional<d::TimePoint>& finish) {
    if (start.has_value() != finish.has_value())
        throw std::invalid_argument("Log: 阶段开始和结束时间必须同时提供");
    if (start) validateInterval(*start, *finish);
}

void validate(const d::InputData& data) {
    validateContext(data.context, true, false);
    finite(data.yaw_rad);
    finite(data.pitch_rad);
}

void validate(const d::PerceptionData& data) {
    validateContext(data.context, true, true);
    validateInterval(data.started_at, data.finished_at);
    for (const auto& object : data.objects) {
        finite(object.confidence);
        finite(object.position_world_m);
        nonnegative(object.reprojection_rms_px);
        for (const auto& point : object.image_points)
            for (double value : point) finite(value);
    }
}

void validate(const d::TrackingData& data) {
    validateContext(data.context, true, true);
    finite(data.observed_position_world_m);
    finite(data.filtered_position_world_m);
    finite(data.filtered_velocity_world_m_s);
    finite(data.target_angular_velocity_rad_s);
}

void validate(const d::PredictData& data) {
    validateContext(data.context, true, true);
    if (!data.context.stamp.plan_id) throw std::invalid_argument("Log: 缺少 plan_id");
    validateInterval(data.started_at, data.finished_at);
    nonnegative(data.extra_predict_time_s);
    for (std::size_t i = 0; i < data.points.size(); ++i) {
        const auto& point = data.points[i];
        if (point.stamp.frame_id != data.context.stamp.frame_id ||
            point.stamp.plan_id != data.context.stamp.plan_id ||
            !point.stamp.reference_index || *point.stamp.reference_index != i)
            throw std::invalid_argument("Log: 计划点的关联编号或顺序不符");
        if (i && point.execution_timestamp < data.points[i - 1].execution_timestamp)
            throw std::invalid_argument("Log: 计划执行时间顺序不符");
        finite(point.aim_position_world_m);
        nonnegative(point.total_predict_time_s);
        nonnegative(point.flight_time_s);
        finite(point.gimbal_yaw_rad);
        finite(point.gimbal_pitch_rad);
        finite(point.yaw_reference_world_rad);
        finite(point.pitch_reference_rad);
    }
}

void validate(const d::ControlData& data) {
    validateContext(data.context, false, false);
    if (!data.context.stamp.control_tick_id) throw std::invalid_argument("Log: 缺少 control_tick_id");
    jointName(data.joint);
    if (data.reference_source) referenceName(*data.reference_source);
    if (data.reference_source == d::ReferenceSource::Plan &&
        (!data.context.stamp.plan_id || !data.context.stamp.reference_index))
        throw std::invalid_argument("Log: 计划控制周期缺少计划编号或参考下标");
    nonnegative(data.actual_period_s);
    nonnegative(data.solve_duration_s);
    for (const auto& value : {data.measured_position_rad, data.measured_velocity_rad_s,
             data.solver_position_rad, data.solver_velocity_rad_s, data.reference_position_rad,
             data.previous_predicted_position_rad, data.predicted_next_position_rad,
             data.predicted_next_velocity_rad_s, data.mpc_torque_nm, data.integral_torque_nm,
             data.integral_increment_nm, data.command_torque_nm, data.measured_torque_nm,
             data.command_position_rad, data.command_velocity_rad_s}) finite(value);
}

std::size_t payloadBytes(const d::InputData&) { return sizeof(d::InputData); }
std::size_t payloadBytes(const d::TrackingData&) { return sizeof(d::TrackingData); }
std::size_t payloadBytes(const d::ControlData&) { return sizeof(d::ControlData); }

std::size_t payloadBytes(const d::PerceptionData& data) {
    std::size_t size = sizeof(data);
    for (const auto& object : data.objects) {
        if (object.image_points.size() > kQueueBytes / sizeof(object.image_points[0]))
            throw std::length_error("Log: 关键点记录超过队列容量");
        const auto bytes = sizeof(object) + object.image_points.size() * sizeof(object.image_points[0]);
        if (bytes > kQueueBytes || size > kQueueBytes - bytes)
            throw std::length_error("Log: 感知记录超过队列容量");
        size += bytes;
    }
    return size;
}

std::size_t payloadBytes(const d::PredictData& data) {
    if (data.selection_reason.size() > kQueueBytes ||
        data.points.size() > kQueueBytes / sizeof(d::PredictPoint))
        throw std::length_error("Log: 预测记录超过队列容量");
    return sizeof(data) + data.selection_reason.size() + data.points.size() * sizeof(d::PredictPoint);
}

Json serialize(const d::InputData& data) {
    auto object = contextJson(data.context);
    object["frame_timestamp_ns"] = timeNs(data.frame_timestamp);
    object["yaw_rad"] = data.yaw_rad;
    object["pitch_rad"] = data.pitch_rad;
    return object;
}

Json serialize(const d::PerceptionData& data) {
    auto object = contextJson(data.context);
    putOptional(object, "started_at_ns", data.started_at);
    putOptional(object, "finished_at_ns", data.finished_at);
    object["objects"] = Json::array();
    for (const auto& detection : data.objects) {
        Json item{{"detection_index", detection.detection_index}, {"class_id", detection.class_id},
                  {"confidence", detection.confidence}, {"image_points", detection.image_points},
                  {"pose_valid", detection.pose_valid}};
        putOptional(item, "position_world_m", detection.position_world_m);
        putOptional(item, "reprojection_rms_px", detection.reprojection_rms_px);
        object["objects"].push_back(std::move(item));
    }
    return object;
}

Json serialize(const d::TrackingData& data) {
    auto object = contextJson(data.context);
    putOptional(object, "track_id", data.track_id);
    object["class_id"] = data.class_id;
    object["state_timestamp_ns"] = timeNs(data.state_timestamp);
    object["state_valid"] = data.state_valid;
    putOptional(object, "detection_index", data.detection_index);
    putOptional(object, "observed_position_world_m", data.observed_position_world_m);
    putOptional(object, "filtered_position_world_m", data.filtered_position_world_m);
    putOptional(object, "filtered_velocity_world_m_s", data.filtered_velocity_world_m_s);
    putOptional(object, "target_angular_velocity_rad_s", data.target_angular_velocity_rad_s);
    return object;
}

Json serialize(const d::PredictData& data) {
    auto object = contextJson(data.context);
    object["predictor_timestamp_ns"] = timeNs(data.predictor_timestamp);
    putOptional(object, "started_at_ns", data.started_at);
    putOptional(object, "finished_at_ns", data.finished_at);
    object["extra_predict_time_s"] = data.extra_predict_time_s;
    object["valid"] = data.valid;
    object["selection_reason"] = data.selection_reason;
    object["points"] = Json::array();
    for (const auto& point : data.points) {
        Json item{{"stamp", stampJson(point.stamp)},
                  {"execution_timestamp_ns", timeNs(point.execution_timestamp)}, {"valid", point.valid}};
        putOptional(item, "track_id", point.track_id);
        putOptional(item, "target_index", point.target_index);
        putOptional(item, "aim_position_world_m", point.aim_position_world_m);
        putOptional(item, "total_predict_time_s", point.total_predict_time_s);
        putOptional(item, "flight_time_s", point.flight_time_s);
        putOptional(item, "gimbal_yaw_rad", point.gimbal_yaw_rad);
        putOptional(item, "gimbal_pitch_rad", point.gimbal_pitch_rad);
        putOptional(item, "yaw_reference_world_rad", point.yaw_reference_world_rad);
        putOptional(item, "pitch_reference_rad", point.pitch_reference_rad);
        putOptional(item, "fire_allowed", point.fire_allowed);
        object["points"].push_back(std::move(item));
    }
    return object;
}

Json serialize(const d::ControlData& data) {
    auto object = contextJson(data.context);
    object["joint"] = jointName(data.joint);
    if (data.reference_source) object["reference_source"] = referenceName(*data.reference_source);
    object["started_at_ns"] = timeNs(data.started_at);
    object["integral_enabled"] = data.integral_enabled;
    putOptional(object, "solution_valid", data.solution_valid);
    putOptional(object, "actual_period_s", data.actual_period_s);
    putOptional(object, "feedback_received_at_ns", data.feedback_received_at);
    putOptional(object, "feedback_sample_timestamp_ns", data.feedback_sample_timestamp);
    putOptional(object, "measured_position_rad", data.measured_position_rad);
    putOptional(object, "measured_velocity_rad_s", data.measured_velocity_rad_s);
    putOptional(object, "solver_position_rad", data.solver_position_rad);
    putOptional(object, "solver_velocity_rad_s", data.solver_velocity_rad_s);
    putOptional(object, "reference_position_rad", data.reference_position_rad);
    putOptional(object, "previous_predicted_position_rad", data.previous_predicted_position_rad);
    putOptional(object, "predicted_next_position_rad", data.predicted_next_position_rad);
    putOptional(object, "predicted_next_velocity_rad_s", data.predicted_next_velocity_rad_s);
    putOptional(object, "mpc_torque_nm", data.mpc_torque_nm);
    putOptional(object, "integral_torque_nm", data.integral_torque_nm);
    putOptional(object, "integral_increment_nm", data.integral_increment_nm);
    putOptional(object, "command_torque_nm", data.command_torque_nm);
    putOptional(object, "command_position_rad", data.command_position_rad);
    putOptional(object, "command_velocity_rad_s", data.command_velocity_rad_s);
    putOptional(object, "measured_torque_nm", data.measured_torque_nm);
    putOptional(object, "solver_iterations", data.solver_iterations);
    putOptional(object, "solve_duration_s", data.solve_duration_s);
    putOptional(object, "mpc_torque_limited", data.mpc_torque_limited);
    putOptional(object, "mpc_torque_rate_limited", data.mpc_torque_rate_limited);
    putOptional(object, "command_torque_limited", data.command_torque_limited);
    putOptional(object, "command_submitted_at_ns", data.command_submitted_at);
    putOptional(object, "command_send_succeeded", data.command_send_succeeded);
    return object;
}

d::SessionInfo applicationSessionInfo() {
    const auto selector = YAML::LoadFile(PathResolver::resolvePath("config/selector.yaml"));
    const auto name = selector["active_config"].as<std::string>();
    if (name.empty() || name.find('/') != std::string::npos || name.find('\\') != std::string::npos ||
        name.find("..") != std::string::npos)
        throw std::invalid_argument("Log: 机器配置选择器包含无效名称");
    std::ifstream config;
    config.exceptions(std::ios::failbit | std::ios::badbit);
    config.open(PathResolver::resolvePath("config/robots/" + name + ".yaml"));
    std::ostringstream snapshot;
    snapshot << config.rdbuf();
    d::SessionInfo info;
    info.output_directory = PathResolver::resolvePath("logs");
    info.effective_config_yaml = snapshot.str();
    info.build_id = std::string("UnifiedAutoAimPipeline ") + __DATE__ + " " + __TIME__;
#ifdef IS_INFANTRY_2
    info.infantry2_imu_compensation_enabled = static_cast<bool>(IS_INFANTRY_2);
#endif
    return info;
}

bool singleYawFrame(const PipelineResult& result) {
    return result.valid && (result.extra_info.single.valid() || !result.extra_info.big_small.valid());
}

d::RecordContext frameContext(const PipelineResult& result) {
    d::RecordContext context{};
    context.stamp.frame_id = result.extra_info.diagnostic_frame_id;
    if (result.armor.valid == result.power_rune.valid)
        throw std::invalid_argument("Log: 流水线结果必须具有唯一来源");
    context.pipeline = result.armor.valid ? d::Pipeline::Armor : d::Pipeline::PowerRune;
    validateContext(context, true, true);
    return context;
}

template <typename T>
d::WorldPosition worldPosition(const cv::Vec<T, 3>& value) {
    return {static_cast<double>(value[0]), static_cast<double>(value[1]), static_cast<double>(value[2])};
}
} // namespace

struct Log::DiagnosticState {
    struct Record {
        std::size_t stage;
        std::size_t bytes;
        Payload payload;
    };
    std::mutex lifecycle_mutex;
    std::mutex input_mutex;
    std::mutex application_mutex;
    std::mutex control_mutex;
    const void* control_source = nullptr;
    std::optional<std::uint64_t> previous_control_tick;
    std::optional<d::TimePoint> previous_control_start;
    std::optional<double> previous_integral;
    std::atomic<std::uint64_t> control_dropped_records{0};
    std::mutex queue_mutex;
    std::condition_variable ready;
    std::deque<Record> queue;
    std::thread worker;
    std::atomic<bool> active{false};
    bool stopping = false;
    std::size_t queued_bytes = 0;
    std::uint64_t next_frame_id = 0;
    std::uint64_t next_plan_id = 0;
    std::array<std::uint64_t, 5> accepted{};
    std::array<std::uint64_t, 5> written{};
    std::array<std::ofstream, 5> files;
    std::filesystem::path directory;
    Json metadata;

    void writeMetadata(const char* status) {
        metadata["status"] = status;
        metadata["control_dropped_records"] = control_dropped_records.load();
        for (std::size_t i = 0; i < kStages.size(); ++i)
            metadata["records"][kStageNames[i]] = {{"accepted", accepted[i]}, {"written", written[i]}};
        std::ofstream output;
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output.open(directory / "session.json", std::ios::trunc);
        output << metadata.dump(2) << '\n';
        output.close();
    }

    void run() {
        for (;;) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            ready.wait(lock, [this] { return stopping || !queue.empty(); });
            if (queue.empty() && stopping) break;
            Record record = std::move(queue.front());
            queue.pop_front();
            lock.unlock();
            Json object = std::visit([](const auto& data) { return serialize(data); }, record.payload);
            object["written_at_ns"] = timeNs(std::chrono::steady_clock::now());
            files[record.stage] << object.dump() << '\n';
            lock.lock();
            queued_bytes -= record.bytes;
            ++written[record.stage];
        }
    }
};

Log& Log::instance() {
    static Log instance;
    return instance;
}

Log::Log() : diagnostic_(std::make_unique<DiagnosticState>()) {}
Log::~Log() { finishSession(); }

bool Log::enabled(Stage stage) const noexcept { return kStages[stageIndex(stage)]; }
bool Log::sessionActive() const noexcept { return diagnostic_->active.load(std::memory_order_acquire); }

void Log::startSession(const SessionInfo& info) {
    auto& state = *diagnostic_;
    std::lock_guard<std::mutex> lifecycle_lock(state.lifecycle_mutex);
    if (state.active.load() || state.worker.joinable()) throw std::logic_error("Log: 诊断会话已经开启");
    if (info.output_directory.empty() || info.effective_config_yaml.empty() || info.build_id.empty())
        throw std::invalid_argument("Log: 会话输出目录、有效配置和构建标识必须填写");
    std::filesystem::create_directories(info.output_directory);
    static std::atomic<std::uint64_t> session_number{0};
    const auto wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    state.directory = info.output_directory /
        ("diagnostic_" + std::to_string(wall_ns) + "_" + std::to_string(::getpid()) +
         "_" + std::to_string(session_number.fetch_add(1)));
    if (!std::filesystem::create_directory(state.directory)) throw std::runtime_error("Log: 会话目录已经存在");
    state.accepted.fill(0);
    state.control_dropped_records.store(0);
    state.written.fill(0);
    state.queued_bytes = 0;
    state.stopping = false;
    state.metadata = Json{{"schema_version", 1}, {"build_id", info.build_id},
                          {"created_at_unix_ns", wall_ns}, {"clock", "steady_clock"},
                          {"timestamp_unit", "ns"}, {"queue_capacity", kQueueCapacity},
                          {"queue_bytes", kQueueBytes}};
    putOptional(state.metadata, "infantry2_imu_compensation_enabled", info.infantry2_imu_compensation_enabled);
    for (std::size_t i = 0; i < kStages.size(); ++i) {
        state.metadata["stages"][kStageNames[i]] = kStages[i];
        if (!kStages[i]) continue;
        state.files[i].exceptions(std::ios::failbit | std::ios::badbit);
        state.files[i].open(state.directory / (std::string(kStageNames[i]) + ".jsonl"),
                            std::ios::out | std::ios::trunc);
    }
    {
        std::ofstream config;
        config.exceptions(std::ios::failbit | std::ios::badbit);
        config.open(state.directory / "effective_config.yaml");
        config << info.effective_config_yaml;
        config.close();
    }
    state.writeMetadata("recording");
    state.worker = std::thread([&state] { state.run(); });
    state.active.store(true, std::memory_order_release);
}

void Log::finishSession() {
    auto& state = *diagnostic_;
    std::lock_guard<std::mutex> lifecycle_lock(state.lifecycle_mutex);
    if (!state.worker.joinable()) return;
    {
        std::lock_guard<std::mutex> lock(state.queue_mutex);
        state.active.store(false, std::memory_order_release);
        state.stopping = true;
    }
    state.ready.notify_one();
    state.worker.join();
    for (auto& file : state.files) {
        if (!file.is_open()) continue;
        file.flush();
        file.close();
    }
    state.writeMetadata("complete");
}

template <typename Data>
void Log::recordData(Stage stage, const Data& data) {
    if (!enabled(stage)) return;
    const auto recorded_at = std::chrono::steady_clock::now();
    validate(data);
    const std::size_t bytes = payloadBytes(data) + sizeof(DiagnosticState::Record);
    if (bytes > kQueueBytes) throw std::length_error("Log: 单条诊断记录超过队列容量");
    auto& state = *diagnostic_;
    std::lock_guard<std::mutex> lock(state.queue_mutex);
    if (!state.active.load(std::memory_order_acquire)) throw std::logic_error("Log: 尚未开启接收诊断数据的会话");
    if (state.queue.size() >= kQueueCapacity || state.queued_bytes > kQueueBytes - bytes)
        throw std::overflow_error("Log: 诊断队列已满");
    const auto index = stageIndex(stage);
    Data owned = data;
    owned.context.recorded_at = recorded_at;
    state.queue.push_back({index, bytes, std::move(owned)});
    state.queued_bytes += bytes;
    ++state.accepted[index];
    state.ready.notify_one();
}

void Log::recordInputData(const InputData& data) { recordData(Stage::Input, data); }

diagnostics::DiagnosticStamp Log::recordInputData(diagnostics::TimePoint frame_timestamp,
                                                 ExtraInputInfo& extra_info) {
    auto& state = *diagnostic_;
    std::lock_guard<std::mutex> lock(state.input_mutex);
    if (state.next_frame_id == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("Log: 输入图像编号已经耗尽");
    diagnostics::DiagnosticStamp stamp;
    stamp.frame_id = state.next_frame_id++;
    extra_info.diagnostic_frame_id = stamp.frame_id;
    if (!enabled(Stage::Input)) return stamp;
    if (!extra_info.single.valid() && extra_info.big_small.valid()) return stamp;

    InputData data{};
    data.context.stamp = stamp;
    data.frame_timestamp = frame_timestamp;
    data.yaw_rad = extra_info.single.yaw_pos;
    data.pitch_rad = extra_info.single.pitch_angle;
    validate(data);
    ensureApplicationSession();
    recordData(Stage::Input, data);
    return stamp;
}

void Log::recordPerceptionData(const PerceptionData& data) { recordData(Stage::Perception, data); }
void Log::recordTrackingData(const TrackingData& data) { recordData(Stage::Tracking, data); }
void Log::recordPredictData(const PredictData& data) { recordData(Stage::Predict, data); }
void Log::recordControlData(const ControlData& data) { recordData(Stage::Control, data); }

void Log::recordControlData(tcs::RobotController& controller, bool finish) {
    recordControlSource(controller, finish);
}

void Log::recordControlData(tcs::McuMpcController& controller, bool finish) {
    recordControlSource(controller, finish);
}

template <typename Controller>
void Log::recordControlSource(Controller& controller, bool finish) {
    auto& state = *diagnostic_;
    std::lock_guard<std::mutex> lock(state.control_mutex);
    const bool active = enabled(Stage::Control);
    controller.setControlRecordingEnabled(active && !finish);
    if (state.control_source != &controller) {
        state.control_source = &controller;
        state.previous_control_tick.reset();
        state.previous_control_start.reset();
        state.previous_integral.reset();
    }
    auto batch = controller.takeControlRecords();
    if (active) recordControlBatch(batch);
    if (finish || !active) {
        state.previous_control_tick.reset();
        state.previous_control_start.reset();
        state.previous_integral.reset();
    }
}

void Log::recordControlBatch(const tcs::ControlRecordBatch& batch) {
    if (batch.records.empty() && batch.dropped_records == 0) return;
    ensureApplicationSession();
    auto& state = *diagnostic_;
    const auto dropped = state.control_dropped_records.load();
    if (batch.dropped_records > std::numeric_limits<std::uint64_t>::max() - dropped)
        throw std::overflow_error("Log: 控制记录溢出数量超过范围");
    state.control_dropped_records.store(dropped + batch.dropped_records);
    for (const auto& record : batch.records) {
        ControlData yaw{};
        yaw.context.stamp.frame_id = record.stamp.frame_id;
        yaw.context.stamp.plan_id = record.stamp.plan_id;
        yaw.context.stamp.control_tick_id = record.stamp.control_tick_id;
        // 子模组的参考下标属于 yaw 序列。
        if (record.stamp.plan_id) yaw.context.stamp.reference_index = record.stamp.reference_index;
        yaw.joint = diagnostics::Joint::Yaw;
        yaw.started_at = record.started_at;
        yaw.integral_enabled = record.integral_enabled;
        yaw.integral_torque_nm = record.integral;
        yaw.reference_position_rad = record.reference_position;
        yaw.command_position_rad = record.yaw_target_angle;
        yaw.command_velocity_rad_s = record.yaw_target_velocity;
        yaw.command_torque_nm = record.yaw_torque;
        yaw.command_send_succeeded = record.command_send_succeeded;
        if (record.command_send_succeeded.has_value()) yaw.command_submitted_at = record.command_submitted_at;
        const bool continuous = record.stamp.control_tick_id && state.previous_control_tick &&
            *record.stamp.control_tick_id > *state.previous_control_tick &&
            *record.stamp.control_tick_id - *state.previous_control_tick == 1;
        if (continuous && state.previous_control_start)
            yaw.actual_period_s = std::chrono::duration<double>(record.started_at - *state.previous_control_start).count();
        if (continuous && state.previous_integral)
            yaw.integral_increment_nm = record.integral - *state.previous_integral;
        recordData(Stage::Control, yaw);

        ControlData pitch{};
        pitch.context = yaw.context;
        // pitch 序列经过提前截取，现有子模组未返回其原始计划下标。
        pitch.context.stamp.reference_index.reset();
        pitch.joint = diagnostics::Joint::Pitch;
        pitch.started_at = record.started_at;
        pitch.actual_period_s = yaw.actual_period_s;
        pitch.integral_enabled = false;
        pitch.command_position_rad = record.pitch_target_angle;
        pitch.command_submitted_at = yaw.command_submitted_at;
        pitch.command_send_succeeded = record.command_send_succeeded;
        recordData(Stage::Control, pitch);
        state.previous_control_tick = record.stamp.control_tick_id;
        state.previous_control_start = record.started_at;
        state.previous_integral = record.integral;
    }
}

void Log::ensureApplicationSession() {
    std::lock_guard<std::mutex> lock(diagnostic_->application_mutex);
    if (!sessionActive()) startSession(applicationSessionInfo());
}

void Log::recordPerceptionData(const PipelineResult& result) {
    if (!enabled(Stage::Perception) || !singleYawFrame(result)) return;
    PerceptionData data{};
    data.context = frameContext(result);
    if (result.armor.valid) {
        const auto& armor = result.armor;
        if (armor.objects.size() != armor.world_positions.size() ||
            armor.objects.size() != armor.pose_valid.size())
            throw std::invalid_argument("Log: 装甲板检测与位姿结果数量不符");
        std::size_t projection_index = 0;
        for (std::size_t i = 0; i < armor.objects.size(); ++i) {
            const auto& object = armor.objects[i];
            d::PerceptionObject detection{};
            detection.detection_index = static_cast<std::uint32_t>(i);
            detection.class_id = object.label;
            detection.confidence = object.prob;
            for (std::size_t k = 0; k < 4; ++k)
                detection.image_points.push_back({object.landmarks[2 * k], object.landmarks[2 * k + 1]});
            detection.pose_valid = armor.pose_valid[i];
            if (detection.pose_valid) {
                detection.position_world_m = worldPosition(armor.world_positions[i]);
                const auto& projected = armor.reprojected_points.at(projection_index++);
                if (projected.size() != detection.image_points.size())
                    throw std::invalid_argument("Log: 装甲板重投影角点数量不符");
                double squared_error = 0;
                for (std::size_t k = 0; k < projected.size(); ++k) {
                    const double dx = projected[k].x - detection.image_points[k][0];
                    const double dy = projected[k].y - detection.image_points[k][1];
                    squared_error += dx * dx + dy * dy;
                }
                detection.reprojection_rms_px = std::sqrt(squared_error / projected.size());
            }
            data.objects.push_back(std::move(detection));
        }
        if (projection_index != armor.reprojected_points.size())
            throw std::invalid_argument("Log: 装甲板重投影结果数量不符");
    } else {
        const auto& rune = result.power_rune;
        for (std::size_t i = 0; i < rune.detections.size(); ++i) {
            const auto& object = rune.detections[i];
            d::PerceptionObject detection{};
            detection.detection_index = static_cast<std::uint32_t>(i);
            detection.class_id = object.class_id;
            detection.confidence = object.confidence;
            for (const auto& point : object.keypoints) detection.image_points.push_back({point.x, point.y});
            // 能量机关使用联合 PnP，共享的解算结果表示机关旋转中心。
            detection.pose_valid = rune.pose_valid;
            if (rune.pose_valid) detection.position_world_m = worldPosition(rune.pr_world_posi);
            data.objects.push_back(std::move(detection));
        }
    }
    ensureApplicationSession();
    recordData(Stage::Perception, data);
}

void Log::recordTrackingData(const PipelineResult& result) {
    if (!enabled(Stage::Tracking) || !singleYawFrame(result)) return;
    TrackingData data{};
    data.context = frameContext(result);
    data.state_timestamp = result.frame_timestamp;
    if (result.armor.valid) {
        const auto& armor = result.armor;
        data.class_id = armor.target_label;
        data.state_valid = armor.target_valid;
        if (armor.target_valid) data.filtered_position_world_m = worldPosition(armor.target_pos);
        if (armor.target_valid && result.predictor.omega_valid)
            data.target_angular_velocity_rad_s = result.predictor.target_omega;
    } else {
        const auto& rune = result.power_rune;
        data.class_id = -1;
        data.state_valid = !rune.filtered_R.empty();
        if (rune.pose_valid) data.observed_position_world_m = worldPosition(rune.pr_world_posi);
        if (data.state_valid) data.filtered_position_world_m = worldPosition(rune.filtered_pos);
        // filtered_omega 表示机关绕自身轴的转速，不能填写世界 z 轴角速度。
    }
    ensureApplicationSession();
    recordData(Stage::Tracking, data);
}

void Log::recordPredictData(const PipelineResult& result, OutputContext& context,
                           diagnostics::TimePoint timestamp) {
    if (!singleYawFrame(result)) return;
    {
        std::lock_guard<std::mutex> lock(diagnostic_->input_mutex);
        if (diagnostic_->next_plan_id == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Log: 预测计划编号已经耗尽");
        context.diagnostic_plan_id = diagnostic_->next_plan_id++;
    }
    if (!enabled(Stage::Predict)) return;
    PredictData data{};
    data.context = frameContext(result);
    data.context.stamp.plan_id = context.diagnostic_plan_id;
    data.predictor_timestamp = result.predictor_valid ? result.predictor.timestamp : result.frame_timestamp;
    const auto& config = RobotConfig::instance().common;
    data.extra_predict_time_s = config.predictedBallistic.extraPredictTime;
    if (result.predictor_valid)
        data.extra_predict_time_s += std::chrono::duration<double>(timestamp - result.predictor.timestamp).count();
    data.valid = context.predict_result.valid;
    const auto& items = context.predict_result.items;
    if (items.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("Log: 预测计划点数量超过编号范围");
    for (std::size_t i = 0; i < items.size(); ++i) {
        const auto& item = items[i];
        d::PredictPoint point{};
        point.stamp = data.context.stamp;
        point.stamp.reference_index = static_cast<std::uint32_t>(i);
        point.execution_timestamp = timestamp + std::chrono::duration_cast<d::TimePoint::duration>(
            std::chrono::duration<double>(config.predictedBallistic.extraPredictTime + (i + 1) * config.dtControl()));
        point.valid = item.success;
        if (item.target_index != -1) point.target_index = item.target_index;
        if (item.success) {
            point.aim_position_world_m = worldPosition(item.predicted_point);
            point.total_predict_time_s = item.predict_time;
            point.flight_time_s = item.flight_time;
            point.gimbal_yaw_rad = item.gimbal_yaw;
            point.gimbal_pitch_rad = item.gimbal_pitch;
            point.yaw_reference_world_rad = item.yaw;
            point.pitch_reference_rad = item.pitch;
        }
        data.points.push_back(std::move(point));
    }
    ensureApplicationSession();
    recordData(Stage::Predict, data);
}
