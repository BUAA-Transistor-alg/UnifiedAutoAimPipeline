#ifndef DEBUG_LOG_H
#define DEBUG_LOG_H

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "tcs/communication/McuDiagnostics.h"
#include "tcs/communication/ImuDiagnostics.h"

struct ExtraInputInfo;
struct PipelineResult;
struct OutputContext;
namespace tcs {
class RobotController;
class McuMpcController;
struct ControlRecordBatch;
}

namespace diagnostics {

/// 同一上位机进程内的单调时间；设备时间转换到该时间域后才能参与时间差计算。
using TimePoint = std::chrono::steady_clock::time_point;

/// 诊断阶段；是否记录各阶段由 Log.cpp 中的常量分别控制。
enum class Stage { Input, Perception, Tracking, Predict, Control, Mcu, Imu };

/// 感知与预测的数据来源；共用输入、无目标保持等记录允许不指定来源。
enum class Pipeline { Armor, PowerRune };

/// 控制关节；一条 ControlData 对应一个关节在一个控制周期中的完整数据。
enum class Joint { Yaw, Pitch };

/// 一条记录与图像、预测计划及控制周期之间的关联。
/// 编号在处理过程中产生并随数据传递，日志开关不参与编号分配。
/// 尚未产生或不适用的编号使用 nullopt；reference_index 从 0 开始。
struct DiagnosticStamp {
    /// 原始图像编号，同一幅图像经过各阶段时保持相同。
    std::optional<std::uint64_t> frame_id;
    /// 一次预测处理的计划编号；每次预测处理分配新编号，无有效计划时仍保留编号。
    std::optional<std::uint64_t> plan_id;
    /// 实际控制周期编号；同一控制循环中的多个关节可以共享该编号。
    std::optional<std::uint64_t> control_tick_id;
    /// 本条记录对应的计划点下标；控制阶段填写本周期实际消费的下标。
    std::optional<std::uint32_t> reference_index;
    /// 本条记录关联的原始 MCU 接收样本编号。
    std::optional<std::uint64_t> mcu_sample_id;
    /// 本条记录关联的原始 IMU 接收样本编号。
    std::optional<std::uint64_t> imu_sample_id;
};

/// 各阶段记录共用的关联信息及采集时刻。
struct RecordContext {
    /// 数据在处理流程中的关联编号。
    DiagnosticStamp stamp;
    /// record*Data() 接收本条数据时自动填写的时间；后台写入时间另行记录。
    TimePoint recorded_at;
    /// 对应的感知流水线；输入阶段尚未区分流水线时使用 nullopt。
    std::optional<Pipeline> pipeline;
};

/// 世界坐标系中的位置，单位为米；坐标方向与 RobotTfTree 保持一致。
using WorldPosition = std::array<double, 3>;

/// 世界坐标系中的速度，单位为米每秒。
using WorldVelocity = std::array<double, 3>;

/// 一幅图像的采集时间及程序实际为其配对的云台姿态。
struct InputData {
    /// 必须包含 frame_id。
    RecordContext context;
    /// 程序给该图像记录的时间戳；当前 Camera 实现为像素转换完成后的时间。
    TimePoint frame_timestamp;
    /// 与本幅图像配对并交给后续处理的 yaw 关节角，相对底盘，单位为弧度。
    double yaw_rad;
    /// 与本幅图像配对并交给后续处理的 pitch 关节角，相对父节点，单位为弧度。
    double pitch_rad;
    /// 与本幅图像配对、已完成安装补偿的 IMU 欧拉角，单位为弧度。
    double imu_euler_yaw_rad;
    double imu_euler_pitch_rad;
    double imu_euler_roll_rad;
    /// 从相机延迟队列选中的状态采样时间及同次采样保存的 MCU/IMU 接收数据。
    std::optional<TimePoint> state_sampled_at;
    std::optional<tcs::McuReceiveSample> mcu;
    std::optional<tcs::ImuReceiveSample> imu;
};

/// 独立于图像频率记录的有效 MCU 原始报文。
struct McuData {
    RecordContext context;
    tcs::McuReceiveSample sample;
};

/// 独立于图像频率记录的有效 IMU 接收报文。
struct ImuData {
    RecordContext context;
    tcs::ImuReceiveSample sample;
};

/// 一个检测结果经过位姿解算后的记录。
struct PerceptionObject {
    /// 本幅图像内的检测编号，仅在该幅图像内有效。
    std::uint32_t detection_index;
    /// 模型类别编号。
    int class_id;
    /// 检测置信度。
    double confidence;
    /// 原始完整图像中的关键点，坐标单位为像素，顺序与模型定义一致。
    std::vector<std::array<double, 2>> image_points;
    /// 位姿解算是否成功。
    bool pose_valid;
    /// 解算所得世界位置；位姿无效时使用 nullopt。
    std::optional<WorldPosition> position_world_m;
    /// 重投影误差的均方根，单位为像素；未计算时使用 nullopt。
    std::optional<double> reprojection_rms_px;
};

/// 一幅图像的检测和位姿解算结果，包括没有检测到目标的情况。
struct PerceptionData {
    /// 必须包含 frame_id 和 pipeline。
    RecordContext context;
    /// 该阶段开始处理本幅图像的时间；输入结果未提供时省略。
    std::optional<TimePoint> started_at;
    /// 该阶段完成处理本幅图像的时间；输入结果未提供时省略。
    std::optional<TimePoint> finished_at;
    /// 各个检测结果；没有检测结果时保持为空。
    std::vector<PerceptionObject> objects;
};

/// 一个目标的一次跟踪更新或无观测外推。
/// optional 字段仅在调用方提供数据时记录数值。
struct TrackingData {
    /// 必须包含 frame_id 和 pipeline。
    RecordContext context;
    /// 跟踪实例编号；跟踪器未提供实例编号时省略。
    std::optional<std::uint64_t> track_id;
    /// 目标类别编号；未提供目标类别时使用 -1。
    int class_id;
    /// 对应的检测编号；本次没有对应检测时使用 nullopt。
    std::optional<std::uint32_t> detection_index;
    /// 本次跟踪状态对应的时间，也是向未来预测的时间基准。
    TimePoint state_timestamp;
    /// 当前跟踪状态是否可用于后续预测。
    bool state_valid;
    /// 世界坐标系中的观测位置；没有观测时使用 nullopt。
    std::optional<WorldPosition> observed_position_world_m;
    /// 滤波后的目标位置；状态无效时使用 nullopt。
    std::optional<WorldPosition> filtered_position_world_m;
    /// 滤波后的目标平动速度；不可用时使用 nullopt。
    std::optional<WorldVelocity> filtered_velocity_world_m_s;
    /// 目标绕世界 z 轴的旋转角速度，单位为弧度每秒；不可用时使用 nullopt。
    std::optional<double> target_angular_velocity_rad_s;
};

/// 一条计划中一个未来时刻的预测与角度解算结果。
struct PredictPoint {
    /// 必须包含 frame_id、plan_id 和 reference_index。
    DiagnosticStamp stamp;
    /// 该计划点预期开始执行的时间；不包含弹丸飞行时间。
    TimePoint execution_timestamp;
    /// 对应目标编号；未选中目标时使用 nullopt。
    std::optional<std::uint64_t> track_id;
    /// 对应装甲板或靶点索引；未选中时使用 nullopt。
    std::optional<int> target_index;
    /// 本计划点是否成功完成预测与弹道解算。
    bool valid;
    /// 预测命中位置，采用世界坐标系；无有效解时使用 nullopt。
    std::optional<WorldPosition> aim_position_world_m;
    /// 从目标状态时间基准到预测命中时刻的总时间，单位为秒。
    std::optional<double> total_predict_time_s;
    /// 弹道飞行时间，单位为秒。
    std::optional<double> flight_time_s;
    /// 相对底盘的 yaw 关节角，单位为弧度。
    std::optional<double> gimbal_yaw_rad;
    /// 相对父节点的 pitch 关节角，单位为弧度。
    std::optional<double> gimbal_pitch_rad;
    /// 生成的世界方向 yaw 参考，包含偏置，单位为弧度。
    std::optional<double> yaw_reference_world_rad;
    /// 生成的 pitch 参考，包含偏置，单位为弧度。
    std::optional<double> pitch_reference_rad;
    /// 该计划点的火控判定；尚未判定时使用 nullopt。
    std::optional<bool> fire_allowed;
};

/// 一次预测计算及其生成的完整计划。
struct PredictData {
    /// 必须包含 frame_id、plan_id 和 pipeline。
    RecordContext context;
    /// 预测器状态快照的时间基准。
    TimePoint predictor_timestamp;
    /// 预测计算开始的时间；计算结果未提供时省略。
    std::optional<TimePoint> started_at;
    /// 预测计算完成的时间；计算结果未提供时省略。
    std::optional<TimePoint> finished_at;
    /// 本次实际采用的额外预测时间，单位为秒。
    double extra_predict_time_s;
    /// 本次是否生成可用计划。
    bool valid;
    /// 本次选择目标的原因，与本条计划直接关联。
    std::string selection_reason;
    /// 全部计划点，按 reference_index 排列。
    std::vector<PredictPoint> points;
};

/// 控制器实际消费的参考来源。
enum class ReferenceSource { Plan, Hold, Scan, External };

/// 一个关节在一个控制周期内真实使用的输入、求解结果和发送状态。
/// 本结构不含动态数组和字符串，供控制循环在数据产生处填写。
struct ControlData {
    /// 必须包含 control_tick_id；消费计划时同时填写 plan_id 和 reference_index。
    RecordContext context;
    /// 本记录对应的控制关节。
    Joint joint;
    /// 本周期实际使用的参考来源；控制器未提供时省略。
    std::optional<ReferenceSource> reference_source;
    /// 本控制周期开始的时间。
    TimePoint started_at;
    /// 相邻控制周期开始时间之差，单位为秒；首个周期使用 nullopt。
    std::optional<double> actual_period_s;
    /// 本次使用的反馈接收时间；未记录接收时间时使用 nullopt。
    std::optional<TimePoint> feedback_received_at;
    /// 本次使用的反馈采样时间；无法取得或转换到单调时间域时使用 nullopt。
    std::optional<TimePoint> feedback_sample_timestamp;
    /// 求解前收到的原始融合关节角，单位为弧度；反馈无效时使用 nullopt。
    std::optional<double> measured_position_rad;
    /// 求解前收到的原始融合关节速度，单位为弧度每秒。
    std::optional<double> measured_velocity_rad_s;
    /// 反馈延迟外推后实际送入 MPC 的关节角，单位为弧度。
    std::optional<double> solver_position_rad;
    /// 反馈延迟外推后实际送入 MPC 的关节速度，单位为弧度每秒。
    std::optional<double> solver_velocity_rad_s;
    /// 参考转换后实际送入 MPC 的首个未来关节角，单位为弧度。
    std::optional<double> reference_position_rad;
    /// 上一周期预测的本周期位置，单位为弧度，用于匹配模型预测与实测反馈。
    std::optional<double> previous_predicted_position_rad;
    /// 本周期预测的下一控制步位置，单位为弧度。
    std::optional<double> predicted_next_position_rad;
    /// 本周期预测的下一控制步速度，单位为弧度每秒。
    std::optional<double> predicted_next_velocity_rad_s;
    /// MPC 求解所得原始力矩，单位为 N·m。
    std::optional<double> mpc_torque_nm;
    /// 本周期积分补偿是否启用。
    bool integral_enabled;
    /// 本周期结束时的积分补偿力矩，单位为 N·m。
    std::optional<double> integral_torque_nm;
    /// 本周期积分补偿力矩的变化量，单位为 N·m。
    std::optional<double> integral_increment_nm;
    /// 最终组装进发送报文的力矩，单位为 N·m；尚未组装时使用 nullopt。
    std::optional<double> command_torque_nm;
    /// 实际组装进报文的关节位置，单位为弧度。
    std::optional<double> command_position_rad;
    /// 实际组装进报文的关节速度，单位为弧度每秒；报文未提供时省略。
    std::optional<double> command_velocity_rad_s;
    /// 电机实际反馈的力矩，单位为 N·m；设备未提供反馈时使用 nullopt。
    std::optional<double> measured_torque_nm;
    /// 本周期求解是否获得可用解；控制器未提供时省略。
    std::optional<bool> solution_valid;
    /// 实际完成的求解迭代次数；求解器未提供时使用 nullopt。
    std::optional<std::uint32_t> solver_iterations;
    /// 本周期求解耗时，单位为秒；未执行求解时使用 nullopt。
    std::optional<double> solve_duration_s;
    /// MPC 原始力矩是否达到幅值限制；未进行判定时使用 nullopt。
    std::optional<bool> mpc_torque_limited;
    /// MPC 原始力矩是否达到变化率限制；未进行判定时使用 nullopt。
    std::optional<bool> mpc_torque_rate_limited;
    /// 叠加积分后的力矩是否触发最终幅值限制；未进行判定时使用 nullopt。
    std::optional<bool> command_torque_limited;
    /// 报文提交给通信接口的时间；未提交时使用 nullopt。
    std::optional<TimePoint> command_submitted_at;
    /// 通信接口是否报告发送成功；未发送或接口未返回结果时使用 nullopt。
    std::optional<bool> command_send_succeeded;
};

/// 一次日志会话的可复现信息；阶段开关由实现文件提供。
struct SessionInfo {
    /// 会话输出根目录，具体会话子目录由日志模块创建。
    std::filesystem::path output_directory;
    /// 本次运行实际生效的完整机器配置，采用 YAML 文本保存。
    std::string effective_config_yaml;
    /// 本次运行程序的构建标识。
    std::string build_id;
    /// 本次构建是否开启 IS_INFANTRY_2 安装补偿；未知时使用 nullopt。
    std::optional<bool> infantry2_imu_compensation_enabled;
};

} // namespace diagnostics

/// 分阶段诊断数据采集入口；函数实现、阶段开关和后台写入位于 Log.cpp。
/// 调用方可以先检查 enabled()，避免提取关闭阶段的数据。
/// record*Data() 必须在返回前取得数据所有权，后台线程不持有调用方对象的引用。
/// 开关关闭时不记录该阶段；队列溢出和文件写入失败必须显式报告。
class Log {
public:
    /// 诊断阶段的公共类型。
    using Stage = diagnostics::Stage;
    /// 日志会话信息的公共类型。
    using SessionInfo = diagnostics::SessionInfo;
    /// 输入阶段数据的公共类型。
    using InputData = diagnostics::InputData;
    /// 感知阶段数据的公共类型。
    using PerceptionData = diagnostics::PerceptionData;
    /// 跟踪阶段数据的公共类型。
    using TrackingData = diagnostics::TrackingData;
    /// 预测阶段数据的公共类型。
    using PredictData = diagnostics::PredictData;
    /// 控制阶段数据的公共类型。
    using ControlData = diagnostics::ControlData;

    /// 返回进程内共享的日志实例。
    static Log& instance();

    /// 返回实现文件中指定阶段的开关状态，可在提取诊断数据前调用。
    bool enabled(Stage stage) const noexcept;

    /// 创建会话并启动后台记录，启用阶段分别写入 JSONL，可选字段缺失时省略。
    /// 重复开启、目录创建失败或文件打开失败时报告错误。
    void startSession(const SessionInfo& info);

    /// 停止接收记录，写完已接收的数据并关闭文件；调用前应停止各采集入口。
    void finishSession();

    /// 返回当前是否已经开启并接收分阶段诊断数据。
    bool sessionActive() const noexcept;

    /// 记录一幅图像及其实际配对姿态，采集时刻及输出字段由 Log.cpp 整理。
    void recordInputData(const InputData& data);

    /// 保存一个原始 MCU 样本；context.stamp 必须包含匹配的 mcu_sample_id。
    void recordMcuData(const diagnostics::McuData& data);

    /// 按接收频率批量保存 MCU 原始包，finish 时关闭采集并取完队列。
    void recordMcuData(tcs::RobotController& controller, bool finish = false);

    /// 保存 IMU 接收样本，关联编号必须与样本编号相同。
    void recordImuData(const diagnostics::ImuData& data);

    /// 独立开关控制逐包采集，finish 时关闭采集并取完队列。
    void recordImuData(tcs::RobotController& controller, bool finish = false);

    /// 从输入源实际返回的时间戳和配对姿态提取输入记录，并返回本幅图像的关联戳。
    /// 首次记录时自动建立会话；编号写入 extra_info 随帧传递，阶段关闭时仍递增。
    /// 当前仅记录单 yaw 构型的数据。
    diagnostics::DiagnosticStamp recordInputData(diagnostics::TimePoint frame_timestamp,
                                                 ExtraInputInfo& extra_info);

    /// 记录一幅图像的检测和位姿解算结果，在 perception 阶段完成后调用。
    void recordPerceptionData(const PerceptionData& data);

    /// 从流水线结果提取检测及位姿，关联输入编号；在主程序取得有效结果后调用。
    void recordPerceptionData(const PipelineResult& result);

    /// 记录一个目标的跟踪更新或外推，在跟踪器处理完成后调用。
    void recordTrackingData(const TrackingData& data);

    /// 提取本帧实际输出的目标滤波状态；没有提供的实例编号和速度保持缺省。
    void recordTrackingData(const PipelineResult& result);

    /// 记录一次预测结果和完整计划，在预测计算完成后调用。
    void recordPredictData(const PredictData& data);

    /// 提取弹道线程生成的完整计划，将计划编号写入输出上下文；timestamp 为预测调用时间基准。
    void recordPredictData(const PipelineResult& result, OutputContext& context,
                          diagnostics::TimePoint timestamp);

    /// 记录一个关节的完整控制周期，在控制器内部完成求解和发送处理后调用。
    void recordControlData(const ControlData& data);

    /// 根据 control 阶段开关启停单 yaw 调试记录，批量取出并记录各实际周期。
    /// finish 为 true 时关闭子模组采集并取完剩余记录，供诊断线程退出后调用。
    void recordControlData(tcs::RobotController& controller, bool finish = false);

    /// 从单 yaw 底层控制器批量记录，开关和退出语义与 RobotController 入口相同。
    void recordControlData(tcs::McuMpcController& controller, bool finish = false);

    /// 禁止复制共享日志实例。
    Log(const Log&) = delete;

    /// 禁止通过赋值复制共享日志实例。
    Log& operator=(const Log&) = delete;

private:
    /// 分阶段记录的队列、文件及后台线程状态。
    struct DiagnosticState;

    /// 验证并复制一条启用阶段的数据，放入有界队列；未开启会话时报告错误。
    template <typename Data>
    void recordData(Stage stage, const Data& data);

    /// 各业务对象记录入口共用的会话创建操作，保证并发首次调用仅创建一个会话。
    void ensureApplicationSession();

    /// 统一同步子模组采集开关，并将批量读取的数据交给转换实现。
    template <typename Controller>
    void recordControlSource(Controller& controller, bool finish);

    /// 在 control 采集锁保护下转换周期数据，统计溢出数量并计算连续周期差值。
    void recordControlBatch(const tcs::ControlRecordBatch& batch);

    /// 初始化日志管理对象。
    Log();

    /// 释放日志实现及其持有的资源。
    ~Log();

    /// 分阶段诊断会话状态，由 startSession() 和 finishSession() 管理。
    std::unique_ptr<DiagnosticState> diagnostic_;
};

#endif // DEBUG_LOG_H
