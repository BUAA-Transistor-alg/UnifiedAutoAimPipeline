#pragma once

// SequencePredictor：按来源分流的四阶段序列预测器。
// predict() 适配控制器状态，SequenceStageProcessor 依次执行准备、预测解算、选板、序列生成。
// Armor：全部未屏蔽真实板统一 Newton 解算；固定点与几何补点按时间合并；
// 在目标中心处以目标到枪口方向为基准，按带符号 w 向来板侧偏转扇区轴；
// 按 |w| 收窄扇区并衰减选板粘滞。
// 首个输出时刻必须是有效固定精确点；后续允许同板插值/外推，缺同板参考时复制。
// PowerRune：保留独立解算器和选点状态机，所有固定精确点必须成功。
// Result.valid 表示控制参考可用；Armor 合成点的 success=false 不阻止其下发。
// 火控由输出层计算：Armor 仅 track_ok；已有的 PowerRune 激活时间门控保留。
// TODO(CONFIG): 以下 Options 为待上车实测的初值，定标并接入 RobotConfig 后再固化。

#include <chrono>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <opencv2/opencv.hpp>
#include "tcs/RobotController.h"
#include "common/BigSmallYaw/RobotStateForBigSmallYaw.h"
#include "common/TaskPool.h"
#include "common/Ballistic/GimbalSolver.h"
#include "common/Ballistic/PredictedBallisticSolver.h"
#include "common/Ballistic/NewtonPredictedBallisticSolver.h"
#include "PowerRune/PredictedPointSelector.h"

class SequencePredictor {
public:
    struct PredictorSource {
        enum class Kind { NONE = 0, ARMOR, POWER_RUNE };
        Kind kind = Kind::NONE;
        int armor_label = -1;                  // Armor 类别不同也视为来源切换
        static PredictorSource armor(int label) { return {Kind::ARMOR, label}; }
        static PredictorSource powerRune() { return {Kind::POWER_RUNE, -1}; }
        bool operator==(const PredictorSource& other) const {
            return kind == other.kind && armor_label == other.armor_label;
        }
        bool operator!=(const PredictorSource& other) const { return !(*this == other); }
    };
    using Source = PredictorSource;
    using Candidate = PredictedBallisticSolver::Result;
    using TimePoint = std::chrono::steady_clock::time_point;
    using LaunchGeometry = GimbalSolver::LaunchGeometry;

    struct Predictor {
        // 世界系预测函数：输入相对快照的秒数，返回车体中心及原始索引顺序的目标点。
        PredictedBallisticSolver::Predictor function;
        PredictorSource source;
        TimePoint timestamp;                   // 快照时间；prepareCommon 统一补偿快照年龄
        std::vector<int> masked_indices;       // Armor 解算跳过；PowerRune 可解除状态机候选的屏蔽
        double target_omega = 0.0;             // 目标自转角速度，rad/s，带正负
        bool omega_valid = false;             // 不可用时关闭 Armor 扇区、几何补点与粘滞
        bool isIndexMasked(int index) const {
            for (int masked : masked_indices) if (masked == index) return true;
            return false;
        }
    };

    struct Item {
        // Armor：仅本时刻精确解算并选板成功时为 true；合成参考点为 false。
        // PowerRune：沿用旧版合成点继承有效性的语义。控制下发以 Result.valid 为准。
        bool success = false;
        cv::Vec3f predicted_point{0.0f, 0.0f, 0.0f};
        double predict_time = 0.0;             // 相对预测器快照的命中时刻；复制点保留来源时间
        float yaw = 0.0f;                      // 已含底盘修正与偏置的控制角，rad
        float pitch = 0.0f;
        float gimbal_yaw = 0.0f;               // 未叠加偏置的解算关节角，供可视化使用
        float gimbal_pitch = 0.0f;
        double flight_time = 0.0;
        int target_index = -1;                 // 原始目标点索引；不再使用高速合成索引
        double target_age = 0.0;               // PowerRune：截至输出控制时刻的观测年龄，不含飞行时间
        bool target_age_valid = false;
    };

    struct Options {
        // TODO(CONFIG): 演示值，不是实车推荐值。|w|=w0 时半角为上下限均值。
        double sector_w0 = 2.0;                 // rad/s，必须 > 0
        // TODO(CONFIG): phi(w)=0.5*atan(w/w_phi)，独立于半角收窄速度；
        // |w|=w_phi 时偏转 22.5 度，正 phi 表示从目标到枪口方向顺时针偏转。
        double sector_phi_w0 = 2.0;             // rad/s，必须有限且 > 0，待实测
        double armor_width_override = 0.0;      // m；0 表示使用 ArmorModel 板宽
        // 补点预算只限制新增的 Newton 采样时刻数，固定点不会被裁掉。
        std::size_t max_geometry_points = 8;
        int max_prior_probes_per_plate = 128;   // 几何探测上限，不做弹道积分
        int boundary_refine_iterations = 8;     // 几何窗口边界二分次数
        double sample_merge_epsilon = 1e-6;     // s，时间去重容差
        // 角速度有效时，粘滞余量乘以 1/(1+(|w|/w_stick)^2)；不可用时关闭。
        bool enable_sticky_selection = true;
        // TODO(CONFIG): 独立于 sector_w0 的待调演示值；|w|=w_stick 时余量减半。
        double w_stick = 2.0;                   // rad/s，必须有限且 > 0
    };

    enum class SampleOrigin { FIXED, GEOMETRY_PRIOR };
    struct SamplePoint {
        // 控制网格相对调用时刻的偏移；尚未加快照年龄和额外预测延迟。
        double control_time = 0.0;
        SampleOrigin origin = SampleOrigin::FIXED;
        int fixed_output_index = -1;            // 补点不属于固定输出网格
    };
    enum class SampleStatus {
        NO_BALLISTIC_SOLUTION,                  // 没有未屏蔽且成功的弹道候选
        NO_ELIGIBLE_TARGET,                     // 有成功解，但没有通过选取规则
        SELECTED
    };
    // 首个固定精确点的逐板选取诊断。可视化直接使用实际判定数据，
    // 包括被扇区排除的成功弹道候选；不重新解算、不改变选板结果。
    enum class SectorCandidateStatus { MASKED, SOLVE_FAILED, INVALID_GEOMETRY, OUTSIDE, ELIGIBLE };
    struct SectorCandidateDiagnostic {
        int target_index = -1;
        SectorCandidateStatus status = SectorCandidateStatus::SOLVE_FAILED;
        bool geometry_valid = false;
        bool sector_applied = false;           // omega 不可用或 r≈0 时为 false
        cv::Vec3f center{0.0f, 0.0f, 0.0f};    // 该候选自己的命中时刻下的目标中心
        cv::Vec3f point{0.0f, 0.0f, 0.0f};
        cv::Vec3f muzzle{0.0f, 0.0f, 0.0f};    // 该候选解对应的发射枪口
        double predict_time = 0.0;             // 相对 EKF 快照的命中时刻
        double radius = 0.0;
        double distance = 0.0;
        double axis_offset = 0.0;              // phi，rad，顺时针为正；扇区停用时为 0
        // 板相对偏转后扇区轴的有符号夹角，逆时针为正，已归一化到 [-pi,pi]。
        double center_angle = std::numeric_limits<double>::quiet_NaN();
        double half_angle = std::numeric_limits<double>::quiet_NaN();
    };
    struct SelectedSample {
        SamplePoint point;
        Candidate selected;
        SampleStatus status = SampleStatus::NO_BALLISTIC_SOLUTION;
        cv::Vec3f launch_muzzle{0.0f, 0.0f, 0.0f};
        double center_angle = std::numeric_limits<double>::quiet_NaN();
        double allowed_half_angle = std::numeric_limits<double>::quiet_NaN();
        double axis_offset = 0.0;              // 与诊断一致的顺时针轴偏移 phi
        std::vector<SectorCandidateDiagnostic> sector_candidates; // 仅首个固定点填充，控制诊断开销
    };
    enum class FillKind { EXACT, INTERPOLATED, EXTRAPOLATED, COPIED };
    struct OutputPointInfo {
        FillKind kind = FillKind::COPIED;
        double control_time = 0.0;
        double left_anchor_time = 0.0;
        double right_anchor_time = 0.0;
        // 该输出时刻本身是否有通过选板的精确解；不是开火资格。
        bool exact_target_valid = false;
    };
    struct Result {
        // Armor 首点必须精确有效；后续同板插值/外推及复制参考均可进入控制序列。
        // 若只有首点有效，暂按已接受的复制规则填满序列；不另设两锚点门槛。
        bool valid = false;
        bool integral_enable = false;
        std::vector<Item> items;               // 长度 (M-1)*K+1+n，步长 dt_control
        cv::Vec3f first_point{0.0f, 0.0f, 0.0f};
        double first_predict_time = 0.0;
        cv::Vec3f yaw_world_origin{0.0f, 0.0f, 0.0f};
        // 保留每个采样结果与输出点来源，便于上车区分解算失败、选板失败和合成参考。
        std::vector<SelectedSample> samples;
        std::vector<OutputPointInfo> point_info;
        std::size_t geometry_points_added = 0;
        bool prior_budget_limited = false;
    };

    SequencePredictor();
    explicit SequencePredictor(const Options& options);
    Result predict(const tcs::RobotController::State&, const Predictor&, const TimePoint&);
    Result predict(const bsy::RobotState&, const Predictor&, const TimePoint&);
    void invalidate();

    // 敌方中心处的半角；单位 m、s、rad。非法几何返回 nullopt。
    static std::optional<double> sectorHalfAngle(
        double omega, double muzzle_center_distance, double plate_radius,
        double plate_width, double w0);

    // 顺时针为正的轴偏移 phi：w>0 为逆时针旋转，来板侧位于枪口连线顺时针侧。
    // phi(0)=0，phi(w) 在 w→±∞ 时趋于 ±pi/4；非法输入返回 NaN。
    static double sectorAxisOffset(double omega, double w_phi);

private:
    struct InputSnapshot {
        bool big_small = false;
        ExtraInputInfo info;
        double bullet_velocity = 0.0;
        bool has_bullet_velocity = false;
        bool auto_aim_switch = false;
        double chassis_yaw_correction = 0.0;
        std::vector<double> pred_big_azimuth_seq;
    };
    // 阶段 1 的输出是本帧局部数据，不用大批临时成员隐式连接各阶段。
    struct PreparedFrame {
        InputSnapshot input;
        Predictor predictor;
        TimePoint timestamp;
        double predictor_age = 0.0;
        double base_predict_time = 0.0;         // 快照年龄 + 配置额外延迟
        int output_count = 0;
        bool all_aims_masked = false;
        bool use_sector = false;
        double sector_axis_offset = 0.0;       // 每帧统一计算；补点与最终选板共用
        double plate_width = 0.0;
        bool use_sticky = false;               // 本帧是否启用选板粘滞
        double base_stick_delta = 0.0;          // 原比例参数 × 平均板中心距离，m
        double stick_delta = 0.0;               // 按 |w| 衰减后的切换余量，m
        cv::Vec3f current_muzzle{0.0f, 0.0f, 0.0f};
        cv::Vec3f yaw_world_origin{0.0f, 0.0f, 0.0f};
        std::vector<cv::Point3f> initial_points;
        std::vector<double> plate_radii;
        std::vector<int> solve_masked_indices;
        std::vector<SamplePoint> samples;        // 固定点 + 补点，按时间排序
        std::vector<float> sample_yaw_big;
        std::vector<LaunchGeometry> launch_geometries;
        std::size_t geometry_points_added = 0;
        bool prior_budget_limited = false;
    };
    // 阶段 2：外层是时间，内层是原始板索引，不删除失败/屏蔽占位符。
    using SolvedSamples = std::vector<std::vector<Candidate>>;
    // 阶段 3：每个采样时刻至多一个选中结果，同时保留失败原因。
    using SelectedSamples = std::vector<SelectedSample>;

    Result SequenceStageProcessor(const InputSnapshot&, const Predictor&, TimePoint);

    // Armor 的四个入口。
    std::optional<PreparedFrame> prepareArmor(const InputSnapshot&, const Predictor&, TimePoint);
    SolvedSamples solveArmor(const PreparedFrame&);
    SelectedSamples selectArmorTargets(const PreparedFrame&, const SolvedSamples&);
    Result generateArmorSequence(const PreparedFrame&, const SelectedSamples&);

    // PowerRune 的四个入口。
    std::optional<PreparedFrame> preparePowerRune(const InputSnapshot&, const Predictor&, TimePoint);
    SolvedSamples solvePowerRune(const PreparedFrame&);
    SelectedSamples selectPowerRuneTarget(const PreparedFrame&, const SolvedSamples&);
    Result generatePowerRuneSequence(const PreparedFrame&, const SelectedSamples&);

    std::optional<PreparedFrame> prepareCommon(const InputSnapshot&, const Predictor&, TimePoint);
    void appendGeometrySamples(PreparedFrame&) const;
    bool priorInsideSector(const PreparedFrame&, int plate, double control_time,
                           double estimated_flight_time) const;
    void captureSampleGeometry(PreparedFrame&);
    float yawBigAtTime(const InputSnapshot&, double control_time) const;
    static InputSnapshot snapshotFromSingle(const tcs::RobotController::State&);
    static InputSnapshot snapshotFromBigSmall(const bsy::RobotState&);
    static double sectorRelativeAngle(const cv::Vec3f& center, const cv::Vec3f& plate,
                                      const cv::Vec3f& muzzle, double clockwise_offset);
    static bool usable(const Candidate&);
    Item makeItem(const PreparedFrame&, const Candidate&) const;
    static Item blendItem(const Item&, const Item&, double fraction);
    Result assembleSequence(const PreparedFrame&, const SelectedSamples&,
                            bool require_all_samples) const;

    Options options_;
    TaskPool pool_;
    std::vector<std::shared_ptr<GimbalSolver>> gimbals_;
    std::vector<NewtonPredictedBallisticSolver> armor_solvers_;
    std::vector<PredictedBallisticSolver> power_rune_solvers_;
    PredictedPointSelector power_rune_selector_;
    double extra_predict_time_;
    double dt_control_;
    double pitch_bias_;
    double yaw_bias_;
    int prediction_points_;
    int interpolation_refine_;
    int exact_lead_points_;
    double aim_stick_ratio_;
    // 跨帧状态；不再维护 slow_latch / fast_latch。
    Source active_source_;
    int last_first_target_index_ = -1;
    std::vector<float> last_yaw_big_seq_;
};
