// PipelineResult.h — 流水线统一输出结果（两个流水线共用）
//
// 流水线只做感知（检测 → 位姿 → 滤波/预测）与 Predictor 组装（弹道解算所需的
// 目标预测器 Predictor 在流水线内部组装完成，随结果输出，见 PipelineResult::
// predictor）；弹道解算、控制序列生成与可视化均为输出模式（common/Output/）。
// 输出模式通过 tryPopFrame() 拿到本结构体，串口/云台状态则由输出模式直接读取
// RobotController，不经过流水线。
#ifndef PIPELINE_RESULT_H
#define PIPELINE_RESULT_H

#include <array>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common/Input/IInputMode.h"
#include "common/Ballistic/SequencePredictor.h"
#include "Armor/ArmorInfer.h"
#include "PowerRune/PowerRuneInfer.h"
#include "PowerRune/RollPredictor.h"

// ============================================================================
// Armor 感知结果（Armor 流水线填充）
// ============================================================================

// 使用的目标对应的滤波器种类（见 ArmorPipelineData::Stage5Data.target_filter_type）
enum class TargetFilterType {
    NONE = 0,        // 无有效目标
    OUTPOST_ESEKF,   // label 6：OutpostESEKF 误差状态卡尔曼滤波
    CLASS_EKF,       // label 0~5：移植 SuperPower EKF（sp_ekf::ClassEKF）
    NEWEST_OBJECT,   // label 7~8：最新物体位姿保持（sp_ekf::NewestObjectTracker）
};

struct ArmorPerception {
    bool valid = false;

    // ── 阶段3：检测 ──
    std::vector<ArmorDetect::Object> objects;

    // ── 阶段4：PnP + 坐标转换 ──
    std::vector<cv::Vec3f> world_positions;                       // 每物体世界坐标
    std::vector<cv::Vec3f> world_eulers;                          // 每物体世界欧拉角
    std::vector<std::vector<cv::Point2f>> reprojected_points;     // PnP 角点重投影（图像坐标）

    // ── 阶段5：目标滤波（OutpostESEKF 或移植 EKF 选出的最近目标）──
    bool target_valid = false;                    // 所选目标滤波是否有效
    int  target_label = -1;                       // 所选目标类别：0~5=对应 ClassEKF，6=OutpostESEKF，7~8=最新物体；-1=无
    TargetFilterType target_filter_type = TargetFilterType::NONE;  // 使用的目标对应的滤波器种类
    cv::Vec3d target_pos = cv::Vec3d(0, 0, 0);    // 所选目标车体中心（world，米）
    cv::Mat   target_R64;                          // 所选目标旋转矩阵（CV_64F）
    std::vector<cv::Point3f> target_world_points;  // 所选目标世界关键点（esekf：12 点；ClassEKF：4 块装甲；最新物体：1 点）
    std::vector<cv::Point3f> target_pred_center_points;  // 预测目标关键点 t+0（world）
    // 目标关键点预测函数快照不再存放于本结构：流水线已在输出结果时把它与来源
    // 标注、快照时间戳一起组装进 PipelineResult::predictor（见下），供弹道线程
    // 直接作为 sequence_predictor.predict 的输入。

    size_t detection_count = 0;
};

// ============================================================================
// PowerRune 感知结果（PowerRune 流水线填充）
// ============================================================================
struct PowerRunePerception {
    bool valid = false;

    // ── 阶段3：检测 ──
    std::vector<PoseDetection> detections;

    // ── 阶段4：联合 PnP + 坐标转换 ──
    bool pose_valid = false;
    cv::Vec3f pr_world_posi = cv::Vec3f(0, 0, 0);
    cv::Vec3f pr_world_euler = cv::Vec3f(0, 0, 0);
    cv::Mat   pr_world_rot_mat;

    // ── 阶段5：YAxisFilter 滤波 ──
    cv::Vec3f filtered_pos = cv::Vec3f(0, 0, 0);
    cv::Mat   filtered_R;                                        // CV_32F
    float filtered_omega = 0.0f;
    int   jump_a = 0;
    bool  flip = false;
    std::vector<int> filtered_rotation_counts;

    // ── 阶段5：RollPredictor 拟合与预测 ──
    bool fit_valid = false;
    RollPredictor::BigParams   big_params;
    RollPredictor::SmallParams small_params;
    int    direction = 1;
    std::string fit_method;                                      // "big" / "small"
    float  correction_bias = 0.0f;
    std::vector<std::pair<float, float>> fitted_curve;
    std::vector<std::pair<float, float>> raw_points;
    // 位姿预测函数快照：std::pair<cv::Vec3f, cv::Mat>(float dt)（world 系位置 + 旋转矩阵）
    std::unique_ptr<std::function<std::pair<cv::Vec3f, cv::Mat>(float)>> predictor_lambda;
    // 靶点预测函数快照不再存放于本结构：流水线已在输出结果时把它与来源标注、
    // 快照时间戳一起组装进 PipelineResult::predictor（见下；predictor_lambda 仍
    // 保留，供可视化绘制位姿预测）。

    size_t detection_count = 0;
};

// ============================================================================
// 流水线统一输出
// ============================================================================
struct PipelineResult {
    // 始终有效（不受 valid 控制）
    struct QueueSizes {
        int input = 0;
        int inter0 = 0, inter1 = 0, inter2 = 0, inter3 = 0;
        int output = 0;
    };
    QueueSizes queue_sizes;

    // 受 valid 控制
    bool valid = false;
    std::chrono::steady_clock::time_point frame_timestamp;
    ExtraInputInfo extra_info;          // 该帧的 tf 状态（输出模式重建树用）
    cv::Mat frame;                      // 原始帧（移动自流水线数据，供可视化输出模式绘制）

    // 预组装的预测器（sequence_predictor.predict 的直接输入）：激活的流水线在
    // tryPopFrame 输出本结果时组装完成——预测函数快照（Predictor::function，
    // 由本帧目标滤波/拟合产生，world 系）+ 来源标注（Predictor::source：
    // Armor 流水线按 target_label 记 armor(label)，PowerRune 流水线记
    // powerRune()）+ 快照时间戳（Predictor::timestamp，dt 零点 = 快照帧的
    // frame_timestamp）+ 目标屏蔽索引列表（Predictor::masked_indices，索引对应
    // 瞄准点不参与目标选择）。
    // predictor_valid == false 表示本帧无可用目标预测器：main 弹道线程此时调用
    // sequence_predictor.invalidate() 而非 predict()（predict_result 保持无效 →
    // 输出模式进入保持模式）。
    bool predictor_valid = false;
    SequencePredictor::Predictor predictor;

    // 按当前流水线模式二选一填充
    ArmorPerception   armor;
    PowerRunePerception power_rune;
};

#endif // PIPELINE_RESULT_H
