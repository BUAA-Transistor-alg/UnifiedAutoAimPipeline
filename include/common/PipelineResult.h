// PipelineResult.h — 流水线统一输出结果（两个流水线共用）
//
// 流水线只做感知（检测 → 位姿 → 滤波/预测），弹道解算、控制序列生成与可视化
// 均为输出模式（common/Output/）。输出模式通过 tryPopFrame() 拿到本结构体，
// 串口/云台状态则由输出模式直接读取 RobotController，不经过流水线。
#ifndef PIPELINE_RESULT_H
#define PIPELINE_RESULT_H

#include <array>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "IInputMode.h"
#include "OpenvinoInfer.h"
#include "YoloPoseInfer.h"
#include "RollPredictor.h"

// ============================================================================
// Outpost 感知结果（Outpost 流水线填充）
// ============================================================================
struct OutpostPerception {
    bool valid = false;

    // ── 阶段3：检测 ──
    std::vector<OutpostDetect::Object> objects;

    // ── 阶段4：PnP + 坐标转换 ──
    std::vector<cv::Vec3f> world_positions;                       // 每物体世界坐标
    std::vector<cv::Vec3f> world_eulers;                          // 每物体世界欧拉角
    std::vector<std::vector<cv::Point2f>> reprojected_points;     // PnP 角点重投影（图像坐标）

    // ── 阶段5：ESEKF ──
    bool esekf_initialized = false;
    cv::Vec3d ekf_pos = cv::Vec3d(0, 0, 0);
    cv::Mat   ekf_R64;                                           // CV_64F
    std::vector<cv::Point3f> ekf_world_points;                   // 12 个世界关键点
    std::vector<cv::Point3f> pred_center_points;                 // 预测目标中心 t+0（world）
    // 目标中心点预测函数快照：std::vector<cv::Point3f>(double dt)（world 系）
    std::unique_ptr<std::function<std::vector<cv::Point3f>(double)>> predictor;

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
    // 靶点预测函数快照：std::vector<cv::Vec3f>(float dt)（world 系）
    std::unique_ptr<std::function<std::vector<cv::Vec3f>(float)>> target_predictor;

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

    // 按当前流水线模式二选一填充
    OutpostPerception   outpost;
    PowerRunePerception power_rune;
};

#endif // PIPELINE_RESULT_H
