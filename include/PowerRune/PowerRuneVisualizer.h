#ifndef POWER_RUNE_VISUALIZER_H
#define POWER_RUNE_VISUALIZER_H

#include "PowerRune/PowerRunePoseSolver.h"
#include "common/TransformTree/RobotTfTree.h"
#include "common/CameraProjection.h"
#include "PowerRune/PowerRuneInfer.h"
#include "PowerRune/RollPredictor.h"

#include <opencv2/opencv.hpp>
#include <array>
#include <vector>
#include <string>
#include <utility>

/**
 * @brief 可视化所需的数据打包结构体
 *
 * 由数据处理流程(PowerRunePipeline)填充，传递给PowerRuneVisualizer进行绘制。
 * 内部按数据来源分组为嵌套结构体。
 */
struct PowerRuneVisualizationData {
    // ---- 检测数据(来自PowerRuneInfer推理) ----
    struct DetectionData {
        std::vector<PoseDetection> detections;
    } detection;

    // ---- 原始位姿数据(来自PnP解算+坐标转换) ----
    struct RawPoseData {
        cv::Vec3f world_pos   = cv::Vec3f(0, 0, 0);
        cv::Vec3f world_euler = cv::Vec3f(0, 0, 0);
        cv::Vec3f world_tvec;
        cv::Mat   world_rot_mat;
    } raw_pose;

    // ---- 滤波数据(来自YAxisFilter) ----
    struct FilteredPoseData {
        float filtered_omega  = 0.0f;
        int   jump_a          = 0;
        bool  flip            = false;
        cv::Vec3f filtered_pos;
        cv::Mat   filtered_R;
        cv::Mat filter_predicted_R;
        std::vector<int> target_rotation_counts;
    } filtered_pose;

    // ---- Roll 预测器数据(来自RollPredictor) ----
    struct RollPredictorData {
        bool  fit_valid    = false;
        RollPredictor::BigParams   big_params;
        RollPredictor::SmallParams small_params;
        int   direction    = 1;
        std::string fit_method;   // "big" 或 "small"
        float correction_bias = 0.0f;

        // 可视化曲线数据
        std::vector<std::pair<float, float>> fitted_curve;
        std::vector<std::pair<float, float>> raw_points;
        std::pair<cv::Vec3f, cv::Mat> predictor_prediction;
    } roll_predictor;

    // ---- 目标位置可视化点(由TargetPositionCalculator计算) ----
    std::vector<cv::Vec3f> filtered_target_points;   // 滤波位姿下的目标位置世界坐标(黄色)
    std::vector<cv::Vec3f> predictor_target_points;  // 预测位姿下的目标位置世界坐标(天蓝)
};

/**
 * @brief 能量机关可视化绘制类
 *
 * 将所有仅用于可视化的绘制逻辑从PowerRunePipeline中分离到此类的render方法中。
 * 包括：检测框绘制、位姿坐标轴(五边形+三轴)绘制、文字叠加等。
 */
class PowerRuneVisualizer {
public:
    // ==================== 类别名称 ====================
    static constexpr const char* CLASS_NAMES[] = {
        "R_red", "target_red", "arrow_red", "small_activating_red",
        "R_blue", "target_blue", "arrow_blue", "small_activating_blue"
    };

    // ==================== 颜色常量 ====================
    static const cv::Scalar COLOR_WHITE;
    static const cv::Scalar COLOR_BLACK;
    static const cv::Scalar COLOR_GRAY;

    // 6 种高饱和颜色 (BGR)
    static const cv::Scalar OBJECT_PALETTE[];
    static constexpr int OBJECT_PALETTE_SIZE = 6;

    /**
     * @brief 主绘制入口
     *
     * 将 VisualizationData 中的所有可视化信息绘制到 image 上。
     *
     * @param image       待绘制的图像
     * @param data        可视化数据
     * @param tf_tree     变换树（world→cam 投影；调用方需已同步并上锁）
     * @param camera_proj 相机投影封装（3D→2D 投影）
     */
    void render(cv::Mat& image,
                const PowerRuneVisualizationData& data,
                const RobotTfTree& tf_tree,
                const CameraProjection& camera_proj) const;

    /**
     * @brief 绘制检测框和关键点(独立工具函数)
     * @param image       待绘制的图像
     * @param detections  检测结果
     */
    static void drawDetections(cv::Mat& image,
                               const std::vector<PoseDetection>& detections);

    /**
     * @brief 绘制五边形与三轴
     *
     * 所有顶点(五边形/三轴终点/中心点)均通过 world→cam→PnP→image 流程投影。
     *
     * @param pentagon_color     五边形颜色, 若纯黑则不绘制
     * @param pentagon_thickness 五边形线宽
     * @param axis_colors        三轴颜色数组(3个Scalar)
     * @param axis_thickness     三轴线宽
     * @param center_color       中心点颜色
     * @param center_radius      中心点半径
     */
    static void drawPoseAxes(
        cv::Mat& image,
        const cv::Vec3f& world_pos,
        const cv::Mat& world_rot_mat,
        const cv::Scalar& pentagon_color,
        int pentagon_thickness,
        const std::array<cv::Scalar, 3>& axis_colors,
        int axis_thickness,
        const cv::Scalar& center_color,
        int center_radius,
        const RobotTfTree& tf_tree,
        const CameraProjection& camera_proj);

    /**
     * @brief 将世界坐标点投影到图像平面（经 tf 树 world→cam + 相机投影）
     */
    cv::Point2f worldToImage(const cv::Vec3f& p_world,
                             const RobotTfTree& tf_tree,
                             const CameraProjection& camera_proj) const;

private:
    // ==================== 五边形生成 ====================
    static std::vector<cv::Point3f> generatePentagonPoints();

    // 五边形顶点(局部坐标)
    static const std::vector<cv::Point3f> pentagon_local_;
};

#endif // POWER_RUNE_VISUALIZER_H