// ArmorVisualizer.h — 装甲板自瞄可视化（单一接口绘制，照搬 Power_Rune_Auto_Aim 的
//                        PowerRuneVisualizer 结构：所有绘制逻辑集中在 render）
#ifndef ARMOR_VISUALIZER_H
#define ARMOR_VISUALIZER_H

#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "Armor/ArmorInfer.h"
#include "RobotController.h"
#include "common/TransformTree/RobotTfTree.h"
#include "common/CameraProjection.h"

/**
 * @brief 可视化所需的数据打包结构体（由 ArmorPipeline 各阶段填充）
 *
 * 与 Power_Rune_Auto_Aim 的 VisualizationData 相同：按数据来源分组为嵌套结构体，
 * 全部世界系数据（EKF 关键点 / 预测中心 / 瞄准点）由 Visualizer 内部完成
 * world→cam→image 投影绘制。
 */
struct ArmorVisualizationData {
    // ---- 检测数据（来自推理后处理） ----
    struct DetectionData {
        std::vector<ArmorDetect::Object> objects;   // 检测结果（已缩放至原图坐标）
    } detection;

    // ---- 原始位姿数据（来自 PnP 解算 + 坐标转换） ----
    struct RawPoseData {
        bool valid = false;
        std::vector<cv::Vec3f> world_positions;   // 每个物体的世界坐标
        std::vector<cv::Vec3f> world_eulers;      // 每个物体的世界欧拉角
        std::vector<std::vector<cv::Point2f>> reprojected_points;  // PnP 角点重投影（图像坐标）
    } raw_pose;

    // ---- 目标滤波数据（OutpostESEKF 或移植 EKF 选出的最近目标）----
    struct FilteredPoseData {
        bool valid = false;
        cv::Vec3d pos = cv::Vec3d(0, 0, 0);              // 滤波位置（world）
        cv::Mat   R64;                                   // 滤波旋转矩阵（CV_64F）
        std::vector<cv::Point3f> world_points;           // 目标世界关键点（esekf：12 个；ClassEKF：4 块装甲）
        std::vector<cv::Point3f> pred_center_points;     // 预测目标关键点 t+0（world）
    } filtered_pose;

    // ---- 瞄准/控制序列数据（来自生成控制序列阶段） ----
    struct AimData {
        bool auto_aim_enable = false;
        cv::Vec3f predicted_point = cv::Vec3f(0, 0, 0);  // 使用的预测目标点（world）
        double predict_time = 0.0;                       // 预测时间（秒）
    } aim;

    // ---- XY 平面窗口数据（由 VisualizeOutput 填充；仅 XY 平面窗口绘制使用） ----
    struct XYPlaneData {
        cv::Vec3f chassis_position = cv::Vec3f(0, 0, 0);  // 自身底盘位置（world，米）
        bool      aim_valid = false;                       // 瞄准目标位置是否有效
        cv::Vec3f aim_point = cv::Vec3f(0, 0, 0);          // 瞄准目标位置（world，米）
    } xy;

    // ---- 通信数据（取帧时刻 RobotController 状态快照） ----
    RobotController::State robot_state;

    // ---- 帧信息 ----
    double fps = 0.0;
    std::chrono::steady_clock::time_point frame_timestamp;
};

/**
 * @brief 装甲板自瞄可视化绘制类
 *
 * 将所有仅用于可视化的绘制逻辑集中到 render 单一接口中，包括：
 * 检测框/关键点、PnP 重投影、世界坐标位姿文字、OutpostESEKF 滤波位姿与投影、
 * 预测中心点、瞄准点、MCU/IMU/MPC 通信信息、云台温度、FPS/时间戳。
 */
class ArmorVisualizer {
public:
    /**
     * @brief 主绘制入口
     * @param image       待绘制的图像
     * @param data        可视化数据（各阶段填充）
     * @param tf_tree     当前帧的变换树（world→cam 投影；调用方需已同步并上锁）
     * @param camera_proj 相机投影封装（3D→2D 投影）
     */
    void render(cv::Mat& image,
                const ArmorVisualizationData& data,
                const RobotTfTree& tf_tree,
                const CameraProjection& camera_proj) const;

    // ---- XY 平面窗口（顶视图；仿照 transistor_rm2027_algorithm_visual_ws 的
    //      RMM 顶视图窗口）----
    // 进入 Armor 模式且可视化开启时由 main 调用 openXYWindow()，退出 Armor 模式
    // 时调用 closeXYWindow()（两者均仅在可视化开启时执行）；每帧由
    // VisualizeOutput 调用 renderXY() 刷新窗口。
    static constexpr const char* XY_WINDOW_NAME = "Armor XY Plane";

    /// 创建 XY 平面窗口（幂等；窗口相关操作须在可视化线程调用）
    void openXYWindow();
    /// 关闭 XY 平面窗口（幂等；窗口相关操作须在可视化线程调用）
    void closeXYWindow();
    /// XY 平面窗口当前是否开启（用户手动关闭窗口后自动复位为 false）
    bool xyWindowOpen() const;

    /// 每帧绘制 XY 平面（车体中心 / 装甲板位置（预测函数 t=0 快照）/
    /// 瞄准目标位置 / 自身 chassis 位置 + chassis→瞄准目标连线）并刷新窗口；
    /// 窗口未开启或已被用户手动关闭时直接返回。仅可视化线程调用。
    void renderXY(const ArmorVisualizationData& data);

    // ---- 独立静态工具函数（供其他用途复用） ----

    /// 绘制检测框 + 关键点 + 标签（复制自原 RobotDetectionModel 绘制逻辑）
    static void drawDetectionResults(cv::Mat& image,
                                     const std::vector<ArmorDetect::Object>& objects);

private:
    // ── 世界坐标位姿文字 ──
    static void drawWorldPose(cv::Mat& img, int objIdx,
                              const cv::Vec3f& world_pos, const cv::Vec3f& world_euler);

    // ── EKF 滤波后的位姿文字 ──
    static void drawEskfPose(cv::Mat& img, const cv::Vec3f& filtered_pos,
                             const cv::Vec3f& filtered_euler);

    // ── PnP 角点重投影圆点 + 序号 ──
    static void drawReprojectedPoints(cv::Mat& img, int objIdx,
                                      const std::vector<cv::Point2f>& reprojected_pts);

    // ── OutpostESEKF 世界关键点投影（红） ──
    void drawEskfProjectedPoints(cv::Mat& img,
                                 const std::vector<cv::Point3f>& world_points,
                                 const RobotTfTree& tf_tree,
                                 const CameraProjection& camera_proj) const;

    // ── 预测目标中心关键点投影（绿，t+0） ──
    void drawPredictedCenterPoints(cv::Mat& img,
                                   const std::vector<cv::Point3f>& pred_world,
                                   const RobotTfTree& tf_tree,
                                   const CameraProjection& camera_proj) const;

    // ── 瞄准点投影（品红） ──
    void drawAimPoint(cv::Mat& img,
                      const cv::Vec3f& aim_world, double predict_time,
                      const RobotTfTree& tf_tree,
                      const CameraProjection& camera_proj) const;

    // ── XY 平面窗口状态 ──
    bool xy_window_open_ = false;   // 窗口是否开启（用户手动关闭后自动复位）
    cv::Mat xy_buf_;                // XY 平面绘制缓冲（复用，仅可视化线程访问）
};

#endif // ARMOR_VISUALIZER_H
