// OutpostVisualizer.h — 前哨站自瞄可视化（单一接口绘制，照搬 Power_Rune_Auto_Aim 的
//                        PowerRuneVisualizer 结构：所有绘制逻辑集中在 render）
#ifndef OUTPOST_VISUALIZER_H
#define OUTPOST_VISUALIZER_H

#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "OpenvinoInfer.h"
#include "RobotController.h"
#include "TransformTree/RobotTfTree.h"
#include "CameraProjection.h"

/**
 * @brief 可视化所需的数据打包结构体（由 OutpostPipeline 各阶段填充）
 *
 * 与 Power_Rune_Auto_Aim 的 VisualizationData 相同：按数据来源分组为嵌套结构体，
 * 全部世界系数据（EKF 关键点 / 预测中心 / 瞄准点）由 Visualizer 内部完成
 * world→cam→image 投影绘制。
 */
struct OutpostVisualizationData {
    // ---- 检测数据（来自推理后处理） ----
    struct DetectionData {
        std::vector<OutpostDetect::Object> objects;   // 检测结果（已缩放至原图坐标）
    } detection;

    // ---- 原始位姿数据（来自 PnP 解算 + 坐标转换） ----
    struct RawPoseData {
        bool valid = false;
        std::vector<cv::Vec3f> world_positions;   // 每个物体的世界坐标
        std::vector<cv::Vec3f> world_eulers;      // 每个物体的世界欧拉角
        std::vector<std::vector<cv::Point2f>> reprojected_points;  // PnP 角点重投影（图像坐标）
    } raw_pose;

    // ---- ESEKF 滤波数据 ----
    struct FilteredPoseData {
        bool valid = false;
        cv::Vec3d pos = cv::Vec3d(0, 0, 0);              // 滤波位置（world）
        cv::Mat   R64;                                   // 滤波旋转矩阵（CV_64F）
        std::vector<cv::Point3f> world_points;           // EKF 世界关键点（12 个）
        std::vector<cv::Point3f> pred_center_points;     // 预测目标中心关键点 t+0（world）
    } filtered_pose;

    // ---- 瞄准/控制序列数据（来自生成控制序列阶段） ----
    struct AimData {
        bool auto_aim_enable = false;
        cv::Vec3f predicted_point = cv::Vec3f(0, 0, 0);  // 使用的预测目标点（world）
        double predict_time = 0.0;                       // 预测时间（秒）
    } aim;

    // ---- 通信数据（取帧时刻 RobotController 状态快照） ----
    RobotController::State robot_state;

    // ---- 帧信息 ----
    double fps = 0.0;
    std::chrono::steady_clock::time_point frame_timestamp;
};

/**
 * @brief 前哨站自瞄可视化绘制类
 *
 * 将所有仅用于可视化的绘制逻辑集中到 render 单一接口中，包括：
 * 检测框/关键点、PnP 重投影、世界坐标位姿文字、ESEKF 滤波位姿与投影、
 * 预测中心点、瞄准点、MCU/IMU/MPC 通信信息、云台温度、FPS/时间戳。
 */
class OutpostVisualizer {
public:
    /**
     * @brief 主绘制入口
     * @param image       待绘制的图像
     * @param data        可视化数据（各阶段填充）
     * @param tf_tree     当前帧的变换树（world→cam 投影；调用方需已同步并上锁）
     * @param camera_proj 相机投影封装（3D→2D 投影）
     */
    void render(cv::Mat& image,
                const OutpostVisualizationData& data,
                const RobotTfTree& tf_tree,
                const CameraProjection& camera_proj) const;

    // ---- 独立静态工具函数（供其他用途复用） ----

    /// 绘制检测框 + 关键点 + 标签（复制自原 RobotDetectionModel 绘制逻辑）
    static void drawDetectionResults(cv::Mat& image,
                                     const std::vector<OutpostDetect::Object>& objects);

private:
    // ── 通信信息（MCU / IMU / FUSED / STRICT / MPC） ──
    static void drawCommInfo(cv::Mat& img, const RobotController::State& st);

    // ── 云台电机温度（按温度区间变色） ──
    static void drawYawTemperature(cv::Mat& img, int temp, bool valid);

    // ── FPS 与时间戳 ──
    static void drawFps(cv::Mat& img, double fps,
                        const std::chrono::steady_clock::time_point& ts);

    // ── 世界坐标位姿文字 ──
    static void drawWorldPose(cv::Mat& img, int objIdx,
                              const cv::Vec3f& world_pos, const cv::Vec3f& world_euler);

    // ── EKF 滤波后的位姿文字 ──
    static void drawEskfPose(cv::Mat& img, const cv::Vec3f& filtered_pos,
                             const cv::Vec3f& filtered_euler);

    // ── PnP 角点重投影圆点 + 序号 ──
    static void drawReprojectedPoints(cv::Mat& img, int objIdx,
                                      const std::vector<cv::Point2f>& reprojected_pts);

    // ── ESEKF 世界关键点投影（红） ──
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
};

#endif // OUTPOST_VISUALIZER_H
