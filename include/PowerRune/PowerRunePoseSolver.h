#ifndef POWER_RUNE_POSE_SOLVER_H
#define POWER_RUNE_POSE_SOLVER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "YoloPoseInfer.h"
#include "CameraProjection.h"

// 联合位姿解算结果
struct CombinedPoseResult {
    cv::Mat rvec;   // 联合解算的旋转向量
    cv::Mat tvec;   // 联合解算的平移向量
    int error_code; // 0 表示成功，非零表示失败类型
    std::vector<int> rotation_counts; // class_id为1或5的物体的rotation_count

    CombinedPoseResult() : error_code(1) {}
    bool valid() const { return error_code == 0; }
};

class PowerRunePoseSolver {
public:
    /// 相机内参/畸变来自公共 CameraProjection（配置由 RobotConfig 提供）
    explicit PowerRunePoseSolver(std::shared_ptr<CameraProjection> camera_proj);

    // 对所有检测结果执行联合 PnP 位姿解算
    CombinedPoseResult estimateCombinedPose(
        const std::vector<PoseDetection>& detections) const;

    // 投影 3D 点到图像平面（PnP坐标系，零位姿）
    std::vector<cv::Point2f> projectPoints(std::vector<cv::Point3f> pnp_points_3d) const;

private:
    std::shared_ptr<CameraProjection> camera_proj_;

    // 从 rvec 中提取物体局部坐标系的 z 轴单位向量（在相机坐标系下）
    static cv::Vec3f getObjectZAxis(const cv::Mat& rvec);

    // 从 rvec 中提取物体局部坐标系的 -y 轴单位向量（在相机坐标系下）
    static cv::Vec3f getObjectNegYAxis(const cv::Mat& rvec);

    // Rodrigues 旋转公式：将向量 v 绕单位轴 k 旋转 angle 弧度
    static cv::Vec3f rotateVector(const cv::Vec3f& v, const cv::Vec3f& axis, float angle);

    // 将向量 v 投影到以 normal 为法向的平面上
    static cv::Vec3f projectToPlane(const cv::Vec3f& v, const cv::Vec3f& normal);

    // 绕 z 轴旋转 3D 点
    static cv::Point3f rotateAroundZ(const cv::Point3f& pt, float angle);
};

#endif // POWER_RUNE_POSE_SOLVER_H
