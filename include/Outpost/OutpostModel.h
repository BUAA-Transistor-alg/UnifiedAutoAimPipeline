// OutpostModel.h — 前哨站 3D 模型关键点（从 test/main.cpp 挪出，供 PnP 与 ESEKF 阶段共用）
//
// 前哨站：3 个装甲面绕中心轴旋转，每面由 4 个角点组成（小装甲板尺寸
// w=0.133m 宽, h=0.05m 高），面 1 倾斜 15° 并沿 y 偏移 -0.275m，
// 三面绕 x 轴（世界 z 轴）间隔 120° 布置。
#ifndef OUTPOST_MODEL_H
#define OUTPOST_MODEL_H

#include <vector>
#include <opencv2/opencv.hpp>

#include "TransformTree/CoordinateTransform.h"

namespace OutpostModel {

// 物体 3D 尺寸（w=0.133m 宽, h=0.05m 高）
constexpr float OBJ_W = 0.133f;
constexpr float OBJ_H = 0.050f;

// 物体四个角点（局部坐标系，逆时针顺序）
inline const std::vector<cv::Point3f> OBJECT_POINTS_3D_LOCAL = {
    cv::Point3f(-OBJ_W/2.0f, 0.0f,  OBJ_H/2.0f),  // 左上
    cv::Point3f(-OBJ_W/2.0f, 0.0f, -OBJ_H/2.0f),  // 左下
    cv::Point3f( OBJ_W/2.0f, 0.0f, -OBJ_H/2.0f),  // 右下
    cv::Point3f( OBJ_W/2.0f, 0.0f,  OBJ_H/2.0f)   // 右上
};

inline const std::vector<cv::Point3f> TARGET_CENTER_3D_LOCAL = {
    cv::Point3f(0.0f, 0.0f, 0.0f)
};

// ── 辅助：3x3 矩阵 × 三维列向量 ──
inline cv::Vec3f matVecMul(const cv::Mat& M, const cv::Vec3f& v) {
    CV_Assert(M.rows == 3 && M.cols == 3);
    CV_Assert(M.channels() == 1);
    int depth = M.depth();
    CV_Assert(depth == CV_32F || depth == CV_64F);

    cv::Mat vecMat, resultMat;
    if (depth == CV_32F) {
        vecMat = (cv::Mat_<float>(3, 1) << v[0], v[1], v[2]);
        resultMat = M * vecMat;
        return cv::Vec3f(resultMat.at<float>(0, 0),
                         resultMat.at<float>(1, 0),
                         resultMat.at<float>(2, 0));
    } else {
        vecMat = (cv::Mat_<double>(3, 1) << v[0], v[1], v[2]);
        resultMat = M * vecMat;
        return cv::Vec3f(static_cast<float>(resultMat.at<double>(0, 0)),
                         static_cast<float>(resultMat.at<double>(1, 0)),
                         static_cast<float>(resultMat.at<double>(2, 0)));
    }
}

// ── 辅助：二维点云扁平化 ──
inline std::vector<cv::Point3f> flattenPointCloud(
    const std::vector<std::vector<cv::Point3f>>& vec2d) {
    size_t totalSize = 0;
    for (const auto& inner : vec2d) totalSize += inner.size();
    std::vector<cv::Point3f> result;
    result.reserve(totalSize);
    for (const auto& inner : vec2d)
        result.insert(result.end(), inner.begin(), inner.end());
    return result;
}

// ── 构造前哨站全体坐标：3 个装甲面的 3D 关键点列表 ──
inline std::vector<std::vector<cv::Point3f>> buildOutpostPoints3D(
    const std::vector<cv::Point3f>& small_armor_points_3d_local) {
    std::vector<std::vector<cv::Point3f>> outpost_points_3d_list;

    cv::Mat slant_R = CoordinateTransform::eulerToRotationMatrix(0.0f, 15.0f * M_PI / 180.0f, 0.0f);
    cv::Vec3f move_vec = cv::Vec3f(0.0f, -0.275f, 0.0f);
    std::vector<cv::Vec3f> slant_move_points_vec;
    for (const cv::Point3f& p3 : small_armor_points_3d_local) {
        slant_move_points_vec.push_back(matVecMul(slant_R, cv::Vec3f(p3.x, p3.y, p3.z)) + move_vec);
    }
    for (int i = 0; i < 3; i += 1) {
        cv::Mat rotate_R = CoordinateTransform::eulerToRotationMatrix((float)i * M_PI * 2.0f / 3.0f, 0.0f, 0.0f);
        std::vector<cv::Point3f> one_armor_points;
        for (const cv::Vec3f& p3v : slant_move_points_vec) {
            cv::Vec3f temp_p3v = matVecMul(rotate_R, p3v);
            one_armor_points.push_back({temp_p3v[0], temp_p3v[1], temp_p3v[2]});
        }
        outpost_points_3d_list.push_back(one_armor_points);
    }
    return outpost_points_3d_list;
}

// 3 个装甲面（每面 4 角点）局部 3D 关键点
inline const std::vector<std::vector<cv::Point3f>> OUTPOST_POINTS_3D_LIST =
    buildOutpostPoints3D(OBJECT_POINTS_3D_LOCAL);
// 扁平化（12 个点）
inline const std::vector<cv::Point3f> FLATTEN_OUTPOST_POINTS_3D_LIST =
    flattenPointCloud(OUTPOST_POINTS_3D_LIST);
// 3 个面中心（目标中心关键点）
inline const std::vector<cv::Point3f> OUTPOST_TARGET_CENTER_3D_LIST =
    flattenPointCloud(buildOutpostPoints3D(TARGET_CENTER_3D_LOCAL));

} // namespace OutpostModel

#endif // OUTPOST_MODEL_H
