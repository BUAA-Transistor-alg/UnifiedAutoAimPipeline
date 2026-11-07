// PowerRune 物体的世界坐标关键点（自动生成）。
// 由 generate_world_keypoints.py 生成，请勿手动修改。

#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace power_rune_keypoints {

// R  (object_type=0, max_keypoints=8)
inline const std::vector<cv::Point3f> kps_R = {
    { 0.000000f, 0.000000f, -0.166400f },
    { -0.033200f, -0.034200f, -0.166400f },
    { -0.036100f, 0.026800f, -0.166400f },
    { 0.033100f, 0.037400f, -0.166400f },
    { 0.035300f, 0.023700f, -0.166400f },
    { 0.034300f, -0.015000f, -0.166400f },
    { 0.015700f, -0.034200f, -0.166400f },
    { 0.028200f, -0.029100f, -0.166400f },
};

// target  (object_type=2, max_keypoints=9)
inline const std::vector<cv::Point3f> kps_target = {
    { 0.000000f, -0.699600f, 0.000000f },
    { 0.150000f, -0.699600f, 0.000000f },
    { 0.000000f, -0.549600f, 0.000000f },
    { -0.150000f, -0.699600f, 0.000000f },
    { 0.000000f, -0.849600f, 0.000000f },
    { 0.095459f, -0.604141f, 0.000000f },
    { -0.095459f, -0.604141f, 0.000000f },
    { -0.095459f, -0.795059f, 0.000000f },
    { 0.095459f, -0.795059f, 0.000000f },
};

// arrow  (object_type=3, max_keypoints=4)
inline const std::vector<cv::Point3f> kps_arrow = {
    { -0.030000f, -0.504300f, 0.000000f },
    { -0.030000f, -0.174300f, 0.000000f },
    { 0.030000f, -0.174300f, 0.000000f },
    { 0.030000f, -0.504300f, 0.000000f },
};

// small_activating  (object_type=4, max_keypoints=11)
inline const std::vector<cv::Point3f> kps_small_activating = {
    { -0.186946f, -0.540163f, 0.000000f },
    { -0.030496f, -0.505596f, 0.000000f },
    { -0.198401f, -0.331255f, 0.000000f },
    { -0.059535f, -0.175000f, 0.000000f },
    { -0.147558f, -0.731186f, 0.000000f },
    { 0.000000f, -0.850763f, 0.000000f },
    { 0.186946f, -0.540163f, 0.000000f },
    { 0.030496f, -0.505596f, 0.000000f },
    { 0.198401f, -0.331255f, 0.000000f },
    { 0.059535f, -0.175000f, 0.000000f },
    { 0.147558f, -0.731186f, 0.000000f },
};

/// extract_keypoints() 返回的结构体
struct ExtractedKeypoints {
    const std::vector<cv::Point3f>* world_keypoints;
    std::vector<cv::Point2f>   image_keypoints;
};

/// 对给定的颜色扩展类型（0 .. 7），从全局 all_keypoints 向量
/// （长度为 32）中取出对应的世界坐标关键点与图像层关键点。
/// 输入非法时抛出 std::invalid_argument。
inline ExtractedKeypoints extract_keypoints(
    int type,
    const std::vector<cv::Point2f>& all_keypoints)
{
    if (type < 0 || type >= 8) {
        throw std::invalid_argument(
            "extract_keypoints: type " + std::to_string(type)
            + " out of range [0, 8)");
    }
    if (all_keypoints.size() != 32) {
        throw std::invalid_argument(
            "extract_keypoints: all_keypoints.size() "
            + std::to_string(all_keypoints.size())
            + " != 32");
    }

    int yolo_cls = type % 4;
    ExtractedKeypoints result;

    switch (yolo_cls) {
        case 0:
            result.world_keypoints = &kps_R;
            result.image_keypoints.assign(
                all_keypoints.begin() + 0,
                all_keypoints.begin() + 0 + 8);
            break;
        case 1:
            result.world_keypoints = &kps_target;
            result.image_keypoints.assign(
                all_keypoints.begin() + 8,
                all_keypoints.begin() + 8 + 9);
            break;
        case 2:
            result.world_keypoints = &kps_arrow;
            result.image_keypoints.assign(
                all_keypoints.begin() + 17,
                all_keypoints.begin() + 17 + 4);
            break;
        case 3:
            result.world_keypoints = &kps_small_activating;
            result.image_keypoints.assign(
                all_keypoints.begin() + 21,
                all_keypoints.begin() + 21 + 11);
            break;
        default:
            throw std::invalid_argument(
                "extract_keypoints: unexpected yolo_cls "
                + std::to_string(yolo_cls));
    }
    return result;
}

}  // namespace power_rune_keypoints