// CoordinateTransform.h
#ifndef COORDINATE_TRANSFORM_H
#define COORDINATE_TRANSFORM_H

#include <opencv2/opencv.hpp>
#include <vector>

// 纯坐标转换工具类：仅提供欧拉角与旋转矩阵之间的互转。
// 坐标系定义（CamFrame）：
//   x：向右，y：向前，z：向上
//   yaw：绕 z 轴 从上方看逆时针 x 轴转向 y 轴
//   pitch：绕 x 轴 抬头 y 轴转向 z 轴
//   roll：绕 y 轴 从画面看顺时针 z 轴转向 x 轴
class CoordinateTransform {
public:
    // 欧拉角 -> 旋转矩阵
    static cv::Mat eulerToRotationMatrix(float yaw, float pitch, float roll);
    static cv::Mat eulerToRotationMatrix(cv::Vec3f euler) {
        return eulerToRotationMatrix(euler[0], euler[1], euler[2]);
    }

    // 旋转矩阵 -> 欧拉角
    static cv::Vec3f rotationMatrixToEuler(cv::Mat R);

    // 将物体本体坐标系下的点，变换到物体所在坐标系下。
    // position：物体所在坐标系下的位置；euler：物体所在坐标系下的欧拉角(yaw, pitch, roll)。
    // points：物体本体坐标系下的点集。
    // 返回：这些点在物体所在坐标系下的坐标。
    static std::vector<cv::Vec3f> transformPoints(const cv::Vec3f& position,
                                                  const cv::Vec3f& euler,
                                                  const std::vector<cv::Vec3f>& points);

    // 重载：输入输出点为 cv::Point3f（坐标和欧拉角仍为 cv::Vec3f）。
    static std::vector<cv::Point3f> transformPoints(const cv::Vec3f& position,
                                                    const cv::Vec3f& euler,
                                                    const std::vector<cv::Point3f>& points);
};

#endif // COORDINATE_TRANSFORM_H
