// CoordinateTransform.cpp
#include "common/TransformTree/CoordinateTransform.h"

#include <cmath>

cv::Mat CoordinateTransform::eulerToRotationMatrix(float yaw, float pitch, float roll) {
    float cy = std::cos(yaw);
    float sy = std::sin(yaw);
    float cp = std::cos(pitch);
    float sp = std::sin(pitch);
    float cr = std::cos(roll);
    float sr = std::sin(roll);

    return (cv::Mat_<float>(3, 3) <<
        cy * cr - sy * sp * sr, -sy * cp, cy * sr + sy * sp * cr,
        sy * cr + cy * sp * sr,  cy * cp, sy * sr - cy * sp * cr,
        -cp * sr,                sp,      cp * cr
    );
}

cv::Vec3f CoordinateTransform::rotationMatrixToEuler(cv::Mat R) {
    float M00 = R.at<float>(0, 0);
    float M01 = R.at<float>(0, 1);
    float M10 = R.at<float>(1, 0);
    float M11 = R.at<float>(1, 1);
    float M20 = R.at<float>(2, 0);
    float M21 = R.at<float>(2, 1);
    float M22 = R.at<float>(2, 2);

    float pitch = std::asin(std::max(-1.0f, std::min(1.0f, M21)));
    const float epsilon = 1e-6f;
    float yaw, roll;
    if (std::fabs(std::cos(pitch)) > epsilon) {
        yaw = std::atan2(-M01, M11);
        roll = std::atan2(-M20, M22);
    } else {
        roll = 0.0f;
        yaw = std::atan2(M10, M00);
    }
    return cv::Vec3f(yaw, pitch, roll);
}

std::vector<cv::Vec3f> CoordinateTransform::transformPoints(const cv::Vec3f& position,
                                                            const cv::Vec3f& euler,
                                                            const std::vector<cv::Vec3f>& points) {
    cv::Mat R = eulerToRotationMatrix(euler);
    std::vector<cv::Vec3f> out;
    out.reserve(points.size());
    for (const auto& p : points) {
        cv::Mat v = (cv::Mat_<float>(3, 1) << p[0], p[1], p[2]);
        cv::Mat w = R * v;
        out.emplace_back(
            w.at<float>(0, 0) + position[0],
            w.at<float>(1, 0) + position[1],
            w.at<float>(2, 0) + position[2]
        );
    }
    return out;
}

std::vector<cv::Point3f> CoordinateTransform::transformPoints(const cv::Vec3f& position,
                                                              const cv::Vec3f& euler,
                                                              const std::vector<cv::Point3f>& points) {
    cv::Mat R = eulerToRotationMatrix(euler);
    std::vector<cv::Point3f> out;
    out.reserve(points.size());
    for (const auto& p : points) {
        cv::Mat v = (cv::Mat_<float>(3, 1) << p.x, p.y, p.z);
        cv::Mat w = R * v;
        out.emplace_back(
            w.at<float>(0, 0) + position[0],
            w.at<float>(1, 0) + position[1],
            w.at<float>(2, 0) + position[2]
        );
    }
    return out;
}
