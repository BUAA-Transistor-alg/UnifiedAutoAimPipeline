#include "PowerRune/TargetPositionCalculator.h"
#include <cmath>

std::vector<cv::Vec3f> TargetPositionCalculator::calculate(
    const cv::Vec3f& position,
    const cv::Mat& rotation_matrix,
    const std::vector<int>& rotation_counts)
{
    constexpr float PI = 3.14159265358979323846f;
    constexpr float TWO_PI_OVER_5 = 2.0f * PI / 5.0f;

    // 本地坐标：(0.0, 0.0, 0.7)
    const float local_x = 0.0f;
    const float local_y = 0.0f;
    const float local_z = 0.7f;

    std::vector<cv::Vec3f> results;
    results.reserve(rotation_counts.size());

    for (int count : rotation_counts) {
        float angle = count * TWO_PI_OVER_5;

        // Rodrigues: 绕 Y 轴: (x, y, z) -> (x*c + z*s, y, -x*s + z*c)
        float c = std::cos(angle);
        float s = std::sin(angle);
        float rx = local_x * c + local_z * s;
        float ry = local_y;
        float rz = -local_x * s + local_z * c;

        // 变换到全局坐标: R * rotated_local
        float gx = rotation_matrix.at<float>(0, 0) * rx +
                   rotation_matrix.at<float>(0, 1) * ry +
                   rotation_matrix.at<float>(0, 2) * rz;
        float gy = rotation_matrix.at<float>(1, 0) * rx +
                   rotation_matrix.at<float>(1, 1) * ry +
                   rotation_matrix.at<float>(1, 2) * rz;
        float gz = rotation_matrix.at<float>(2, 0) * rx +
                   rotation_matrix.at<float>(2, 1) * ry +
                   rotation_matrix.at<float>(2, 2) * rz;

        results.emplace_back(
            position[0] + gx,
            position[1] + gy,
            position[2] + gz
        );
    }

    return results;
}

TargetPositionCalculator::TargetPosFuncPtr TargetPositionCalculator::compose(
    PredictorFuncPtr predictor,
    const std::vector<int>& rotation_counts)
{
    // 将 unique_ptr 转移为 shared_ptr，使 lambda 可拷贝（满足 std::function 要求）
    auto predictor_shared = std::shared_ptr<PredictorFunc>(std::move(predictor));
    auto counts = rotation_counts;  // 复制 rotation_counts

    // 返回统一签名 std::vector<cv::Point3f>(double)（SequencePredictor 直接消费，
    // 不再需要外部把 float/Vec3f 包装为 double/Point3f）
    auto func = [predictor_shared, counts](double delta_t) -> std::vector<cv::Point3f> {
        auto [pos, R] = (*predictor_shared)(static_cast<float>(delta_t));
        const std::vector<cv::Vec3f> pts =
            TargetPositionCalculator::calculate(pos, R, counts);
        std::vector<cv::Point3f> out;
        out.reserve(pts.size());
        for (const auto& p : pts) out.emplace_back(p[0], p[1], p[2]);
        return out;
    };

    return std::make_unique<TargetPosFunc>(std::move(func));
}