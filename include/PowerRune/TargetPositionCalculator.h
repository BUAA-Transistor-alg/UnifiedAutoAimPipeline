#ifndef TARGET_POSITION_CALCULATOR_H
#define TARGET_POSITION_CALCULATOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include <memory>

/**
 * @brief 根据位置、姿态旋转矩阵和旋转次数，计算目标点在世界坐标系中的位置。
 *
 * 对于每个 rotation_count，将输入姿态绕其本地 Y 轴旋转 2π/5 * count，
 * 取旋转后姿态下本地坐标 (0.0, 0.0, 0.7) 在全局坐标中的位置，加上输入位置。
 */
class TargetPositionCalculator {
public:
    // ---- 类型别名 ----
    /// 预测器函数：输入 delta_t (秒)，返回 (世界位置, 旋转矩阵)
    using PredictorFunc    = std::function<std::pair<cv::Vec3f, cv::Mat>(float)>;
    using PredictorFuncPtr = std::unique_ptr<PredictorFunc>;

    /// 目标位置函数：输入 delta_t (秒)，返回目标世界坐标数组
    using TargetPosFunc    = std::function<std::vector<cv::Vec3f>(float)>;
    using TargetPosFuncPtr = std::unique_ptr<TargetPosFunc>;

    /**
     * @brief 计算目标世界坐标
     * @param position         世界坐标系下的位置
     * @param rotation_matrix  3x3 浮点旋转矩阵 (CV_32F)
     * @param rotation_counts  旋转次数向量
     * @return 按顺序排列的世界坐标点向量
     */
    static std::vector<cv::Vec3f> calculate(
        const cv::Vec3f& position,
        const cv::Mat& rotation_matrix,
        const std::vector<int>& rotation_counts);

    /**
     * @brief 将预测器函数与目标位置计算组合，返回一个新的可调用对象。
     *
     * 新函数签名为 std::vector<cv::Vec3f>(float delta_t)：
     *   1. 调用 predictor(delta_t) 得到 (position, rotation_matrix)
     *   2. 调用 calculate(position, rotation_matrix, rotation_counts) 并返回结果
     *
     * @param predictor        由 RollPredictor::capturePredictor() 返回的快照函数
     * @param rotation_counts  旋转次数向量（将被复制到新函数内部）
     * @return unique_ptr 管理的组合函数
     */
    static TargetPosFuncPtr compose(
        PredictorFuncPtr predictor,
        const std::vector<int>& rotation_counts);
};

#endif // TARGET_POSITION_CALCULATOR_H