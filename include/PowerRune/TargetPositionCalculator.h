#ifndef TARGET_POSITION_CALCULATOR_H
#define TARGET_POSITION_CALCULATOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include <memory>
#include <utility>

/**
 * @brief 根据位置、姿态旋转矩阵和旋转次数，计算目标点在世界坐标系中的位置。
 *
 * 对于每个 rotation_count，将输入姿态绕其本地 Y 轴旋转 2π/5 * count，
 * 取旋转后姿态下本地坐标 (0.0, 0.0, 0.7) 在全局坐标中的位置，加上输入位置。
 *
 * 能量机关共 kBladeCount（5）个靶点，旋转计数 0..4 即靶点编号：调用方通常用
 * allRotationCounts() 预测**全部 5 个靶点**（列表下标 = 旋转计数），再用
 * missingRotationCounts() 算出本帧不存在（已激活）的靶点索引交给下游屏蔽。
 */
class TargetPositionCalculator {
public:
    /// 能量机关靶点总数：旋转计数 0..kBladeCount-1 与本地 Y 轴 2π/kBladeCount 均布一一对应
    static constexpr int kBladeCount = 5;

    // ---- 类型别名 ----
    /// 预测器函数：输入 delta_t (秒)，返回 (世界位置, 旋转矩阵)
    using PredictorFunc    = std::function<std::pair<cv::Vec3f, cv::Mat>(float)>;
    using PredictorFuncPtr = std::unique_ptr<PredictorFunc>;

    /// 目标位置函数：输入 delta_t (秒)，返回 (预测车体中心位置, 目标世界坐标数组)。
    /// 返回类型直接采用 SequencePredictor 所需的统一签名
    /// （double 秒 → std::pair<cv::Point3f, std::vector<cv::Point3f>>，与 Armor
    /// predictor 一致），使流水线输出的 target_predictor 可直接传入
    /// SequencePredictor，无需外部再包装。能量机关无车体，中心取旋转中心
    /// （即定位预测器给出的 position，world 系）。
    using TargetPosFunc    = std::function<std::pair<cv::Point3f, std::vector<cv::Point3f>>(double)>;
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
     * @brief 全部靶点的旋转计数（升序 {0, 1, ..., kBladeCount-1}）。
     *
     * 传给 compose()/calculate() 即"预测全部 5 个靶点"：返回列表下标 = 旋转计数，
     * 本帧不存在的靶点由 missingRotationCounts() 给出索引并在下游屏蔽。
     */
    static std::vector<int> allRotationCounts();

    /**
     * @brief 本帧不存在的靶点索引（= 需要屏蔽的掩码）。
     *
     * @param present_counts 本帧实际存在的靶点旋转计数（如流水线的
     *                       filtered_rotation_counts：已被观测/尚未激活的靶点）
     * @return 0..kBladeCount-1 中未出现在 present_counts 里的计数（升序）。
     *         下标语义与 allRotationCounts() 生成的预测列表一致（下标即旋转计数），
     *         可直接作为 Predictor::masked_indices。present_counts 中越界的值忽略。
     */
    static std::vector<int> missingRotationCounts(const std::vector<int>& present_counts);

    /**
     * @brief 将预测器函数与目标位置计算组合，返回一个新的可调用对象。
     *
     * 新函数签名为 std::pair<cv::Point3f, std::vector<cv::Point3f>>(double delta_t)
     * （与 PredictedBallisticSolver::Predictor / SequencePredictor 所需签名一致）：
     *   1. 调用 predictor((float)delta_t) 得到 (position, rotation_matrix)
     *   2. 调用 calculate(position, rotation_matrix, rotation_counts) 并返回
     *      (车体中心 = position, 结果列表)（cv::Vec3f 逐个转为 cv::Point3f）
     *
     * 返回列表顺序与 rotation_counts 顺序严格一致。传 allRotationCounts() 时列表
     * 长度恒为 kBladeCount、下标 = 旋转计数，配合 missingRotationCounts() 得到的
     * 掩码即可表达"全部靶点都预测、只屏蔽本帧不存在的靶点"。
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