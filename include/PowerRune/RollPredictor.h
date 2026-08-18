#ifndef ROLL_PREDICTOR_H
#define ROLL_PREDICTOR_H

#include <deque>
#include <chrono>
#include <vector>
#include <utility>
#include <functional>
#include <memory>
#include <opencv2/opencv.hpp>

#include "TaskPool.h"

/**
 * @brief 基于历史 roll 角数据的预测器，用于预测未来绕物体 Y 轴的旋转角度。
 *
 * 输入：欧拉角中的 roll 分量（范围 [-π, π]），内部自动做相位连续化 (unwrap)。
 * 支持两种拟合模型，自动选择 MSE 更小者：
 *   - "big"   : r(t) = sign * (-a/omega * cos(omega*(t+o_t)) + (2.090 - a)*(t+o_t))
 *   - "small" : r(t) = sign * pi/3 * (t + o_t)
 * 其中 t=0 对应当前时刻，a、omega、o_t 为拟合参数，sign = ±1 由角速度方向决定。
 *
 * 使用方式：
 *   1. 每帧调用 update(roll_raw, timestamp) 输入观测 roll 和时间戳
 *   2. 调用 predict(delta_t) 获取未来 delta_t 秒处的预测 roll
 *   3. 调用 isValid() 检查拟合是否有效
 */
class RollPredictor
{
public:
    /// 拟合方法类型
    enum class FitMethod { BIG, SMALL };

    /**
     * @brief 构造函数
     * @param max_time_window  队列中数据的最长保留时间（秒）
     * @param min_data_points  进行拟合所需的最小数据点数
     * @param correction_window 修正偏置计算所用的最新数据点数（不超过 min_data_points）
     * @param grid_search_interval 两次完整网格搜索之间的最小间隔（秒），≤0 表示每帧都做网格搜索
     */
    RollPredictor(float max_time_window = 5.0f, int min_data_points = 20,
                  float queue_time_threshold = 1.0f, int correction_window = 5,
                  float grid_search_interval = 0.5f);

    /**
     * @brief 输入观测 roll 值和时间戳，内部做 unwrap 连续化、维护队列、执行拟合。
     * @param roll_raw   rotationMatrixToEuler 输出的 roll 分量（范围 [-π, π]）
     * @param timestamp  当前帧的时间戳
     */
    void update(float roll_raw,
                const std::chrono::steady_clock::time_point& timestamp,
                cv::Vec3f posi,
                cv::Mat y_axis_R);

    /**
     * @brief 无观测数据时的特殊更新：仅根据时间推移移动 o_t，不重新拟合。
     *        用于没有 filtered_euler 的帧，保持 t=0 始终对应当前时刻。
     * @param timestamp  当前帧的时间戳
     */
    void updateWithoutData(const std::chrono::steady_clock::time_point& timestamp);

    /**
     * @brief 基于拟合模型，预测未来 delta_t 秒处的 roll 值。
     * @param delta_t  预测时间（秒），> 0 表示向未来预测
     * @return 预测的连续 roll 值，若拟合无效则返回 0.0f
     */
    float predict_roll(float delta_t) const;

    std::pair<cv::Vec3f, cv::Mat> predict_posi_and_R(float delta_t) const;

    /**
     * @brief 计算当前拟合在已有观测数据上的 RMSE
     * @return 均方根误差，若拟合无效或数据不足则返回 +inf
     */
    float computeRMSE() const;

    /**
     * @brief 捕获当前预测器状态的快照，返回一个独立于预测器后续状态变化的可调用对象。
     *
     * 返回的 lambda 内部复制了调用时刻的所有拟合参数、位置和旋转矩阵（cv::Mat 深度克隆），
     * 因此不会随 RollPredictor 后续 update/reset 而改变。
     * lambda 的行为与 predict_posi_and_R(delta_t) 完全一致。
     *
     * @return unique_ptr 管理的 function，签名为 std::pair<cv::Vec3f, cv::Mat>(float delta_t)
     */
    std::unique_ptr<std::function<std::pair<cv::Vec3f, cv::Mat>(float)>> capturePredictor() const;


    /**
     * @brief 重置预测器，清空队列和拟合状态。
     *        若已处于重置状态则自动跳过。
     */
    void reset();

    /**
     * @brief 当前拟合是否有效
     */
    bool isValid() const { return fit_valid_; }

    /**
     * @brief 获取当前使用的拟合方法
     * @return FitMethod::BIG 表示非线性模型，FitMethod::SMALL 表示线性模型
     */
    FitMethod getFitMethod() const { return fit_method_; }

    /**
     * @brief 获取当前拟合方法名称字符串
     * @return "big" 或 "small"
     */
    const char* getFitMethodName() const {
        return fit_method_ == FitMethod::BIG ? "big" : "small";
    }

    // ==================== 拟合参数结构体 ====================

    /// big 模型参数: r(t) = sign * (-a/omega * cos(omega*(t+o_t)) + (2.090 - a)*(t+o_t))
    struct BigParams {
        float a     = 0.0f;
        float omega = 0.0f;
        float o_t   = 0.0f;
    };

    /// small 模型参数: r(t) = sign * pi/3 * (t + o_t)
    struct SmallParams {
        float o_t = 0.0f;
    };

    /**
     * @brief 获取 big 模型拟合参数（值拷贝，线程安全）
     */
    BigParams getBigParams() const { return big_params_; }

    /**
     * @brief 获取 small 模型拟合参数（值拷贝，线程安全）
     */
    SmallParams getSmallParams() const { return small_params_; }

    /**
     * @brief 设置旋转方向（正转 +1，反转 -1）
     */
    void setDirection(int direction) { direction_ = direction > 0 ? 1 : -1; }

    /**
     * @brief 获取当前旋转方向
     */
    int getDirection() const { return direction_; }

    /**
     * @brief 获取当前连续化的 roll 值 r(0)
     */
    float getCurrentRoll() const { return continuous_roll_; }

    /**
     * @brief 获取修正偏置（值拷贝，线程安全）
     */
    float getCorrectionBias() const { return correction_bias_; }

    /**
     * @brief 生成用于可视化拟合效果的曲线点列
     *
     * 返回两列点：
     * - fitted_curve: 从 t_min (queue_中最旧数据的时间差，负数) 到 t=0 的拟合曲线，
     *                 通过 predict() 生成 (t, r(t))
     * - raw_points:   queue_中的观测数据，t 记为该点到最新数据的时间差（负值），
     *                 格式为 (t, continuous_roll)
     *
     * 调用方需先将此数据转换为合适的像素坐标后绘制。
     *
     * @param fitted_curve  输出：拟合曲线点列
     * @param raw_points    输出：原始观测点列
     * @param num_samples   拟合曲线的采样点数
     */
    void getVisualizationPoints(
        std::vector<std::pair<float, float>>& fitted_curve,
        std::vector<std::pair<float, float>>& raw_points,
        int num_samples = 200) const;

private:
    struct DataPoint
    {
        float continuous_roll;
        std::chrono::steady_clock::time_point timestamp;
    };

    std::deque<DataPoint> queue_;
    float max_time_window_;
    float queue_time_threshold_;  // 队列时间跨度最小阈值，≤ max_time_window_
    int min_data_points_;

    // 拟合参数
    BigParams  big_params_;
    SmallParams small_params_;
    bool       fit_valid_    = false;

    // 当前使用的拟合方法
    FitMethod fit_method_ = FitMethod::BIG;

    // 相位连续化 (unwrap) 状态（仿照 YAxisFilter 的 jump_a_ 机制）
    float last_signed_roll_    = 0.0f;
    float continuous_roll_  = 0.0f;
    int   jump_count_       = 0;
    bool  first_update_     = true;

    // 旋转方向: +1 正转, -1 反转
    int direction_ = 1;

    // 修正偏置
    int   correction_window_ = 3;       // 用于计算偏置的最新数据点数
    float correction_bias_   = 0.0f;

    // 上次更新时间戳（用于 updateWithoutData / performFit 计算时间差 以及 可视化 t=0 参考点）
    std::chrono::steady_clock::time_point last_update_timestamp_;

    // 网格搜索间隔控制
    float grid_search_interval_;                                      // 两次完整网格搜索的最小间隔（秒）
    std::chrono::steady_clock::time_point last_grid_search_timestamp_; // 上次网格搜索时间

    // 自旋轴朝向旋转矩阵
    cv::Mat y_axis_R_;
    // 中心位置
    cv::Vec3f posi_;

    /**
     * @brief 将内部状态重置为初始值（由构造函数和 reset() 共用）
     */
    void resetState();

    /**
     * @brief 检查队列数据是否满足拟合有效性判据：
     *        1. 数据点数 >= min_data_points_
     *        2. 队列时间跨度（最新-最旧） >= queue_time_threshold_
     * @return 满足判据则返回 true
     */
    bool checkFitValidity() const;

    /**
     * @brief 执行最小二乘拟合，结果存入 big_params_, small_params_, fit_valid_
     */
    void performFit();

    TaskPool fit_pool_;   // 持久线程池，构造时分配
};

#endif // ROLL_PREDICTOR_H