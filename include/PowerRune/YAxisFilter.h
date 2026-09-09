#ifndef Y_AXIS_FILTER_H
#define Y_AXIS_FILTER_H

#include <opencv2/core.hpp>
#include <Eigen/Geometry>
#include <utility>
#include <chrono>

/**
 * @brief 针对"绕物体自身 Y 轴快速旋转，位置及轴倾斜缓慢变化"的专用滤波器。
 * 
 * 输入：观测到的物体位置 (cv::Vec3f) 和姿态旋转矩阵 (cv::Mat, 3x3 float)
 * 输出：滤波后的位置和姿态矩阵。
 * 
 * 核心原理：将姿态误差分解为绕物体当前 Y 轴的旋转（快通道）和垂直方向的倾斜（慢通道），
 * 分别应用不同增益，从而实现位置和轴方向平滑、绕轴角度快速跟踪。
 * 
 * 绕 Y 轴旋转分量额外维护角速度状态 omega_est_，通过预测-校正实现对匀速/慢变转速
 * 的精确估计与跟踪。
 * 
 * 当观测跳变超过阈值时（例如由五边形对称性导致的 2π/5 整数倍跳变），
 * 自动执行特殊预测：在正常预测基础上尝试 ±2π/5 偏移，选择最接近观测的候选姿态。
 *
 * 异常值防护（见 update() 返回值 UpdateResult）：
 *  - 输入含 NaN / ±inf / 绝对值超过 kAnomalyAbsLimit 的异常值时，跳过该输入
 *    （等同该帧未识别到物体），内部状态保持不变；
 *  - 更新后输出状态出现异常值时立即自动重置，异常值不会流出到下一级滤波。
 * 可选位置坐标限位（setPositionLimits）：把滤波位置 xy 限制在以 chassis 系原点在
 * world 系下 xy 为圆心的圆内、z 限制在相对 chassis z 的 ± 半范围内。
 */
class YAxisFilter
{
public:
    /// 异常值判定阈值：任何输入/状态分量的绝对值超过该值即视为异常（NaN/±inf 同理）
    static constexpr float kAnomalyAbsLimit = 1e6f;

    /**
     * @brief update() 的更新结果状态
     */
    enum class UpdateResult {
        NORMAL,                 ///< 本帧正常处理：状态已推进
        INPUT_INVALID_SKIPPED,  ///< 输入含异常值（NaN/inf/绝对值>kAnomalyAbsLimit），
                                ///< 本帧被跳过（等同该帧未识别到物体），内部状态保持不变
        OUTPUT_INVALID_RESET    ///< 更新后输出状态出现异常值，已立即自动重置
                                ///< （状态回到未初始化；调用方不应把本帧输出送入下一级滤波）
    };

    /**
     * @brief 构造函数
     * @param alpha_slow   轴倾斜（垂直方向）的增益，推荐 0.01~0.05，越小越平滑
     * @param alpha_fast   绕轴自旋（平行方向）的增益，推荐 1.0，实现快速响应
     * @param alpha_pos    位置更新增益，推荐 0.01~0.05，越小越平滑
     * @param alpha_omega  角速度更新的增益，推荐 0.01~0.1，越小越平滑
     * @param alpha_reg    正则化增益，促使局部Y轴平行于全局XY平面，0 表示禁用
     */
    YAxisFilter(float alpha_slow  = 0.05f,
                float alpha_fast  = 1.0f,
                float alpha_pos   = 0.05f,
                float alpha_omega = 0.05f,
                float alpha_reg   = 1.0f);

    /**
     * @brief 配置位置坐标限位（相对 chassis 系原点在 world 系下的坐标）：
     *        - xy：限在以 chassis_world_origin 的 xy 为圆心、半径 xy_radius 的圆内；
     *        - z ：限在 chassis_world_origin.z ± z_half_range 内。
     *        配置后，滤波位置 p_est_（含初始化与每次更新后的状态）都会被钳制在该范围内，
     *        保证滤波输出的坐标不会越过该物理范围。
     * @param chassis_world_origin  chassis 系原点在 world 系下的坐标（米）
     * @param xy_radius             xy 限位圆半径（米，>0 时启用限位）
     * @param z_half_range          z 相对 chassis_world_origin.z 的半范围（米，>0 时启用）
     */
    void setPositionLimits(const cv::Vec3f& chassis_world_origin,
                           float xy_radius, float z_half_range);

    /**
     * @brief 关闭位置坐标限位（构造后的默认状态）
     */
    void disablePositionLimits();

    /**
     * @brief 核心更新函数，自动检测首次调用并初始化。
     *        输入观测值和时间戳，内部自动计算时间步长 dt。
     *
     * 异常值处理：
     *  - 输入异常（观测位置/旋转含 NaN、±inf 或绝对值超过 kAnomalyAbsLimit 的分量）时，
     *    本帧被跳过（返回 INPUT_INVALID_SKIPPED，等同该帧未识别到物体），内部状态保持不变；
     *  - 更新完成后若发现输出状态（滤波位置/姿态/角速度）出现异常值，立即自动重置
     *    滤波器并返回 OUTPUT_INVALID_RESET，调用方不应把本帧输出送入下一级滤波。
     *
     * @param obs_pos           观测到的位置
     * @param obs_rot           观测到的姿态旋转矩阵（3x3 float）
     * @param frame_timestamp   当前帧的时间戳，内部用于计算 dt
     * @param is_continuous     是否为连续帧（dt < 阈值）
     * @param roll_predictor_R  当 is_continuous==false 时，可选的 RollPredictor 预测旋转矩阵，
     *                          用于替代 R2 作为 specialPrediction 的基准
     * @return UpdateResult::NORMAL 正常；INPUT_INVALID_SKIPPED 输入异常被跳过；
     *         OUTPUT_INVALID_RESET 输出异常、已自动重置
     */
    UpdateResult update(const cv::Vec3f& obs_pos, const cv::Mat& obs_rot,
                        const std::chrono::steady_clock::time_point& frame_timestamp,
                        bool is_continuous = true,
                        const cv::Mat& roll_predictor_R = cv::Mat());

    /**
     * @brief 获取当前滤波后的位置
     */
    cv::Vec3f getPosition() const;

    /**
     * @brief 获取当前滤波后的姿态旋转矩阵（3x3 float）
     * @param apply_jump_correction 若为 true，则在返回值上额外绕物体Y轴旋转 2π/5 * jump_count_
     */
    cv::Mat getRotation(bool apply_jump_correction = false) const;

    /**
     * @brief 获取当前估计的绕 Y 轴角速度（rad/s）
     */
    float getAngularVelocity() const;

    /**
     * @brief 获取跳变累计计数 a
     */
    int getJumpA() const;

    /**
     * @brief 获取当前 flip 状态
     */
    bool getFlip() const;

    /**
     * @brief 对输入的旋转次数向量做跳变修正：每个值 +jump_count_ 后对5取余，排序后返回
     * @param rotation_counts 输入的旋转次数向量
     * @return 修正并排序后的旋转次数向量
     */
    std::vector<int> processRotationCounts(const std::vector<int>& rotation_counts) const;

    /**
     * @brief 利用当前滤波姿态矩阵和角速度进行前向预测
     * @param delta_t                预测时间（秒），>0 表示向未来预测
     * @param apply_jump_correction  若为 true，则在返回值上额外绕物体Y轴旋转 2π/5 * jump_count_
     * @return 预测时间后的姿态旋转矩阵（3x3 float），仅绕 Y 轴旋转
     */
    cv::Mat predictRotation(float delta_t, bool apply_jump_correction = false) const;

    /**
     * @brief 重置滤波器，使其回到刚构造完、未调用 update 的初始状态。
     *        若已经处于重置状态（从未初始化或已 reset），则自动跳过，避免重复重置。
     */
    void reset();

private:
    // 辅助转换函数
    static Eigen::Quaternionf matToQuaternion(const cv::Mat& R);
    static cv::Mat quaternionToMat(const Eigen::Quaternionf& q);

    /**
     * @brief 计算从 R1 沿测地线旋转到 R2 的过程中，与 R3 的最小夹角（弧度）
     * @param R1  起始旋转矩阵
     * @param R2  终点旋转矩阵
     * @param R3  目标旋转矩阵
     * @return 最小夹角（弧度），范围 [0, π]
     */
    static float minAngleDuringRotation(const Eigen::Matrix3f& R1,
                                         const Eigen::Matrix3f& R2,
                                         const Eigen::Matrix3f& R3);

    /**
     * @brief 特殊预测：在基准旋转 q_base 上尝试 ±2π/5 偏移，选择最接近 q_R3 的候选
     * @param q_base   基准旋转（由调用方预先确定：正常预测 R2 或 RollPredictor 预测矩阵）
     * @param q_R3     完整正常更新后的姿态，作为目标
     * @param a_range  偏移系数 a 的搜索半范围（例如1→a∈{-1,0,1}；2→a∈{-2,-1,0,1,2}）
     * @return pair<候选四元数, 所选偏移系数 a∈{-a_range,...,a_range}>
     */
    static std::pair<Eigen::Quaternionf, int> specialPrediction(
        const Eigen::Quaternionf& q_base,
        const Eigen::Quaternionf& q_R3,
        int a_range);

    /**
     * @brief 绕四元数自身局部 Y 轴旋转指定角度（弧度）
     * @param q      原始四元数
     * @param angle  绕局部 Y 轴旋转的角度（弧度）
     * @return 旋转后的四元数
     */
    static Eigen::Quaternionf rotateAroundLocalY(const Eigen::Quaternionf& q, float angle);

    /**
     * @brief 根据累计跳变 jump_count_ 绕自身局部 Y 轴施加反向修正旋转 -2π/5 * jump_count_
     * @param q 原始四元数
     * @return 修正后的四元数
     */
    Eigen::Quaternionf applyJumpCorrection(const Eigen::Quaternionf& q) const;

    /**
     * @brief applyJumpCorrection 的反函数：根据累计跳变 jump_count_ 绕自身局部 Y 轴施加
     *        正向修正旋转 +2π/5 * jump_count_，用于撤销跳变修正
     * @param q 原始四元数（通常是已经过 applyJumpCorrection 修正的四元数）
     * @return 撤销修正后的四元数
     */
    Eigen::Quaternionf disapplyJumpCorrection(const Eigen::Quaternionf& q) const;

    // 时间追踪（用于内部计算 dt）
    std::chrono::steady_clock::time_point last_timestamp_;

    // 初始化状态
    bool inited_ = false;

    // 状态
    cv::Vec3f p_est_;                 // 滤波位置
    Eigen::Quaternionf q_est_;        // 滤波姿态（四元数）
    float omega_est_;                 // 绕 Y 轴角速度估计 (rad/s)

    // 跳变累计信息
    bool flip_ = false;
    int  jump_count_ = 0;

    // 将状态重置为初始值（由构造函数和 reset() 共用）
    void resetState();

    // 将位置钳制到配置的坐标范围内（未启用限位时原样返回）
    cv::Vec3f clampPositionToLimits(const cv::Vec3f& p) const;

    // 位置坐标限位配置（默认关闭，见 setPositionLimits / disablePositionLimits）
    bool      range_enabled_  = false;
    cv::Vec3f pos_center_     = cv::Vec3f(0.0f, 0.0f, 0.0f);  // chassis 系原点在 world 系坐标（米）
    float     xy_radius_      = 0.0f;  // xy 限位圆半径（米）
    float     z_half_range_   = 0.0f;  // z 相对 pos_center_.z 的半范围（米）

    // 增益参数
    float alpha_slow_;
    float alpha_fast_;
    float alpha_pos_;
    float alpha_omega_;
    float alpha_reg_;
};

#endif // Y_AXIS_FILTER_H