#include "PowerRune/YAxisFilter.h"
#include <cmath>
#include <limits>
#include <algorithm>

// 辅助：将 OpenCV 3x3 浮点矩阵转为 Eigen 四元数
Eigen::Quaternionf YAxisFilter::matToQuaternion(const cv::Mat& R)
{
    // 假设 R 是 3x3 CV_32F 矩阵
    CV_Assert(R.type() == CV_32F && R.rows == 3 && R.cols == 3);
    Eigen::Matrix3f eigR;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            eigR(i, j) = R.at<float>(i, j);
    return Eigen::Quaternionf(eigR).normalized();
}

// 辅助：将 Eigen 四元数转为 OpenCV 3x3 浮点矩阵
cv::Mat YAxisFilter::quaternionToMat(const Eigen::Quaternionf& q)
{
    Eigen::Matrix3f eigR = q.normalized().toRotationMatrix();
    cv::Mat R(3, 3, CV_32F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R.at<float>(i, j) = eigR(i, j);
    return R;
}

// 构造函数
YAxisFilter::YAxisFilter(float alpha_slow, float alpha_fast, float alpha_pos, float alpha_omega, float alpha_reg)
    : alpha_slow_(alpha_slow), alpha_fast_(alpha_fast), alpha_pos_(alpha_pos), alpha_omega_(alpha_omega), alpha_reg_(alpha_reg)
{
    resetState();
}

// 计算旋转矩阵到轴角的辅助函数
static std::pair<Eigen::Vector3f, float> rotationMatrixToAxisAngle(const Eigen::Matrix3f& R)
{
    float cos_theta = (R.trace() - 1.0f) / 2.0f;
    cos_theta = std::max(-1.0f, std::min(1.0f, cos_theta));
    float theta = std::acos(cos_theta);

    Eigen::Vector3f u;
    if (theta < 1e-6f) {
        u = Eigen::Vector3f::UnitX();
        theta = 0.0f;
    } else {
        float s = 2.0f * std::sin(theta);
        u(0) = (R(2, 1) - R(1, 2)) / s;
        u(1) = (R(0, 2) - R(2, 0)) / s;
        u(2) = (R(1, 0) - R(0, 1)) / s;
    }
    return {u, theta};
}

// 计算从 R1 沿测地线旋转到 R2 的过程中，与 R3 的最小夹角（弧度）
float YAxisFilter::minAngleDuringRotation(const Eigen::Matrix3f& R1,
                                           const Eigen::Matrix3f& R2,
                                           const Eigen::Matrix3f& R3)
{
    // 1. R12 = R1^T * R2, 提取轴角 (u, phi)
    Eigen::Matrix3f R12 = R1.transpose() * R2;
    auto [u, phi] = rotationMatrixToAxisAngle(R12);

    if (phi < 1e-6f) {
        // 测地线退化为一个点，直接返回 R1 与 R3 的夹角
        Eigen::Matrix3f dR = R1.transpose() * R3;
        float cos_alpha = (dR.trace() - 1.0f) / 2.0f;
        cos_alpha = std::max(-1.0f, std::min(1.0f, cos_alpha));
        return std::acos(cos_alpha);
    }

    // 2. B = R1^T * R3
    Eigen::Matrix3f B = R1.transpose() * R3;

    // 3. 系数
    float C0 = u.transpose() * B * u;
    float A = B.trace() - C0;

    Eigen::Vector3f v;
    v(0) = (B(2, 1) - B(1, 2)) / 2.0f;
    v(1) = (B(0, 2) - B(2, 0)) / 2.0f;
    v(2) = (B(1, 0) - B(0, 1)) / 2.0f;
    float C = -2.0f * u.dot(v);

    // 4. 驻点并截断到 [0, phi]
    float x_c = std::atan2(C, A);
    float x_star = std::max(0.0f, std::min(phi, x_c));

    // 5. 最大迹值
    float f_max = C0 + A * std::cos(x_star) + C * std::sin(x_star);

    // 6. 最小夹角
    float cos_alpha = (f_max - 1.0f) / 2.0f;
    cos_alpha = std::max(-1.0f, std::min(1.0f, cos_alpha));
    return std::acos(cos_alpha);
}

// 绕四元数自身局部 Y 轴旋转指定角度（弧度）
Eigen::Quaternionf YAxisFilter::rotateAroundLocalY(const Eigen::Quaternionf& q, float angle)
{
    Eigen::Matrix3f R = q.toRotationMatrix();
    Eigen::Vector3f y_axis = R.col(1);
    Eigen::AngleAxisf aa(angle, y_axis);
    Eigen::Quaternionf result = Eigen::Quaternionf(aa) * q;
    result.normalize();
    return result;
}

// 根据累计跳变 jump_count_ 绕自身局部 Y 轴施加反向修正旋转 -2π/5 * jump_count_
Eigen::Quaternionf YAxisFilter::applyJumpCorrection(const Eigen::Quaternionf& q) const
{
    constexpr float TWO_PI_OVER_5 = 2.0f * static_cast<float>(M_PI) / 5.0f;
    return rotateAroundLocalY(q, -static_cast<float>(jump_count_) * TWO_PI_OVER_5);
}

// applyJumpCorrection 的反函数：绕自身局部 Y 轴施加正向修正旋转 +2π/5 * jump_count_
Eigen::Quaternionf YAxisFilter::disapplyJumpCorrection(const Eigen::Quaternionf& q) const
{
    constexpr float TWO_PI_OVER_5 = 2.0f * static_cast<float>(M_PI) / 5.0f;
    return rotateAroundLocalY(q, static_cast<float>(jump_count_) * TWO_PI_OVER_5);
}

// 特殊预测
std::pair<Eigen::Quaternionf, int> YAxisFilter::specialPrediction(
    const Eigen::Quaternionf& q_base,
    const Eigen::Quaternionf& q_R3,
    int a_range)
{
    constexpr float TWO_PI_OVER_5 = 2.0f * static_cast<float>(M_PI) / 5.0f;

    Eigen::Matrix3f R3 = q_R3.toRotationMatrix();

    int best_a = 0;
    float best_angle = std::numeric_limits<float>::max();
    Eigen::Quaternionf best_q = Eigen::Quaternionf::Identity();

    for (int a = -a_range; a <= a_range; ++a) {
        // q_base_a = q_base 绕本体 Y 轴旋转 a * 2π/5
        Eigen::Quaternionf q_base_a = rotateAroundLocalY(q_base, static_cast<float>(a) * TWO_PI_OVER_5);
        Eigen::Matrix3f R_base_a = q_base_a.toRotationMatrix();

        // 计算 R_base_a 与 R3 的夹角
        Eigen::Matrix3f dR = R_base_a.transpose() * R3;
        float cos_angle = (dR.trace() - 1.0f) / 2.0f;
        cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));
        float angle = std::acos(cos_angle);

        if (angle < best_angle) {
            best_angle = angle;
            best_a = a;
            best_q = q_base_a.normalized();
        }
    }

    return {best_q, best_a};
}

// 核心更新（带角速度预测-校正与五边形对称跳变检测，自动初始化）
void YAxisFilter::update(const cv::Vec3f& obs_pos, const cv::Mat& obs_rot,
                         const std::chrono::steady_clock::time_point& frame_timestamp,
                         bool is_continuous,
                         const cv::Mat& roll_predictor_R)
{
    // 自动初始化
    if (!inited_) {
        p_est_ = obs_pos;
        q_est_ = matToQuaternion(obs_rot);
        omega_est_ = 0.0f;
        inited_ = true;
        last_timestamp_ = frame_timestamp;
        return;
    }

    // 内部计算 dt
    float dt = std::chrono::duration<float>(frame_timestamp - last_timestamp_).count();
    last_timestamp_ = frame_timestamp;

    // 保存上一帧的姿态作为 R1（在执行任何修改之前）
    Eigen::Quaternionf q_R1 = q_est_;
    Eigen::Matrix3f R1 = q_R1.toRotationMatrix();

    // 提前将观测转为四元数
    Eigen::Quaternionf q_meas = matToQuaternion(obs_rot);

    if ((dt > 1e-6f)) {
        // ====== 在临时副本上计算 R2（正常预测）和 R3（完整更新后姿态） ======
        Eigen::Quaternionf q_temp = q_est_;
        float omega_temp = omega_est_;

        // 正常预测 → R2
        {
            q_temp = rotateAroundLocalY(q_temp, omega_temp * dt);
        }
        Eigen::Quaternionf q_R2 = q_temp;

        // 完整校正 → R3（使用临时副本，不修改真实状态）
        {
            Eigen::Quaternionf dq = q_meas * q_temp.inverse();
            if (dq.w() < 0.0f) {
                dq.coeffs() = -dq.coeffs();
            }
            // 正确提取轴角（非小角度近似，支持任意角度精确修正）
            float sin_half = dq.vec().norm();              // = |sin(θ/2)|, 无抵消问题
            float half_angle = std::atan2(sin_half, dq.w()); // atan2 比 acos 更稳健
            float angle = 2.0f * half_angle;
            Eigen::Vector3f dtheta;
            if (sin_half > 1e-8f) {
                dtheta = angle * dq.vec() / sin_half;
            } else {
                dtheta = Eigen::Vector3f::Zero();
            }
            Eigen::Matrix3f R_temp = q_temp.toRotationMatrix();
            Eigen::Vector3f y_axis = R_temp.col(1);
            float spin_scalar = dtheta.dot(y_axis);
            Eigen::Vector3f dtheta_spin = spin_scalar * y_axis;
            Eigen::Vector3f dtheta_tilt = dtheta - dtheta_spin;
            Eigen::Vector3f spin_vec = alpha_fast_ * dtheta_spin;
            Eigen::Vector3f tilt_vec = alpha_slow_ * dtheta_tilt;

            // 分别构造旋转：先 spin（绕 y_axis，不改变 y_axis），后 tilt
            Eigen::Quaternionf q_corr = Eigen::Quaternionf::Identity();
            float spin_norm = spin_vec.norm();
            if (spin_norm > 1e-6f) {
                Eigen::AngleAxisf aa_spin(spin_norm, spin_vec / spin_norm);
                q_corr = Eigen::Quaternionf(aa_spin);
            }
            float tilt_norm = tilt_vec.norm();
            if (tilt_norm > 1e-6f) {
                Eigen::AngleAxisf aa_tilt(tilt_norm, tilt_vec / tilt_norm);
                q_corr = Eigen::Quaternionf(aa_tilt) * q_corr;
            }
            q_temp = q_corr * q_temp;
            q_temp.normalize();
        }
        Eigen::Quaternionf q_R3 = q_temp;

        // ====== 判断是否需要特殊预测 ======
        Eigen::Matrix3f R2_mat = q_R2.toRotationMatrix();
        Eigen::Matrix3f R3_mat = q_R3.toRotationMatrix();
        float min_angle = minAngleDuringRotation(R1, R2_mat, R3_mat);

        if ((min_angle > static_cast<float>(M_PI) / 5.0f) || (!is_continuous)) {
            // 特殊预测：确定基准旋转和 a 搜索范围，尝试 ±2π/5 偏移选最接近 R3 的候选
            Eigen::Quaternionf q_base;
            int a_range = 1;
            if ((!is_continuous) && (!roll_predictor_R.empty())) {
                // 优先使用 RollPredictor 的预测旋转矩阵，否则退化为正常预测 R2
                a_range = 2;
                q_base = disapplyJumpCorrection(matToQuaternion(roll_predictor_R));
            } else {
                q_base = rotateAroundLocalY(q_R1, omega_est_ * dt);
            }
            auto [q_candidate, a] = specialPrediction(q_base, q_R3, a_range);
            q_est_ = q_candidate;
            // 更新跳变累计信息
            flip_ = !flip_;
            jump_count_ += a;
        } else {
            // 正常预测
            q_est_ = rotateAroundLocalY(q_est_, omega_est_ * dt);
        }
    }

    // ====== 以下为步骤 2-10：基于（可能经特殊预测调整后的）预测姿态进行校正 ======

    // 2. 计算残差四元数（测量相对于预测后的估计）
    Eigen::Quaternionf dq = q_meas * q_est_.inverse();
    // 保证最短路径（w > 0）
    if (dq.w() < 0.0f) {
        dq.coeffs() = -dq.coeffs();  // 取反，保持 w>0
    }

    // 3. 正确提取轴角（非小角度近似，支持任意角度精确修正）
    float sin_half = dq.vec().norm();              // = |sin(θ/2)|, 无抵消问题
    float half_angle = std::atan2(sin_half, dq.w()); // atan2 比 acos 更稳健
    float angle = 2.0f * half_angle;
    Eigen::Vector3f dtheta;
    if (sin_half > 1e-8f) {
        dtheta = angle * dq.vec() / sin_half;
    } else {
        dtheta = Eigen::Vector3f::Zero();
    }

    // 4. 提取当前估计的物体 Y 轴在世界坐标系下的方向（旋转矩阵的第二列）
    Eigen::Matrix3f R_est = q_est_.toRotationMatrix();
    Eigen::Vector3f y_axis = R_est.col(1);  // 物体 Y 轴

    // 5. 分解误差为平行（绕轴自旋）和垂直（轴倾斜）分量
    float spin_scalar = dtheta.dot(y_axis);                  // 投影长度 (rad)
    Eigen::Vector3f dtheta_spin = spin_scalar * y_axis;      // 平行分量
    Eigen::Vector3f dtheta_tilt = dtheta - dtheta_spin;      // 垂直分量

    // 6. 角速度校正：spin_scalar 是预测后剩余的角误差，spin_scalar/dt 即角速度误差
    if (is_continuous && dt > 1e-6f) {
        omega_est_ += alpha_omega_ * spin_scalar / dt;
    }

    // 7. 分别应用不同增益到姿态修正
    Eigen::Vector3f spin_vec = alpha_fast_ * dtheta_spin;
    Eigen::Vector3f tilt_vec = alpha_slow_ * dtheta_tilt;

    // 8. 正则化项：促使局部Y轴尽可能平行于全局XY平面
    //    Cost: C = 0.5 * yz² (y轴Z分量越小越好)
    //    ∇C = yz * (∂yz/∂yx, ∂yz/∂yy, ∂yz/∂yz) → 对旋转扰动 dθ，梯度 = yz * (yy, -yx, 0)
    //    dθ_reg = -alpha_reg * yz * (yy, -yx, 0)   (梯度下降，且该方向 ⊥ y_axis，纯 tilt)
    if (alpha_reg_ > 0.0f) {
        float yz = y_axis.z();
        Eigen::Vector3f dtheta_reg(-alpha_reg_ * yz * y_axis.y(),
                                    alpha_reg_ * yz * y_axis.x(),
                                    0.0f);
        tilt_vec += dtheta_reg;
    }

    // 8. 分别构造修正四元数：先 spin（绕 y_axis，不改变 y_axis），后 tilt
    Eigen::Quaternionf q_corr = Eigen::Quaternionf::Identity();
    float spin_norm = spin_vec.norm();
    if (spin_norm > 1e-6f) {
        Eigen::AngleAxisf aa_spin(spin_norm, spin_vec / spin_norm);
        q_corr = Eigen::Quaternionf(aa_spin);
    }
    float tilt_norm = tilt_vec.norm();
    if (tilt_norm > 1e-6f) {
        Eigen::AngleAxisf aa_tilt(tilt_norm, tilt_vec / tilt_norm);
        q_corr = Eigen::Quaternionf(aa_tilt) * q_corr;
    }

    // 9. 更新姿态（左乘修正）
    q_est_ = q_corr * q_est_;
    q_est_.normalize();  // 防止数值漂移

    // 10. 更新位置（慢增益）
    p_est_ += alpha_pos_ * (obs_pos - p_est_);
}

// 将内部状态重置为初始值（由构造函数和 reset() 共用）
void YAxisFilter::resetState()
{
    q_est_ = Eigen::Quaternionf::Identity();
    p_est_ = cv::Vec3f(0.0f, 0.0f, 0.0f);
    omega_est_ = 0.0f;
    flip_ = false;
    jump_count_ = 0;
    inited_ = false;
}

// 重置滤波器，使其回到刚构造完、未调用 update 的初始状态
// 若已经处于重置状态（从未初始化或已 reset），则自动跳过，避免重复重置
void YAxisFilter::reset()
{
    if (!inited_) {
        return;  // 已经处于重置状态，跳过
    }
    resetState();
}

// 获取滤波后的位置
cv::Vec3f YAxisFilter::getPosition() const
{
    return p_est_;
}

// 获取滤波后的姿态矩阵
cv::Mat YAxisFilter::getRotation(bool apply_jump_correction) const
{
    if (!apply_jump_correction) {
        return quaternionToMat(q_est_);
    }
    return quaternionToMat(applyJumpCorrection(q_est_));
}

// 获取当前估计的绕 Y 轴角速度
float YAxisFilter::getAngularVelocity() const
{
    return omega_est_;
}

// 获取跳变累计计数 a
int YAxisFilter::getJumpA() const
{
    return jump_count_;
}

// 获取当前 flip 状态
bool YAxisFilter::getFlip() const
{
    return flip_;
}

// 对输入的旋转次数向量做跳变修正：每个值 +jump_count_ 后对5取余，排序后返回
std::vector<int> YAxisFilter::processRotationCounts(const std::vector<int>& rotation_counts) const
{
    std::vector<int> result;
    result.reserve(rotation_counts.size());
    for (int v : rotation_counts) {
        int corrected = (v + jump_count_) % 5;
        if (corrected < 0) corrected += 5;
        result.push_back(corrected);
    }
    std::sort(result.begin(), result.end());
    return result;
}

// 前向预测：基于当前滤波姿态和角速度，预测 delta_t 秒后的旋转矩阵
cv::Mat YAxisFilter::predictRotation(float delta_t, bool apply_jump_correction) const
{
    // 绕 Y 轴旋转 omega_est * delta_t
    Eigen::Quaternionf q_pred = rotateAroundLocalY(q_est_, omega_est_ * delta_t);

    if (apply_jump_correction) {
        q_pred = applyJumpCorrection(q_pred);
    }

    return quaternionToMat(q_pred);
}