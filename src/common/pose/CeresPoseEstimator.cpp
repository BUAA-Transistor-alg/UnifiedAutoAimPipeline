#include "common/pose/CeresPoseEstimator.h"

#include <cassert>
#include <cmath>

void CeresPoseEstimator::initFrom(const CameraProjection& camera_proj) {
    const cv::Mat& camera_matrix = camera_proj.getCameraMatrix();
    const cv::Mat& dist_coeffs  = camera_proj.getDistCoeffs();
    CV_Assert(camera_matrix.rows == 3 && camera_matrix.cols == 3);
    K_[0] = camera_matrix.at<double>(0, 0);
    K_[1] = camera_matrix.at<double>(1, 1);
    K_[2] = camera_matrix.at<double>(0, 2);
    K_[3] = camera_matrix.at<double>(1, 2);
    if (dist_coeffs.total() >= 4) {
        D_[0] = dist_coeffs.at<double>(0);
        D_[1] = dist_coeffs.at<double>(1);
        D_[2] = dist_coeffs.at<double>(2);
        D_[3] = dist_coeffs.at<double>(3);
        D_[4] = (dist_coeffs.total() >= 5) ? dist_coeffs.at<double>(4) : 0.0;
    } else {
        D_[0] = 0.0; D_[1] = 0.0; D_[2] = 0.0; D_[3] = 0.0; D_[4] = 0.0;
    }
    width_    = camera_proj.getWidth();
    height_   = camera_proj.getHeight();
    max_tan2_ = camera_proj.getMaxTan2();
}

CeresPoseEstimator::CeresPoseEstimator(const CameraProjection& camera_proj) {
    initFrom(camera_proj);
}

CeresPoseEstimator::CeresPoseEstimator(std::shared_ptr<CameraProjection> camera_proj) {
    CV_Assert(camera_proj != nullptr);
    initFrom(*camera_proj);
}

// ── solve：物体轴-方向夹角硬约束模式（无初始位姿重载）──
// 不传初始位姿 → 从默认位姿（rvec=0、tvec 前向 z=1）开始优化（从头求解）。
bool CeresPoseEstimator::solve(
    const std::vector<cv::Point3f>& points_3d,
    const std::vector<cv::Point2f>& points_2d,
    cv::Mat& rvec, cv::Mat& tvec,
    const std::vector<AxisCosConstraintParam>& constraints)
{
    return solve(points_3d, points_2d, rvec, tvec,
                 cv::Mat(), cv::Mat(), constraints);
}

// ── solve：物体轴-方向夹角硬约束模式（带初始位姿重载）──
bool CeresPoseEstimator::solve(
    const std::vector<cv::Point3f>& points_3d,
    const std::vector<cv::Point2f>& points_2d,
    cv::Mat& rvec, cv::Mat& tvec,
    const cv::Mat& rvec_init, const cv::Mat& tvec_init,
    const std::vector<AxisCosConstraintParam>& constraints)
{
    return solveImpl(points_3d, points_2d, rvec, tvec,
                     rvec_init, tvec_init, constraints);
}

// ── 共同求解实现 ──
bool CeresPoseEstimator::solveImpl(
    const std::vector<cv::Point3f>& points_3d,
    const std::vector<cv::Point2f>& points_2d,
    cv::Mat& rvec, cv::Mat& tvec,
    const cv::Mat& rvec_init, const cv::Mat& tvec_init,
    const std::vector<AxisCosConstraintParam>& constraints)
{
    // ── 初始位姿：优先采用传入的初值（进一步优化/精化）；
    //    未传初值时使用默认位姿（角轴全零、沿光轴前向 1m，从头求解）──
    double camera_rvec[3] = {0.0, 0.0, 0.0};
    double camera_tvec[3] = {0.0, 0.0, 1.0};
    if (!rvec_init.empty() && !tvec_init.empty()) {
        CV_Assert(rvec_init.type() == CV_64F && rvec_init.rows == 3 && rvec_init.cols == 1);
        CV_Assert(tvec_init.type() == CV_64F && tvec_init.rows == 3 && tvec_init.cols == 1);
        camera_rvec[0] = rvec_init.at<double>(0);
        camera_rvec[1] = rvec_init.at<double>(1);
        camera_rvec[2] = rvec_init.at<double>(2);
        camera_tvec[0] = tvec_init.at<double>(0);
        camera_tvec[1] = tvec_init.at<double>(1);
        camera_tvec[2] = tvec_init.at<double>(2);
    }

    rvec.create(3, 1, CV_64F);
    tvec.create(3, 1, CV_64F);

    ceres::Problem problem;
    ceres::LossFunction* lossfunction = NULL;
    for (size_t i = 0; i < points_3d.size(); i++) {
        Eigen::Vector3d p3d(points_3d[i].x, points_3d[i].y, points_3d[i].z);
        Eigen::Vector2d p2d(points_2d[i].x, points_2d[i].y);

        ceres::CostFunction* costfunction = ReprojectionError::Create(p3d, p2d, K_, D_);
        problem.AddResidualBlock(costfunction, lossfunction, camera_rvec, camera_tvec);

        // ── FoV 硬边界约束：仅对画面内的2D关键点生效，将3D点限制在相机视锥内 ──
        if (max_tan2_ > 0.0 && width_ > 0 && height_ > 0) {
            float u = points_2d[i].x;
            float v = points_2d[i].y;
            if (u >= 0.0f && u < (float)width_ && v >= 0.0f && v < (float)height_) {
                ceres::CostFunction* fov_cost = FoVConstraint::Create(p3d, max_tan2_);
                // 硬边界：使用极大的权重，使约束近似为硬边界
                ceres::LossFunction* fov_loss = new ceres::ScaledLoss(
                    NULL, 1e8, ceres::TAKE_OWNERSHIP);
                problem.AddResidualBlock(fov_cost, fov_loss, camera_rvec, camera_tvec);
            }
        }
    }

    // ── 物体轴-方向夹角硬等号约束（constraints 非空时全程生效）──
    // 每条约束：轴与方向向量均为 cam 系坐标；points_3d 为 cam 系点经 camToPnp
    // 换算的 PnP 点，故轴/方向统一按 camToPnp 转到 PnP 系并归一化后再做
    // (R·axis)·dir，等价于 cam 系下的夹角余弦。与 FoV 硬边界同样使用 ~1e8 权重，
    // 使该等号约束近似为硬边界（不依赖 3D 点是否落在画面内）。
    for (const AxisCosConstraintParam& c : constraints) {
        // cam 系 → PnP 系
        const cv::Vec3f axis_pnp = CameraProjection::camToPnp_posi(c.axis_body_cam);
        const cv::Vec3f dir_pnp  = CameraProjection::camToPnp_posi(c.dir_cam);

        const double axis_norm = std::sqrt((double)axis_pnp[0] * axis_pnp[0] +
                                           (double)axis_pnp[1] * axis_pnp[1] +
                                           (double)axis_pnp[2] * axis_pnp[2]);
        const double dir_norm = std::sqrt((double)dir_pnp[0] * dir_pnp[0] +
                                          (double)dir_pnp[1] * dir_pnp[1] +
                                          (double)dir_pnp[2] * dir_pnp[2]);
        CV_Assert(axis_norm > 1e-12);  // 被约束轴退化为零向量：无意义
        CV_Assert(dir_norm  > 1e-12);  // 方向向量退化为零向量：无意义
        CV_Assert(c.target_cos >= -1.0 - 1e-9 && c.target_cos <= 1.0 + 1e-9);  // 余弦须合法

        // 归一化（保证点乘即夹角余弦）；目标余弦数值安全钳位到 [-1, 1]
        Eigen::Vector3d axis_unit(axis_pnp[0] / axis_norm, axis_pnp[1] / axis_norm,
                                  axis_pnp[2] / axis_norm);
        Eigen::Vector3d dir_unit(dir_pnp[0] / dir_norm, dir_pnp[1] / dir_norm,
                                 dir_pnp[2] / dir_norm);
        const double cos_clamped = std::max(-1.0, std::min(1.0, c.target_cos));

        ceres::CostFunction* cos_cost = AxisCosConstraint::Create(axis_unit, dir_unit, cos_clamped);
        ceres::LossFunction* cos_loss = new ceres::ScaledLoss(
            NULL, 1e8, ceres::TAKE_OWNERSHIP);
        problem.AddResidualBlock(cos_cost, cos_loss, camera_rvec);
    }

    // 设置优化器参数及优化方法
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 100;
    // 硬性时间上限 50ms：超时即停止（配合步数上限双保险，避免拖慢实时流水线；
    // 实际收敛通常远早于两者，仅在步数/收敛判据内提前 CONVERGENCE）
    options.max_solver_time_in_seconds = 0.05;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = false;

    // 调用求解器进行优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 对优化后的rt进行赋值还原
    rvec.at<double>(0, 0) = camera_rvec[0];
    rvec.at<double>(1, 0) = camera_rvec[1];
    rvec.at<double>(2, 0) = camera_rvec[2];
    tvec.at<double>(0, 0) = camera_tvec[0];
    tvec.at<double>(1, 0) = camera_tvec[1];
    tvec.at<double>(2, 0) = camera_tvec[2];

    // 数值异常（如初始化退化导致无法求解）视为失败
    const bool converged = (summary.termination_type != ceres::FAILURE &&
                            summary.termination_type != ceres::USER_FAILURE);
    return converged;
}
