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

// ── 原接口（无 Z 轴夹角约束）→ 内部走共同实现，约束关闭 ──
bool CeresPoseEstimator::solve(
    const std::vector<cv::Point3f>& points_3d,
    const std::vector<cv::Point2f>& points_2d,
    cv::Mat& rvec, cv::Mat& tvec,
    const cv::Mat& rvec_init, const cv::Mat& tvec_init)
{
    return solveImpl(points_3d, points_2d, rvec, tvec,
                     rvec_init, tvec_init, false, nullptr, nullptr, 0.0);
}

// ── Z 轴夹角硬约束模式（无初始位姿重载）──
bool CeresPoseEstimator::solve(
    const std::vector<cv::Point3f>& points_3d,
    const std::vector<cv::Point2f>& points_2d,
    cv::Mat& rvec, cv::Mat& tvec,
    const cv::Vec3f& dir_cam, double target_cos)
{
    return solve(points_3d, points_2d, rvec, tvec,
                 cv::Mat(), cv::Mat(), dir_cam, target_cos);
}

// ── Z 轴夹角硬约束模式（带初始位姿重载）──
bool CeresPoseEstimator::solve(
    const std::vector<cv::Point3f>& points_3d,
    const std::vector<cv::Point2f>& points_2d,
    cv::Mat& rvec, cv::Mat& tvec,
    const cv::Mat& rvec_init, const cv::Mat& tvec_init,
    const cv::Vec3f& dir_cam, double target_cos)
{
    // 被约束轴 = 物体在 cam 系下的 +z：(0,0,1)_cam。
    // points_3d 为 cam 系点经 camToPnp 换算得到的 PnP 系点，故该轴在 PnP 局部系
    // 中表示为 camToPnp((0,0,1)) = (0,-1,0)，求解器旋转矩阵作用的对象是它；
    // dir_cam 同样按 camToPnp 换算到 PnP 系并归一化（保证点乘即 cam 系夹角余弦）。
    const cv::Vec3f axis_pnp = CameraProjection::camToPnp_posi(cv::Vec3f(0.0f, 0.0f, 1.0f));
    double axis[3] = { static_cast<double>(axis_pnp[0]),
                       static_cast<double>(axis_pnp[1]),
                       static_cast<double>(axis_pnp[2]) };

    const cv::Vec3f dir_pnp = CameraProjection::camToPnp_posi(dir_cam);
    double dir[3] = { static_cast<double>(dir_pnp[0]),
                      static_cast<double>(dir_pnp[1]),
                      static_cast<double>(dir_pnp[2]) };
    const double norm = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    CV_Assert(norm > 1e-12);  // 方向向量退化为零向量：无意义
    CV_Assert(target_cos >= -1.0 - 1e-9 && target_cos <= 1.0 + 1e-9);  // 余弦须合法

    // 归一化（保证点乘即夹角余弦）；目标余弦数值安全钳位到 [-1, 1]
    dir[0] /= norm;
    dir[1] /= norm;
    dir[2] /= norm;
    const double cos_clamped = std::max(-1.0, std::min(1.0, target_cos));

    return solveImpl(points_3d, points_2d, rvec, tvec,
                     rvec_init, tvec_init, true, axis, dir, cos_clamped);
}

// ── 共同求解实现 ──
bool CeresPoseEstimator::solveImpl(
    const std::vector<cv::Point3f>& points_3d,
    const std::vector<cv::Point2f>& points_2d,
    cv::Mat& rvec, cv::Mat& tvec,
    const cv::Mat& rvec_init, const cv::Mat& tvec_init,
    bool enforce_z_axis_cos,
    const double axis_pnp_unit[3],
    const double dir_pnp_unit[3], double target_cos)
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

    // ── Z 轴夹角硬等号约束（新模式开启时全程生效）──
    // 与 FoV 硬边界同样使用 ~1e8 权重，使 (R·物体cam-z轴PnP表示)·dir_pnp_unit
    // == target_cos 近似为硬等号约束（不依赖 3D 点是否落在画面内）。
    if (enforce_z_axis_cos) {
        Eigen::Vector3d axis_unit(axis_pnp_unit[0], axis_pnp_unit[1], axis_pnp_unit[2]);
        Eigen::Vector3d dir_unit(dir_pnp_unit[0], dir_pnp_unit[1], dir_pnp_unit[2]);
        ceres::CostFunction* z_cost = ZAxisCosConstraint::Create(axis_unit, dir_unit, target_cos);
        ceres::LossFunction* z_loss = new ceres::ScaledLoss(
            NULL, 1e8, ceres::TAKE_OWNERSHIP);
        problem.AddResidualBlock(z_cost, z_loss, camera_rvec);
    }

    // 设置优化器参数及优化方法
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 100;
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
