#include "common/pose/CeresPoseEstimator.h"

#include <cassert>

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

bool CeresPoseEstimator::solve(
    const std::vector<cv::Point3f>& points_3d,
    const std::vector<cv::Point2f>& points_2d,
    cv::Mat& rvec, cv::Mat& tvec,
    const cv::Mat& rvec_init, const cv::Mat& tvec_init)
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
