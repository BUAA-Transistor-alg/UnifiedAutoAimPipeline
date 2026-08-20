#include "common/CameraProjection.h"

#include <cmath>

CameraProjection::CameraProjection(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
                                   const ImageResolution& resolution) {
    CV_Assert(!camera_matrix.empty());
    CV_Assert(camera_matrix.rows == 3 && camera_matrix.cols == 3);

    camera_matrix_ = camera_matrix.clone();
    dist_coeffs_ = dist_coeffs.clone();
    // 统一为 CV_64F，方便下游使用 at<double> 读取
    if (camera_matrix_.type() != CV_64F) {
        cv::Mat tmp;
        camera_matrix_.convertTo(tmp, CV_64F);
        camera_matrix_ = tmp;
    }
    if (!dist_coeffs_.empty() && dist_coeffs_.type() != CV_64F) {
        cv::Mat tmp;
        dist_coeffs_.convertTo(tmp, CV_64F);
        dist_coeffs_ = tmp;
    }

    width_  = resolution.width;
    height_ = resolution.height;
    computeMaxTan2();
}

void CameraProjection::projectPoints(const std::vector<cv::Point3f>& object_points,
                                     const cv::Mat& rvec, const cv::Mat& tvec,
                                     std::vector<cv::Point2f>& image_points) const {
    cv::projectPoints(object_points, rvec, tvec,
                      camera_matrix_, dist_coeffs_, image_points);
}

bool CameraProjection::solvePnP(const std::vector<cv::Point3f>& object_points,
                                const std::vector<cv::Point2f>& image_points,
                                cv::Mat& rvec, cv::Mat& tvec,
                                bool use_extrinsic_guess,
                                int flags) const {
    return cv::solvePnP(object_points, image_points,
                        camera_matrix_, dist_coeffs_,
                        rvec, tvec, use_extrinsic_guess, flags);
}

void CameraProjection::computeMaxTan2() {
    if (width_ <= 0 || height_ <= 0) {
        max_tan2_ = 0.0;
        return;
    }

    std::vector<cv::Point2f> border_pixels;
    border_pixels.reserve(2 * (width_ + height_));
    for (int x = 0; x < width_; ++x) {
        border_pixels.emplace_back((float)x, 0.0f);
        border_pixels.emplace_back((float)x, (float)(height_ - 1));
    }
    for (int y = 1; y < height_ - 1; ++y) {
        border_pixels.emplace_back(0.0f, (float)y);
        border_pixels.emplace_back((float)(width_ - 1), (float)y);
    }

    std::vector<cv::Point2f> undistorted;
    cv::undistortPoints(border_pixels, undistorted, camera_matrix_, dist_coeffs_);

    double max_val = 0.0;
    for (const auto& p : undistorted) {
        double t2 = (double)p.x * p.x + (double)p.y * p.y;
        if (t2 > max_val) max_val = t2;
    }
    max_tan2_ = max_val;
}

// ── PnP 系 <-> Cam 系转换（静态工具方法）──

cv::Vec3f CameraProjection::pnpRvecToEuler(cv::Mat rvec) {
    if (rvec.empty()) return cv::Vec3f(0, 0, 0);
    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat);

    float pitch = std::asin(-rmat.at<double>(1, 2));
    const float epsilon = 1e-6;
    float yaw, roll;
    if (std::abs(std::cos(pitch)) > epsilon) {
        yaw = std::atan2(-rmat.at<double>(0, 2), rmat.at<double>(2, 2));
        roll = std::atan2(rmat.at<double>(1, 0), rmat.at<double>(1, 1));
    } else {
        roll = 0.0f;
        yaw = std::atan2(rmat.at<double>(2, 0), rmat.at<double>(0, 0));
    }
    return cv::Vec3f(yaw, pitch, roll);
}

cv::Vec3f CameraProjection::pnpTvecToPosi(cv::Mat tvec) {
    return cv::Vec3f(
        static_cast<float>(tvec.at<double>(0)),
        static_cast<float>(tvec.at<double>(1)),
        static_cast<float>(tvec.at<double>(2))
    );
}

cv::Vec3f CameraProjection::pnpToCam_posi(cv::Vec3f posi_pnp) {
    return cv::Vec3f(posi_pnp[0], posi_pnp[2], -posi_pnp[1]);
}

cv::Vec3f CameraProjection::camToPnp_posi(cv::Vec3f posi_cam) {
    return cv::Vec3f(posi_cam[0], -posi_cam[2], posi_cam[1]);
}

cv::Vec3f CameraProjection::pnpTvecToCamPosi(cv::Mat tvec) {
    return pnpToCam_posi(pnpTvecToPosi(tvec));
}

// ── Cam 系输入的高层封装 ──

void CameraProjection::projectPoints_Cam(const std::vector<cv::Point3f>& object_points_cam,
                                         std::vector<cv::Point2f>& image_points) const {
    std::vector<cv::Point3f> pnp_points;
    pnp_points.reserve(object_points_cam.size());
    for (const auto& p : object_points_cam) {
        cv::Vec3f pnp = camToPnp_posi(cv::Vec3f(p.x, p.y, p.z));
        pnp_points.emplace_back(pnp[0], pnp[1], pnp[2]);
    }
    cv::Mat zero_rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat zero_tvec = cv::Mat::zeros(3, 1, CV_64F);
    projectPoints(pnp_points, zero_rvec, zero_tvec, image_points);
}

bool CameraProjection::solvePnP_Cam(const std::vector<cv::Point3f>& object_points_cam,
                                    const std::vector<cv::Point2f>& image_points,
                                    const std::vector<int>& flags,
                                    cv::Vec3f& position_cam,
                                    cv::Vec3f& euler_cam) const {
    if (flags.empty()) {
        return false;
    }

    std::vector<cv::Point3f> pnp_points;
    pnp_points.reserve(object_points_cam.size());
    for (const auto& p : object_points_cam) {
        cv::Vec3f pnp = camToPnp_posi(cv::Vec3f(p.x, p.y, p.z));
        pnp_points.emplace_back(pnp[0], pnp[1], pnp[2]);
    }

    cv::Mat rvec, tvec;
    for (size_t i = 0; i < flags.size(); ++i) {
        bool use_guess = (i > 0);
        if (!solvePnP(pnp_points, image_points, rvec, tvec, use_guess, flags[i])) {
            return false;
        }
    }

    position_cam = pnpTvecToCamPosi(tvec);
    euler_cam = pnpRvecToEuler(rvec);
    return true;
}
