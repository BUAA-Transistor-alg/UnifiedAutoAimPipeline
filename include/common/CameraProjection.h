// CameraProjection.h — 相机投影封装
// 统一封装相机内参矩阵、畸变系数与画面分辨率，
// 并提供 cv::projectPoints / cv::solvePnP 的封装方法。
// 构造时计算视场角最大 tan²θ（FoV 硬边界 a = max(tan²θ)）。
#ifndef CAMERA_PROJECTION_H
#define CAMERA_PROJECTION_H

#include <vector>
#include <opencv2/opencv.hpp>

// 图像分辨率结构体（仅宽和高）
struct ImageResolution {
    int width  = 0;  // 图像宽
    int height = 0;  // 图像高
};

class CameraProjection {
public:
    CameraProjection(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
                     const ImageResolution& resolution);

    // ── 各参数获取方法 ──
    const cv::Mat& getCameraMatrix() const { return camera_matrix_; }
    const cv::Mat& getDistCoeffs()  const { return dist_coeffs_; }
    int   getWidth()  const { return width_; }
    int   getHeight() const { return height_; }
    double getMaxTan2() const { return max_tan2_; }

    // ── 封装 cv::projectPoints ──
    void projectPoints(const std::vector<cv::Point3f>& object_points,
                       const cv::Mat& rvec, const cv::Mat& tvec,
                       std::vector<cv::Point2f>& image_points) const;

    // ── 封装 cv::solvePnP ──
    bool solvePnP(const std::vector<cv::Point3f>& object_points,
                  const std::vector<cv::Point2f>& image_points,
                  cv::Mat& rvec, cv::Mat& tvec,
                  bool use_extrinsic_guess = false,
                  int flags = cv::SOLVEPNP_ITERATIVE) const;

    // ── PnP 系 <-> Cam 系转换（静态工具方法）──
    static cv::Vec3f pnpRvecToEuler(cv::Mat rvec);
    static cv::Vec3f pnpTvecToPosi(cv::Mat tvec);
    static cv::Vec3f pnpToCam_posi(cv::Vec3f posi_pnp);
    static cv::Vec3f camToPnp_posi(cv::Vec3f posi_cam);
    static cv::Vec3f pnpTvecToCamPosi(cv::Mat tvec);

    // ── Cam 系输入的高层封装 ──
    // 将 Cam 系下的点转换到 PnP 系，并以零 rvec/tvec 调用 projectPoints。
    void projectPoints_Cam(const std::vector<cv::Point3f>& object_points_cam,
                           std::vector<cv::Point2f>& image_points) const;

    // 将 Cam 系下的点转换到 PnP 系，按 flags 顺序依次调用 solvePnP
    // （除第一个外均设置 use_extrinsic_guess=true），最终结果转回 Cam 系。
    bool solvePnP_Cam(const std::vector<cv::Point3f>& object_points_cam,
                      const std::vector<cv::Point2f>& image_points,
                      const std::vector<int>& flags,
                      cv::Vec3f& position_cam,
                      cv::Vec3f& euler_cam) const;

private:
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    int     width_;
    int     height_;
    double  max_tan2_;

    // ── 遍历图像边缘所有像素，用 cv::undistortPoints 反算各像素对应的光线方向，
    //     取 max(tan²θ) 存入 max_tan2_ ──
    void computeMaxTan2();
};

#endif // CAMERA_PROJECTION_H
