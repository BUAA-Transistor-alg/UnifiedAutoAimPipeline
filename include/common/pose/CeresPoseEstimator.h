// CeresPoseEstimator.h — 基于 Ceres 的手动 PnP 位姿优化
// 替代/精化 OpenCV solvePnP：使用 Ceres 的非线性最小二乘优化重投影误差。
// 与 CameraProjection 同置于 common/pose/，由 solvePnP_Cam 的自定义 flag
// （CameraProjection::SOLVEPNP_CERES）调用。
//
// 支持两种用法：
//   1) 传入初始位姿（rvec_init/tvec_init）→ 以该位姿为初值做进一步优化（精化），
//      适合接在 cv::solvePnP 等粗解之后使用；
//   2) 不传初始位姿 → 从默认位姿（rvec=0、tvec 前向 z=1）开始优化（从头求解）。
#ifndef CERES_POSE_ESTIMATOR_H
#define CERES_POSE_ESTIMATOR_H

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>

#include "common/pose/CameraProjection.h"

class CeresPoseEstimator {
public:
    // ── 构造函数：从相机投影对象拷贝内参/畸变/分辨率与 FoV 硬边界 a = max(tan²θ) ──
    explicit CeresPoseEstimator(const CameraProjection& camera_proj);
    explicit CeresPoseEstimator(std::shared_ptr<CameraProjection> camera_proj);

    // ── PnP 求解 ──
    // points_3d / points_2d：PnP 系 3D 点与对应图像点（点数相同且 ≥ 4）。
    // rvec_init / tvec_init：可选初始位姿（CV_64F 3x1）。两者均非空 → 以初始位姿
    //   进一步优化；为空（缺省）→ 从默认位姿开始优化。
    // 输出 rvec/tvec（CV_64F 3x1，与 cv::solvePnP 同语义的角轴 + 平移）。
    // 返回 Ceres 是否收敛成功。
    bool solve(const std::vector<cv::Point3f>& points_3d,
               const std::vector<cv::Point2f>& points_2d,
               cv::Mat& rvec, cv::Mat& tvec,
               const cv::Mat& rvec_init = cv::Mat(),
               const cv::Mat& tvec_init = cv::Mat());

private:
    void initFrom(const CameraProjection& camera_proj);

    double K_[4];                           // fx, fy, cx, cy
    double D_[5];                           // k1, k2, p1, p2, k3 (OpenCV 畸变模型)
    double max_tan2_;                       // 视场角最大 tan²θ，即 FoV 锥面约束 a: x²+y² ≤ a·z²
    int    width_, height_;                 // 图像分辨率（用于判断关键点是否在画面内）

    // ── 内嵌重投影代价函数（含畸变校正） ──
    struct ReprojectionError
    {
        public:
            ReprojectionError(Eigen::Vector3d point_, Eigen::Vector2d observed_,
                              const double* K, const double* D)
                : point(point_), observed(observed_)
            {
                K_ = K;
                D_ = D;
            }

            template<typename T>
            bool operator()(const T* const camera_r, const T* const camera_t, T* residuals) const
            {
                // 3D点变换到相机坐标系
                T pt1[3];
                pt1[0] = T(point.x());
                pt1[1] = T(point.y());
                pt1[2] = T(point.z());

                T pt2[3];
                ceres::AngleAxisRotatePoint(camera_r, pt1, pt2);
                pt2[0] = pt2[0] + camera_t[0];
                pt2[1] = pt2[1] + camera_t[1];
                pt2[2] = pt2[2] + camera_t[2];

                // 归一化坐标（齐次除法）
                const T xp = pt2[0] / pt2[2];
                const T yp = pt2[1] / pt2[2];

                // ── 应用畸变模型（OpenCV 标准模型） ──
                const T k1 = T(D_[0]);
                const T k2 = T(D_[1]);
                const T p1 = T(D_[2]);
                const T p2 = T(D_[3]);
                const T k3 = T(D_[4]);

                const T r2 = xp * xp + yp * yp;
                const T r4 = r2 * r2;
                const T r6 = r4 * r2;
                const T radial = T(1.0) + k1 * r2 + k2 * r4 + k3 * r6;

                const T xpp = xp * radial + T(2.0) * p1 * xp * yp + p2 * (r2 + T(2.0) * xp * xp);
                const T ypp = yp * radial + p1 * (r2 + T(2.0) * yp * yp) + T(2.0) * p2 * xp * yp;

                // 投影到像素坐标
                const T u_pred = T(K_[0]) * xpp + T(K_[2]);
                const T v_pred = T(K_[1]) * ypp + T(K_[3]);

                const T u = T(observed.x());
                const T v = T(observed.y());

                // 重投影误差
                residuals[0] = u - u_pred;
                residuals[1] = v - v_pred;
                return true;
            }
            static ceres::CostFunction* Create(Eigen::Vector3d points, Eigen::Vector2d observed,
                                               const double* K, const double* D)
            {
                return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
                                new ReprojectionError(points, observed, K, D)));
            }

        private:
            Eigen::Vector3d point;
            Eigen::Vector2d observed;
            const double* K_;
            const double* D_;
    };

    // ── FoV 视场角硬边界约束代价函数 ──
    //     3D 点变换到相机坐标系后，必须满足 x² + y² ≤ max_tan2_ * z²
    //     即该点必须在相机视锥范围内
    struct FoVConstraint
    {
        FoVConstraint(Eigen::Vector3d point_, double max_tan2)
            : point(point_), max_tan2_(max_tan2) {}

        template <typename T>
        bool operator()(const T* const camera_r, const T* const camera_t, T* residual) const
        {
            T pt1[3] = { T(point.x()), T(point.y()), T(point.z()) };
            T pt2[3];
            ceres::AngleAxisRotatePoint(camera_r, pt1, pt2);
            pt2[0] += camera_t[0];
            pt2[1] += camera_t[1];
            pt2[2] += camera_t[2];

            T r2 = pt2[0] * pt2[0] + pt2[1] * pt2[1];
            T bound = T(max_tan2_) * pt2[2] * pt2[2];
            residual[0] = (r2 > bound) ? (r2 - bound) : T(0.0);
            return true;
        }

        static ceres::CostFunction* Create(Eigen::Vector3d point, double max_tan2)
        {
            return new ceres::AutoDiffCostFunction<FoVConstraint, 1, 3, 3>(
                new FoVConstraint(point, max_tan2));
        }

    private:
        Eigen::Vector3d point;
        double max_tan2_;
    };
};

#endif // CERES_POSE_ESTIMATOR_H
