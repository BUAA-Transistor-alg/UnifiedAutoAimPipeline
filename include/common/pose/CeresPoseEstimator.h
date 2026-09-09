// CeresPoseEstimator.h — 基于 Ceres 的手动 PnP 位姿优化
// 替代/精化 OpenCV solvePnP：使用 Ceres 的非线性最小二乘优化重投影误差。
// 与 CameraProjection 同置于 common/pose/，由 solvePnP_Cam 的自定义 flag
// （CameraProjection::SOLVEPNP_CERES）调用。
//
// 支持三种用法：
//   1) 传入初始位姿（rvec_init/tvec_init）→ 以该位姿为初值做进一步优化（精化），
//      适合接在 cv::solvePnP 等粗解之后使用；
//   2) 不传初始位姿 → 从默认位姿（rvec=0、tvec 前向 z=1）开始优化（从头求解）；
//   3) 额外传入一组 AxisCosConstraintParam 约束（物体轴-方向夹角硬约束模式）→
//      在重投影 + FoV 硬边界之外，再以「硬等号约束」要求：每条约束中被指定的
//      物体本体轴（cam 系表示，如本体 z 轴 = (0,0,1)）经位姿旋转后在 cam 系下
//      的单位方向，与其 cam 系目标方向向量的夹角余弦 == 该条 target_cos。
//      约束与 FoV 硬边界一样以极大权重近似硬边界，传入非空即全程生效；
//      传入空 vector 等价于普通求解（详见下方 solve 重载注释）。
#ifndef CERES_POSE_ESTIMATOR_H
#define CERES_POSE_ESTIMATOR_H

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>

#include "common/pose/CameraProjection.h"

// ── 单条「物体本体轴-方向」夹角硬约束参数 ──
// axis_body_cam：要约束的物体本体坐标系下的向量，坐标按 cam 系给出（如本体 z 轴
//   = (0,0,1)）。求解时按与 points_3d 相同的约定先换算到 PnP 系再参与旋转；
//   cam 系 +x/+y/+z 分别对应位姿为零时物体本体 +x/+y/+z。
// dir_cam：cam 系下的目标方向向量。
// target_cos：要求「axis_body_cam 经位姿旋转后在 cam 系的单位方向 ⊙ dir_cam 的
//   单位方向 == target_cos」。
// 两个方向向量长度任意，内部都会归一化；传入零向量/余弦越界会断言。
struct AxisCosConstraintParam {
    cv::Vec3f axis_body_cam;   // 物体本体坐标系的被约束向量（cam 系表示）
    cv::Vec3f dir_cam;         // cam 系目标方向向量
    double    target_cos = 0.0; // 目标夹角余弦 ∈ [-1, 1]
};

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

    // ── PnP 求解（物体轴-方向夹角硬约束模式）──
    // 在前述重投影 + FoV 硬边界求解基础上，对 constraints 中的每条约束额外以
    // 「硬等号约束」要求：该物体本体轴经位姿旋转后在 cam 系下的单位方向，与
    // dir_cam（cam 系）的夹角余弦 == target_cos。
    //   每条约束：残差 = (R·axis_pnp)·d_pnp - target_cos（axis_pnp/d_pnp 为该轴与
    //   方向向量按 camToPnp 换算到 PnP 系并归一化后的单位向量，等价于 cam 系
    //   下的夹角余弦）；与 FoV 硬边界同样以 ~1e8 权重加入，全程作为硬等号约束
    //   生效（不依赖 3D 点是否落在画面内）。多条约束互不干扰，逐条各加一个残差块。
    //   传入空 vector 等价于不开启该模式（同普通 solve）。
    // 其余参数/输出语义与上方 solve 相同。返回 Ceres 是否收敛成功。
    bool solve(const std::vector<cv::Point3f>& points_3d,
               const std::vector<cv::Point2f>& points_2d,
               cv::Mat& rvec, cv::Mat& tvec,
               const std::vector<AxisCosConstraintParam>& constraints);

    bool solve(const std::vector<cv::Point3f>& points_3d,
               const std::vector<cv::Point2f>& points_2d,
               cv::Mat& rvec, cv::Mat& tvec,
               const cv::Mat& rvec_init, const cv::Mat& tvec_init,
               const std::vector<AxisCosConstraintParam>& constraints);

private:
    void initFrom(const CameraProjection& camera_proj);

    // ── 上述两个 solve 接口的共同实现 ──
    // constraints 非空时，对每条约束各加入一条物体轴-方向夹角硬等号残差块；
    // 轴/方向向量（cam 系表示）在实现内部按 camToPnp 换算到 points_3d 所在 PnP
    // 系并归一化后再参与点乘。为空 vector 表示未开启该模式（同普通 solve）。
    bool solveImpl(const std::vector<cv::Point3f>& points_3d,
                   const std::vector<cv::Point2f>& points_2d,
                   cv::Mat& rvec, cv::Mat& tvec,
                   const cv::Mat& rvec_init, const cv::Mat& tvec_init,
                   const std::vector<AxisCosConstraintParam>& constraints);

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

    // ── 物体轴-方向夹角硬等号约束代价函数（每条 AxisCosConstraintParam 一条）──
    //     axis_：被约束的物体本体轴在 points_3d（PnP 系）局部系下的单位向量
    //     （由 cam 系 axis_body_cam 经 camToPnp 换算并归一化得到，与求解器内
    //     旋转矩阵所作用的点同系；本体 z 轴 (0,0,1)_cam 即对应 (0,-1,0)_pnp）。
    //     axis_ 经旋转矩阵作用后（= 该物体轴在求解器相机系下的单位向量），
    //     与 dir_（PnP 系单位目标方向，由 cam 系 dir_cam 换算并归一化得到）点乘，
    //     结果即 cam 系下的夹角余弦；约束残差 = 余弦 - target_cos_，目标为零。
    //     只依赖旋转 camera_r（与平移无关），加入时配 ~1e8 的权重近似硬边界。
    struct AxisCosConstraint
    {
        AxisCosConstraint(Eigen::Vector3d axis_unit, Eigen::Vector3d dir_unit,
                          double target_cos)
            : axis_(axis_unit), dir_(dir_unit), target_cos_(target_cos) {}

        template <typename T>
        bool operator()(const T* const camera_r, T* residual) const
        {
            // 物体本体轴（PnP 局部系表示）旋转到求解器相机系
            const T axis_local[3] = { T(axis_.x()), T(axis_.y()), T(axis_.z()) };
            T axis_rot[3];
            ceres::AngleAxisRotatePoint(camera_r, axis_local, axis_rot);

            // dir_ 已归一化，点乘即夹角余弦；残差 = 余弦 - 目标值
            residual[0] = axis_rot[0] * T(dir_.x()) +
                          axis_rot[1] * T(dir_.y()) +
                          axis_rot[2] * T(dir_.z()) - T(target_cos_);
            return true;
        }

        static ceres::CostFunction* Create(Eigen::Vector3d axis_unit, Eigen::Vector3d dir_unit,
                                           double target_cos)
        {
            return new ceres::AutoDiffCostFunction<AxisCosConstraint, 1, 3>(
                new AxisCosConstraint(axis_unit, dir_unit, target_cos));
        }

    private:
        Eigen::Vector3d axis_;
        Eigen::Vector3d dir_;
        double target_cos_;
    };
};

#endif // CERES_POSE_ESTIMATOR_H
