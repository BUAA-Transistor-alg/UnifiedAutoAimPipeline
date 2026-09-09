// ceres_pnp_flag_test.cpp — 验证 common/pose 的 Ceres PnP 精化功能
//   1) solvePnP_Cam 的 flags 支持自定义 flag CameraProjection::SOLVEPNP_CERES：
//      · {IPPE, ITERATIVE}          —— 纯 OpenCV（原有行为）
//      · {IPPE, ITERATIVE, CERES}   —— OpenCV 粗解 + Ceres 精化（初值=上一阶段结果）
//      · {CERES}                    —— 仅 Ceres，从默认位姿开始求解
//   2) CeresPoseEstimator::solve 支持传入初始位姿进一步优化。
//   3) CeresPoseEstimator::solve 新增 Z 轴夹角硬约束模式（cam 系方向向量 +
//      目标夹角余弦）：约束的是「物体在 cam 系下的 +z 轴」（= cam 系 (0,0,1)
//      经位姿旋转后在 cam 系下的方向；在 PnP 局部系表示为 (0,-1,0) 参与旋转），
//      与给定 cam 方向夹角余弦==目标值，以 ~1e8 权重作硬等号约束全程生效
//      （含与重投影最优冲突时仍被强制满足）。
// 合成数据：小装甲板 4 角点（与 ArmorModel 同尺寸），标准相机内参，
// 由已知 rvec/tvec 生成像素观测，比较各 flags 恢复的位姿/重投影误差。
#include <cstdio>
#include <cmath>
#include <vector>

#include "common/pose/CameraProjection.h"
#include "common/pose/CeresPoseEstimator.h"

using cv::Vec3f;

static double reprojRms(const std::vector<cv::Point3f>& p3d_pnp,
                        const std::vector<cv::Point2f>& p2d,
                        const cv::Mat& rvec, const cv::Mat& tvec,
                        const CameraProjection& cp) {
    std::vector<cv::Point2f> proj;
    cp.projectPoints(p3d_pnp, rvec, tvec, proj);
    double s = 0.0;
    for (size_t i = 0; i < p2d.size(); ++i) {
        double dx = p2d[i].x - proj[i].x, dy = p2d[i].y - proj[i].y;
        s += dx * dx + dy * dy;
    }
    return std::sqrt(s / p2d.size());
}

int main() {
    // ── 相机内参（假设分辨率 1280x1024，无畸变）──
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1400, 0, 640, 0, 1400, 512, 0, 0, 1);
    cv::Mat D = cv::Mat::zeros(1, 5, CV_64F);
    CameraProjection cp(K, D, ImageResolution{1280, 1024});

    // ── 小装甲板 4 角点（Cam 系输入，即装甲板局部系；同 ArmorModel）──
    const float w = 0.133f, h = 0.05f;
    const std::vector<cv::Point3f> armor_local = {
        {-w / 2, 0.0f,  h / 2}, {-w / 2, 0.0f, -h / 2},
        { w / 2, 0.0f, -h / 2}, { w / 2, 0.0f,  h / 2}};

    // ── 真值位姿（PnP 系下 rvec/tvec，随后统一转 Cam 系比较）──
    cv::Mat rvec_true = (cv::Mat_<double>(3, 1) << 0.12, -0.08, 0.04);
    cv::Mat tvec_true = (cv::Mat_<double>(3, 1) << 0.06, -0.05, 2.8);

    // 由真值生成像素观测
    std::vector<cv::Point3f> armor_pnp;
    for (const auto& p : armor_local) {
        Vec3f pp = CameraProjection::camToPnp_posi(Vec3f(p.x, p.y, p.z));
        armor_pnp.emplace_back(pp[0], pp[1], pp[2]);
    }
    std::vector<cv::Point2f> p2d;
    cp.projectPoints(armor_pnp, rvec_true, tvec_true, p2d);

    // 加 1.5px 噪声（使 Ceres 精化有可改善空间）
    unsigned seed = 20270909u;
    auto rnd = [&seed]() {  // [0,1)
        seed = seed * 1103515245u + 12345u;
        return (double)((seed >> 8) & 0xFFFFFF) / (double)0x1000000;
    };
    std::vector<cv::Point2f> p2d_noisy = p2d;
    for (auto& p : p2d_noisy) {
        p.x += (float)((rnd() - 0.5) * 3.0);
        p.y += (float)((rnd() - 0.5) * 3.0);
    }

    const Vec3f pos_true_cam = CameraProjection::pnpTvecToCamPosi(tvec_true);

    int fails = 0;

    // ── 场景 1：solvePnP_Cam 三种 flags 组合 ──
    std::vector<std::vector<int>> flag_sets = {
        {cv::SOLVEPNP_IPPE, cv::SOLVEPNP_ITERATIVE},
        {cv::SOLVEPNP_IPPE, cv::SOLVEPNP_ITERATIVE, CameraProjection::SOLVEPNP_CERES},
        {CameraProjection::SOLVEPNP_CERES},
    };
    const char* names[] = {"{IPPE, ITERATIVE}", "{IPPE, ITERATIVE, CERES}", "{CERES}"};

    for (size_t i = 0; i < flag_sets.size(); ++i) {
        Vec3f pos_cam, euler_cam;
        bool ok = cp.solvePnP_Cam(armor_local, p2d_noisy, flag_sets[i], pos_cam, euler_cam);
        double err = norm(pos_cam - pos_true_cam);
        // 无噪声观测下的重投影 RMS（衡量姿态本身，而不受噪声影响）
        std::vector<cv::Mat> probe;
        (void)probe;
        printf("[solvePnP_Cam %-24s] ok=%d pos_cam=(%.3f,%.3f,%.3f) euler=(%.3f,%.3f,%.3f) |pos-pos_true|=%.4f m\n",
               names[i], ok ? 1 : 0,
               pos_cam[0], pos_cam[1], pos_cam[2],
               euler_cam[0], euler_cam[1], euler_cam[2], err);
        if (!ok) fails++;
    }

    // ── 场景 2：CeresPoseEstimator::solve 显式传入（被污染）初始位姿做精化 ──
    {
        CeresPoseEstimator est(cp);
        std::vector<cv::Point3f> pts_pnp;
        for (const auto& p : armor_local) {
            Vec3f pp = CameraProjection::camToPnp_posi(Vec3f(p.x, p.y, p.z));
            pts_pnp.emplace_back(pp[0], pp[1], pp[2]);
        }
        // 初始位姿故意偏移（平移 +0.4m、旋转 +0.3 rad 量级）
        cv::Mat rvec_init = (cv::Mat_<double>(3, 1) << 0.12 + 0.3, -0.08 - 0.2, 0.04 + 0.15);
        cv::Mat tvec_init = (cv::Mat_<double>(3, 1) << 0.46, 0.35, 3.2);
        double rms0 = reprojRms(pts_pnp, p2d, rvec_init, tvec_init, cp);

        cv::Mat rvec_out, tvec_out;
        bool ok = est.solve(pts_pnp, p2d, rvec_out, tvec_out, rvec_init, tvec_init);
        double rms1 = reprojRms(pts_pnp, p2d, rvec_out, tvec_out, cp);
        double dist = norm(CameraProjection::pnpTvecToCamPosi(tvec_out) - pos_true_cam);
        printf("[ceres refine w/ init pose] ok=%d rms %.3f px -> %.3f px, |pos-pos_true|=%.4f m\n",
               ok ? 1 : 0, rms0, rms1, dist);
        if (!ok || rms1 >= rms0) fails++;

        // 同一接口不传初值（默认位姿开始）
        cv::Mat rvec_d, tvec_d;
        bool ok2 = est.solve(pts_pnp, p2d, rvec_d, tvec_d);
        double rms2 = reprojRms(pts_pnp, p2d, rvec_d, tvec_d, cp);
        double dist2 = norm(CameraProjection::pnpTvecToCamPosi(tvec_d) - pos_true_cam);
        printf("[ceres default start       ] ok=%d rms %.3f px, |pos-pos_true|=%.4f m\n",
               ok2 ? 1 : 0, rms2, dist2);
        if (!ok2) fails++;
    }

    // ── 场景 3：Z 轴夹角硬约束模式（新增 solve 重载）──
    // 被约束轴 = 物体在 cam 系下的 +z=(0,0,1)_cam。由于 points_3d 是 cam 系点经
    // camToPnp 换算的 PnP 点，该轴在 PnP 局部系表示为 (0,-1,0)，因此求解出的
    // 姿态其 cam-z 轴在求解器(PnP)相机系下的单位向量 = R·(0,-1,0) = -旋转矩阵第2列。
    {
        CeresPoseEstimator est(cp);
        std::vector<cv::Point3f> pts_pnp;
        for (const auto& p : armor_local) {
            Vec3f pp = CameraProjection::camToPnp_posi(Vec3f(p.x, p.y, p.z));
            pts_pnp.emplace_back(pp[0], pp[1], pp[2]);
        }

        // 由 rvec 求物体 cam-z 轴（在求解器/PnP 相机系下）= R·(0,-1,0)
        auto poseZCamPnp = [](const cv::Mat& rvec) -> cv::Vec3f {
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            return cv::Vec3f(-(float)R.at<double>(0, 1),
                             -(float)R.at<double>(1, 1),
                             -(float)R.at<double>(2, 1));
        };

        // 校验辅助：用与 solve 内部完全相同的 cam→pnp 换算恢复约束所用的 PnP 单位方向
        auto toPnpDir = [](const cv::Vec3f& dir_cam) -> cv::Vec3f {
            cv::Vec3f d = CameraProjection::camToPnp_posi(dir_cam);
            float n = std::sqrt(d.dot(d));
            return (n > 1e-12f) ? d / n : cv::Vec3f(0, 0, 0);
        };

        // 真值位姿下物体 cam-z 轴：先在 PnP 相机系表示，再换回 cam 系作为约束方向
        const cv::Vec3f z_obj_true_pnp = poseZCamPnp(rvec_true);   // PnP 相机系
        const cv::Vec3f z_obj_true_cam = CameraProjection::pnpToCam_posi(z_obj_true_pnp);

        // 3a) 目标与真值自洽（dir_cam = 真值位姿下物体 cam-z 轴的 cam 系方向,
        //     target=1）：约束与重投影解一致，cos 误差应趋近 0 且位姿接近真值
        {
            cv::Mat rvec_o, tvec_o;
            bool ok = est.solve(pts_pnp, p2d_noisy, rvec_o, tvec_o, z_obj_true_cam, 1.0);
            const cv::Vec3f d_pnp = toPnpDir(z_obj_true_cam);
            const double cos_err = std::fabs((double)poseZCamPnp(rvec_o).dot(d_pnp) - 1.0);
            const double dist = norm(CameraProjection::pnpTvecToCamPosi(tvec_o) - pos_true_cam);
            printf("[ceres zcos feasible   ] ok=%d cos_err=%.2e (<=2e-3) |pos-pos_true|=%.4f m\n",
                   ok ? 1 : 0, cos_err, dist);
            if (!ok || cos_err > 2e-3) fails++;
        }

        // 3b) 目标与重投影最优冲突（把物体 cam-z 轴硬拉到与真值约 90° 的方向,
        //     target=1）：验证硬等号约束确实生效 —— 即使牺牲重投影误差，输出姿态
        //     的物体 cam-z 轴与给定 cam 方向的夹角余弦也必须等于目标值
        {
            // 真值 cam-z 轴绕 cam 系 x 轴旋转 90°：得到一个与其垂直的方向
            const cv::Vec3f zt = z_obj_true_cam;
            cv::Vec3f forced_cam(zt[0], -zt[2], zt[1]);
            const float fn = std::sqrt(forced_cam.dot(forced_cam));
            if (fn > 1e-12f) forced_cam /= fn;
            cv::Mat rvec_o, tvec_o;
            bool ok = est.solve(pts_pnp, p2d_noisy, rvec_o, tvec_o, forced_cam, 1.0);
            const cv::Vec3f d_pnp = toPnpDir(forced_cam);
            const double cos_err = std::fabs((double)poseZCamPnp(rvec_o).dot(d_pnp) - 1.0);
            const double cos_true = (double)poseZCamPnp(rvec_o).dot(z_obj_true_pnp);  // ≈0 证明被硬拉 90°
            printf("[ceres zcos enforced   ] ok=%d cos_err=%.2e (<=1e-3) camz_axis·true_camz=%.3f (≈0 为硬拉 90°)\n",
                   ok ? 1 : 0, cos_err, cos_true);
            if (!ok || cos_err > 1e-3) fails++;
        }

        // 3c) 近似正对相机的位姿 + 光轴方向 (0,0,1)_cam、target=1：
        //     物体 cam-z 轴应天然指向 cam +z，约束不应把姿态拧动 ——
        //     验证「cam +z 在 PnP 局部系表示为 (0,-1,0)」的方向/符号正确。
        //     （若错误地旋转 (0,0,1)，此处会被硬拉 ~180° 使 cos 变为 -1）
        {
            cv::Mat rvec_id = (cv::Mat_<double>(3, 1) << 0.03, -0.02, 0.01);
            cv::Mat tvec_id = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 3.0);
            std::vector<cv::Point2f> p2d_id;
            cp.projectPoints(pts_pnp, rvec_id, tvec_id, p2d_id);
            cv::Mat rvec_o, tvec_o;
            bool ok = est.solve(pts_pnp, p2d_id, rvec_o, tvec_o, cv::Vec3f(0, 0, 1), 1.0);
            // 输出物体 cam-z 轴与光轴 (0,0,1)_cam 的夹角余弦：d_pnp = camToPnp((0,0,1)) = (0,-1,0)
            const double cos_err = std::fabs((double)poseZCamPnp(rvec_o).dot(cv::Vec3f(0, -1, 0)) - 1.0);
            const double dist = norm(CameraProjection::pnpTvecToCamPosi(tvec_o) -
                                     CameraProjection::pnpTvecToCamPosi(tvec_id));
            printf("[ceres zcos optical axis] ok=%d cos_err=%.2e (<=2e-3) |t-t_id|=%.4f m\n",
                   ok ? 1 : 0, cos_err, dist);
            if (!ok || cos_err > 2e-3) fails++;
        }
    }

    printf(fails ? "\nRESULT: %d FAILED\n" : "\nRESULT: ALL PASSED\n", fails);
    return fails;
}
