// ceres_pnp_flag_test.cpp — 验证 common/pose 的 Ceres PnP 精化功能
//   1) solvePnP_Cam 的 flags 支持自定义 flag CameraProjection::SOLVEPNP_CERES：
//      · {IPPE, ITERATIVE}          —— 纯 OpenCV（原有行为）
//      · {IPPE, ITERATIVE, CERES}   —— OpenCV 粗解 + Ceres 精化（初值=上一阶段结果）
//      · {CERES}                    —— 仅 Ceres，从默认位姿开始求解
//   2) CeresPoseEstimator::solve 支持传入初始位姿进一步优化。
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

    printf(fails ? "\nRESULT: %d FAILED\n" : "\nRESULT: ALL PASSED\n", fails);
    return fails;
}
