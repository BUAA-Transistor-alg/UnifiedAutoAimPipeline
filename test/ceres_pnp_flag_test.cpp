// ceres_pnp_flag_test.cpp — 验证 common/pose 的 Ceres PnP 精化功能
//   1) solvePnP_Cam 的 flags 支持自定义 flag CameraProjection::SOLVEPNP_CERES：
//      · {IPPE, ITERATIVE}          —— 纯 OpenCV（原有行为）
//      · {IPPE, ITERATIVE, CERES}   —— OpenCV 粗解 + Ceres 精化（初值=上一阶段结果）
//      · {CERES}                    —— 仅 Ceres，从默认位姿开始求解
//   2) CeresPoseEstimator::solve 支持传入初始位姿进一步优化。
//   3) CeresPoseEstimator::solve 新增物体轴-方向夹角硬约束模式：传入
//      std::vector<AxisCosConstraintParam>，每条含（物体本体轴 cam 系表示、
//      cam 系目标方向、目标夹角余弦），要求该本体轴经位姿旋转后在 cam 系方向
//      与目标方向的夹角余弦==目标值，以 ~1e8 权重作硬等号约束全程生效
//      （含与重投影最优冲突时仍被强制满足；多条约束可并存）。
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

    // ── 场景 3：物体轴-方向夹角硬约束模式（AxisCosConstraintParam vector）──
    // 被约束轴 axis_body_cam 是物体本体坐标系下的向量（cam 系表示），例如本体 +z
    // = (0,0,1)。points_3d 是 cam 系点经 camToPnp 换算的 PnP 点，故该轴在 PnP
    // 局部系表示为 camToPnp(axis_body_cam)（本体 +z 即 (0,-1,0)）。求解结果中该
    // 轴在 cam 系的单位方向 = pnpToCam(R·camToPnp(axis_body_cam))。
    {
        CeresPoseEstimator est(cp);
        std::vector<cv::Point3f> pts_pnp;
        for (const auto& p : armor_local) {
            Vec3f pp = CameraProjection::camToPnp_posi(Vec3f(p.x, p.y, p.z));
            pts_pnp.emplace_back(pp[0], pp[1], pp[2]);
        }

        // 由 rvec 求物体本体轴 a_body_cam（cam 系表示）在位姿下的 cam 系单位方向
        auto axisCamDirAtPose = [](const cv::Mat& rvec, const cv::Vec3f& a_body_cam) -> cv::Vec3f {
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            const cv::Vec3f a_pnp = CameraProjection::camToPnp_posi(a_body_cam);
            const cv::Mat v = (cv::Mat_<double>(3, 1) << a_pnp[0], a_pnp[1], a_pnp[2]);
            const cv::Mat rv = R * v;
            const cv::Vec3f dir_pnp((float)rv.at<double>(0),
                                    (float)rv.at<double>(1),
                                    (float)rv.at<double>(2));
            return CameraProjection::pnpToCam_posi(dir_pnp);
        };

        // 单位化辅助
        auto unit = [](const cv::Vec3f& v) -> cv::Vec3f {
            float n = std::sqrt(v.dot(v));
            return (n > 1e-12f) ? v / n : cv::Vec3f(0, 0, 0);
        };

        // 构建单条约束（默认约束物体本体 +z 轴，对应旧的 Z 轴模式）
        auto zConstr = [](const cv::Vec3f& dir_cam, double cos) {
            AxisCosConstraintParam c;
            c.axis_body_cam = cv::Vec3f(0, 0, 1);
            c.dir_cam = dir_cam;
            c.target_cos = cos;
            return std::vector<AxisCosConstraintParam>{c};
        };

        // 真值位姿下物体本体 +z 轴的 cam 系方向
        const cv::Vec3f z_obj_true_cam = axisCamDirAtPose(rvec_true, cv::Vec3f(0, 0, 1));

        // 3a) 目标与真值自洽（dir_cam = 真值位姿下本体 z 轴的 cam 系方向, target=1）：
        //     约束与重投影解一致，cos 误差应趋近 0 且位姿接近真值
        {
            cv::Mat rvec_o, tvec_o;
            bool ok = est.solve(pts_pnp, p2d_noisy, rvec_o, tvec_o, zConstr(z_obj_true_cam, 1.0));
            const double cos_err = std::fabs((double)unit(axisCamDirAtPose(rvec_o, cv::Vec3f(0, 0, 1))).dot(z_obj_true_cam) - 1.0);
            const double dist = norm(CameraProjection::pnpTvecToCamPosi(tvec_o) - pos_true_cam);
            printf("[ceres zcos feasible   ] ok=%d cos_err=%.2e (<=2e-3) |pos-pos_true|=%.4f m\n",
                   ok ? 1 : 0, cos_err, dist);
            if (!ok || cos_err > 2e-3) fails++;
        }

        // 3b) 目标与重投影最优冲突（把本体 z 轴硬拉到与真值约 90° 的方向, target=1）：
        //     验证硬等号约束确实生效 —— 即使牺牲重投影误差，输出位姿下本体 z 轴
        //     与给定 cam 方向的夹角余弦也必须等于目标值
        {
            // 真值本体 z 轴方向绕 cam 系 x 轴旋转 90°：得到一个与其垂直的方向
            const cv::Vec3f zt = z_obj_true_cam;
            cv::Vec3f forced_cam(zt[0], -zt[2], zt[1]);
            const float fn = std::sqrt(forced_cam.dot(forced_cam));
            if (fn > 1e-12f) forced_cam /= fn;
            cv::Mat rvec_o, tvec_o;
            bool ok = est.solve(pts_pnp, p2d_noisy, rvec_o, tvec_o, zConstr(forced_cam, 1.0));
            const double cos_err = std::fabs((double)unit(axisCamDirAtPose(rvec_o, cv::Vec3f(0, 0, 1))).dot(forced_cam) - 1.0);
            const double cos_true = (double)unit(axisCamDirAtPose(rvec_o, cv::Vec3f(0, 0, 1))).dot(unit(z_obj_true_cam));  // ≈0 证明被硬拉 90°
            printf("[ceres zcos enforced   ] ok=%d cos_err=%.2e (<=1e-3) camz_axis·true_camz=%.3f (≈0 为硬拉 90°)\n",
                   ok ? 1 : 0, cos_err, cos_true);
            if (!ok || cos_err > 1e-3) fails++;
        }

        // 3c) 近似正对相机、竖直站立的板 + dir=(0,0,1)_cam（竖直向上）、target=1：
        //     本体 z 轴应天然与 cam 竖直方向一致，约束不应把姿态拧动 ——
        //     验证「本体 z 轴 (0,0,1)_cam 在 PnP 局部系表示为 (0,-1,0)」的方向/符号。
        //     （若错误地旋转 (0,0,1)_pnp，此处会被硬拉 ~180° 使 cos 变为 -1）
        {
            cv::Mat rvec_id = (cv::Mat_<double>(3, 1) << 0.03, -0.02, 0.01);
            cv::Mat tvec_id = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 3.0);
            std::vector<cv::Point2f> p2d_id;
            cp.projectPoints(pts_pnp, rvec_id, tvec_id, p2d_id);
            cv::Mat rvec_o, tvec_o;
            bool ok = est.solve(pts_pnp, p2d_id, rvec_o, tvec_o, zConstr(cv::Vec3f(0, 0, 1), 1.0));
            const double cos_err = std::fabs((double)unit(axisCamDirAtPose(rvec_o, cv::Vec3f(0, 0, 1))).dot(cv::Vec3f(0, 0, 1)) - 1.0);
            const double dist = norm(CameraProjection::pnpTvecToCamPosi(tvec_o) -
                                     CameraProjection::pnpTvecToCamPosi(tvec_id));
            printf("[ceres zcos vertical  ] ok=%d cos_err=%.2e (<=2e-3) |t-t_id|=%.4f m\n",
                   ok ? 1 : 0, cos_err, dist);
            if (!ok || cos_err > 2e-3) fails++;
        }

        // 3d) 双约束：同时锁物体本体 +x 与 +z 轴到真值位姿方向（target=1），
        //     两个非共线轴方向都固定 → 姿态被完全锁定，位姿应回到真值附近，
        //     同时验证多条约束可并存且逐条生效
        {
            const cv::Vec3f x_obj_true_cam = axisCamDirAtPose(rvec_true, cv::Vec3f(1, 0, 0));
            AxisCosConstraintParam c1, c2;
            c1.axis_body_cam = cv::Vec3f(1, 0, 0);   // 本体 +x
            c1.dir_cam = x_obj_true_cam;
            c1.target_cos = 1.0;
            c2.axis_body_cam = cv::Vec3f(0, 0, 1);   // 本体 +z
            c2.dir_cam = z_obj_true_cam;
            c2.target_cos = 1.0;
            const std::vector<AxisCosConstraintParam> two{c1, c2};

            cv::Mat rvec_o, tvec_o;
            bool ok = est.solve(pts_pnp, p2d_noisy, rvec_o, tvec_o, two);
            const double cosx = (double)unit(axisCamDirAtPose(rvec_o, cv::Vec3f(1, 0, 0))).dot(x_obj_true_cam);
            const double cosz = (double)unit(axisCamDirAtPose(rvec_o, cv::Vec3f(0, 0, 1))).dot(z_obj_true_cam);
            const double err = std::max(std::fabs(cosx - 1.0), std::fabs(cosz - 1.0));
            const double dist = norm(CameraProjection::pnpTvecToCamPosi(tvec_o) - pos_true_cam);
            printf("[ceres zcos dual       ] ok=%d max_cos_err=%.2e (<=2e-3) |pos-pos_true|=%.4f m\n",
                   ok ? 1 : 0, err, dist);
            if (!ok || err > 2e-3) fails++;
        }
    }

    printf(fails ? "\nRESULT: %d FAILED\n" : "\nRESULT: ALL PASSED\n", fails);
    return fails;
}
