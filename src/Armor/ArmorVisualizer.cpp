// ArmorVisualizer.cpp — 装甲板自瞄可视化实现（全部绘制逻辑从 test/main.cpp 抽取）
#include "Armor/ArmorVisualizer.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "common/TransformTree/CoordinateTransform.h"

void ArmorVisualizer::render(cv::Mat& image,
                               const ArmorVisualizationData& data,
                               const RobotTfTree& tf_tree,
                               const CameraProjection& camera_proj) const {
    // ── 1. 每物体 PnP 角点重投影 + 序号 ──
    for (size_t i = 0; i < data.raw_pose.reprojected_points.size(); ++i) {
        drawReprojectedPoints(image, (int)i, data.raw_pose.reprojected_points[i]);
    }

    // ── 2. OutpostESEKF 滤波结果 ──
    if (data.filtered_pose.valid) {
        // EKF 世界关键点投影（红）
        drawEskfProjectedPoints(image, data.filtered_pose.world_points, tf_tree, camera_proj);

        // 预测目标中心关键点（t+0，绿）
        drawPredictedCenterPoints(image, data.filtered_pose.pred_center_points, tf_tree, camera_proj);

        // 滤波后的位置与欧拉角（文字）
        cv::Mat R64 = data.filtered_pose.R64;
        cv::Mat R32;
        R64.convertTo(R32, CV_32F);
        cv::Vec3f ekf_euler = CoordinateTransform::rotationMatrixToEuler(R32);
        drawEskfPose(image,
                     cv::Vec3f((float)data.filtered_pose.pos[0],
                               (float)data.filtered_pose.pos[1],
                               (float)data.filtered_pose.pos[2]),
                     ekf_euler);
    }

    // ── 3. 瞄准点（品红） ──
    if (data.aim.auto_aim_enable) {
        drawAimPoint(image, data.aim.predicted_point, data.aim.predict_time, tf_tree, camera_proj);
    }

    // ── 4. 通信信息 / 温度 / FPS ──
    // 以下三项改由 main（输出模式窗口覆盖层）统一绘制，保证 Armor / PowerRune
    // 两个流水线的可视化行为一致（串口信息 + 帧数统计 + 热键提醒）：

    // ── 5. 检测结果（最上层） ──
    drawDetectionResults(image, data.detection.objects);

    // ── 6. 每个物体的世界坐标位姿 ──
    for (size_t i = 0; i < data.raw_pose.world_positions.size(); ++i) {
        drawWorldPose(image, (int)i, data.raw_pose.world_positions[i], data.raw_pose.world_eulers[i]);
    }
}

void ArmorVisualizer::drawWorldPose(cv::Mat& img, int objIdx,
                                      const cv::Vec3f& world_pos, const cv::Vec3f& world_euler) {
    int x0 = 608;   // 原 8，右移 600 避开统一覆盖层（串口信息/热键提醒）
    int y0 = img.rows - 80;
    int lineH = 16;
    cv::Scalar color(255, 200, 0);
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.40;
    int thick = 1;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "Obj" << objIdx << " WorldPos(m): ["
        << world_pos[0] << ", " << world_pos[1] << ", " << world_pos[2] << "]";
    cv::putText(img, oss.str(), cv::Point(x0, y0), font, scale, color, thick);

    oss.str("");
    oss << std::fixed << std::setprecision(3);
    oss << "Obj" << objIdx << " WorldEuler(rad): ["
        << world_euler[0] << ", " << world_euler[1] << ", " << world_euler[2] << "]";
    cv::putText(img, oss.str(), cv::Point(x0, y0 + lineH), font, scale, color, thick);
}

void ArmorVisualizer::drawEskfPose(cv::Mat& img, const cv::Vec3f& filtered_pos,
                                     const cv::Vec3f& filtered_euler) {
    int x0 = 608;   // 原 8，右移 600 避开统一覆盖层（串口信息/热键提醒）
    int lineH = 16;
    int y0 = img.rows - 80 - 2 * lineH - 6;  // 位于 drawWorldPose 上方
    cv::Scalar color(0, 255, 0);
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.40;
    int thick = 1;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "EKF WorldPos(m): [" << filtered_pos[0] << ", "
        << filtered_pos[1] << ", " << filtered_pos[2] << "]";
    cv::putText(img, oss.str(), cv::Point(x0, y0), font, scale, color, thick);

    oss.str("");
    oss << std::fixed << std::setprecision(3);
    oss << "EKF WorldEuler(rad): [" << filtered_euler[0] << ", "
        << filtered_euler[1] << ", " << filtered_euler[2] << "]";
    cv::putText(img, oss.str(), cv::Point(x0, y0 + lineH), font, scale, color, thick);
}

void ArmorVisualizer::drawReprojectedPoints(cv::Mat& img, int objIdx,
                                              const std::vector<cv::Point2f>& reprojected_pts) {
    static const cv::Scalar colors[] = {
        cv::Scalar(0, 255, 255),    // 黄
        cv::Scalar(255, 0, 255),    // 品红
        cv::Scalar(255, 255, 0),    // 青
        cv::Scalar(0, 255, 128),    // 绿
    };
    cv::Scalar clr = colors[objIdx % 4];

    for (int j = 0; j < 4 && j < (int)reprojected_pts.size(); ++j) {
        cv::Point pt(reprojected_pts[j].x, reprojected_pts[j].y);
        cv::circle(img, pt, 5, clr, -1);   // 实心圆点
        cv::circle(img, pt, 7, clr, 2);    // 外圈
        cv::putText(img, std::to_string(j), pt + cv::Point(10, -5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, clr, 2);  // 序号
    }
}

void ArmorVisualizer::drawEskfProjectedPoints(cv::Mat& img,
                                                const std::vector<cv::Point3f>& world_points,
                                                const RobotTfTree& tf_tree,
                                                const CameraProjection& camera_proj) const {
    // 世界坐标关键点 -> 相机系 -> 投影
    std::vector<cv::Point3f> ekf_cam;
    ekf_cam.reserve(world_points.size());
    for (const auto& wp : world_points) {
        cv::Vec3f p = tf_tree.transformPoint(RobotTfTree::WORLD, RobotTfTree::CAMERA,
                                             cv::Vec3f(wp.x, wp.y, wp.z));
        ekf_cam.emplace_back(p[0], p[1], p[2]);
    }
    std::vector<cv::Point2f> ekf_projected;
    camera_proj.projectPoints_Cam(ekf_cam, ekf_projected);

    cv::Scalar ekf_color(0, 0, 255);  // 红色，区分 PnP 结果
    for (size_t j = 0; j < ekf_projected.size(); ++j) {
        cv::Point pt(ekf_projected[j].x, ekf_projected[j].y);
        cv::circle(img, pt, 4, ekf_color, -1);
        cv::circle(img, pt, 6, ekf_color, 2);
        cv::putText(img, std::to_string(j), pt + cv::Point(10, -5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, ekf_color, 2);
    }
}

void ArmorVisualizer::drawPredictedCenterPoints(cv::Mat& img,
                                                  const std::vector<cv::Point3f>& pred_world,
                                                  const RobotTfTree& tf_tree,
                                                  const CameraProjection& camera_proj) const {
    // 预测目标中心关键点（capturePosePredictor 的 t+0 快照）投影
    std::vector<cv::Point3f> pred_cam;
    pred_cam.reserve(pred_world.size());
    for (const auto& wp : pred_world) {
        cv::Vec3f p = tf_tree.transformPoint(RobotTfTree::WORLD, RobotTfTree::CAMERA,
                                             cv::Vec3f(wp.x, wp.y, wp.z));
        pred_cam.emplace_back(p[0], p[1], p[2]);
    }
    std::vector<cv::Point2f> pred_projected;
    camera_proj.projectPoints_Cam(pred_cam, pred_projected);

    cv::Scalar pred_color(0, 255, 0);   // 绿色
    for (size_t j = 0; j < pred_projected.size(); ++j) {
        cv::Point pt(pred_projected[j].x, pred_projected[j].y);
        cv::circle(img, pt, 5, pred_color, -1);
        cv::circle(img, pt, 8, pred_color, 2);
        if (j == 0) {
            cv::putText(img, "t+0.0s", pt + cv::Point(12, -8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 2);
        }
    }
}

void ArmorVisualizer::drawAimPoint(cv::Mat& img,
                                     const cv::Vec3f& aim_world, double predict_time,
                                     const RobotTfTree& tf_tree,
                                     const CameraProjection& camera_proj) const {
    cv::Vec3f aim_cam = tf_tree.transformPoint(RobotTfTree::WORLD, RobotTfTree::CAMERA, aim_world);
    std::vector<cv::Point2f> aim_projected;
    camera_proj.projectPoints_Cam({cv::Point3f(aim_cam[0], aim_cam[1], aim_cam[2])}, aim_projected);
    if (aim_projected.empty()) return;

    cv::Point pt(aim_projected[0].x, aim_projected[0].y);
    cv::Scalar aim_color(255, 0, 255);   // 品红
    cv::circle(img, pt, 9, aim_color, -1);
    cv::circle(img, pt, 13, aim_color, 2);
    cv::putText(img, cv::format("AIM t+%.2fs", predict_time),
                pt + cv::Point(15, -10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, aim_color, 2);
}

// ============================================================================
// XY 平面窗口（顶视图；仿照 transistor_rm2027_algorithm_visual_ws 的 RMM 顶视图）
// ============================================================================

void ArmorVisualizer::openXYWindow() {
    cv::namedWindow(XY_WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    xy_window_open_ = true;
}

void ArmorVisualizer::closeXYWindow() {
    if (!xy_window_open_) return;
    try {
        cv::destroyWindow(XY_WINDOW_NAME);
    } catch (const cv::Exception&) {
        // 窗口可能已被用户手动关闭（点 X），忽略
    }
    xy_window_open_ = false;
}

bool ArmorVisualizer::xyWindowOpen() const {
    return xy_window_open_;
}

void ArmorVisualizer::renderXY(const ArmorVisualizationData& data) {
    if (!xy_window_open_) return;

    constexpr int kSize = 900;              // 窗口尺寸（像素）
    constexpr int kMargin = 70;             // 绘制边距（像素）
    constexpr float kDefaultRange = 5.0f;   // 无点/范围过小时默认半边长（米）
    const cv::Scalar kBg(248, 248, 248);          // 白底（仿照 RMM 顶视图）
    const cv::Scalar kGrid(226, 226, 226);        // 网格线
    const cv::Scalar kAxis(150, 150, 150);        // 世界原点十字
    const cv::Scalar kChassisColor(0, 160, 0);    // 绿：自身底盘
    const cv::Scalar kCenterColor(0, 0, 255);     // 红：车体中心
    const cv::Scalar kArmorColor(255, 0, 0);      // 蓝：装甲板位置（t+0 预测）
    const cv::Scalar kAimColor(255, 0, 255);      // 品红：瞄准目标
    const cv::Scalar kTextColor(60, 60, 60);

    xy_buf_.create(kSize, kSize, CV_8UC3);
    xy_buf_.setTo(kBg);

    // ── 收集待绘制点（world 系 x/y，米）──
    struct Pt2 { float x, y; };
    std::vector<Pt2> pts;
    if (data.filtered_pose.valid) {
        pts.push_back({(float)data.filtered_pose.pos[0], (float)data.filtered_pose.pos[1]});
        for (const auto& p : data.filtered_pose.pred_center_points)
            pts.push_back({p.x, p.y});
    }
    if (data.xy.aim_valid)
        pts.push_back({data.xy.aim_point[0], data.xy.aim_point[1]});
    pts.push_back({data.xy.chassis_position[0], data.xy.chassis_position[1]});

    // ── 自动缩放：包围盒 + 边距（无有效点或范围过小时用默认半边长）──
    float min_x = -kDefaultRange, max_x = kDefaultRange;
    float min_y = -kDefaultRange, max_y = kDefaultRange;
    bool have_pts = false;
    for (const auto& p : pts) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) continue;
        if (!have_pts) {
            min_x = max_x = p.x;
            min_y = max_y = p.y;
            have_pts = true;
        } else {
            min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
        }
    }
    if (have_pts) {
        // 以包围盒中心为中心，x/y 跨度较大者 + 默认范围下界为边长（保证近处
        // 目标也留出视野），避免车体中心/瞄准目标贴在窗口边缘
        const float span = std::max({max_x - min_x, max_y - min_y,
                                     2.0f * kDefaultRange});
        const float cx = 0.5f * (min_x + max_x);
        const float cy = 0.5f * (min_y + max_y);
        min_x = cx - span * 0.5f; max_x = cx + span * 0.5f;
        min_y = cy - span * 0.5f; max_y = cy + span * 0.5f;
    }
    const double scale = (double)(kSize - 2 * kMargin) / (double)(max_x - min_x);
    // world (x, y) → 窗口像素（x 向右，y 向上）
    auto w2p = [&](float x, float y) -> cv::Point {
        return cv::Point((int)std::lround(kMargin + (x - min_x) * scale),
                         (int)std::lround(kSize - kMargin - (y - min_y) * scale));
    };

    // ── 网格 + 世界原点十字 ──
    // 自适应网格步长（米）：使网格间距约 >= 40px
    float grid_m = 0.1f;
    {
        static const float kSteps[] = {0.1f, 0.2f, 0.5f, 1.0f, 2.0f, 5.0f,
                                       10.0f, 20.0f, 50.0f, 100.0f};
        for (float s : kSteps) {
            if (s * (float)scale >= 40.0f) { grid_m = s; break; }
        }
    }
    for (float gx = std::floor(min_x / grid_m) * grid_m;
         gx <= max_x + 1e-6f; gx += grid_m) {
        const int px = w2p(gx, 0).x;
        cv::line(xy_buf_, cv::Point(px, kMargin), cv::Point(px, kSize - kMargin),
                 kGrid, 1);
    }
    for (float gy = std::floor(min_y / grid_m) * grid_m;
         gy <= max_y + 1e-6f; gy += grid_m) {
        const int py = w2p(0, gy).y;
        cv::line(xy_buf_, cv::Point(kMargin, py), cv::Point(kSize - kMargin, py),
                 kGrid, 1);
    }
    if (min_x <= 0.0f && 0.0f <= max_x) {   // 世界原点 x=0 竖线
        const int px = w2p(0, 0).x;
        cv::line(xy_buf_, cv::Point(px, kMargin), cv::Point(px, kSize - kMargin),
                 kAxis, 1);
    }
    if (min_y <= 0.0f && 0.0f <= max_y) {   // 世界原点 y=0 横线
        const int py = w2p(0, 0).y;
        cv::line(xy_buf_, cv::Point(kMargin, py), cv::Point(kSize - kMargin, py),
                 kAxis, 1);
    }
    cv::putText(xy_buf_, "x+", cv::Point(kSize - 62, kMargin + 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, kTextColor, 1, cv::LINE_AA);
    cv::putText(xy_buf_, "y+", cv::Point(kMargin + 4, kMargin + 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, kTextColor, 1, cv::LINE_AA);

    // ── chassis → 瞄准目标连线（先画，避免压住标记）──
    if (data.xy.aim_valid) {
        const cv::Point a = w2p(data.xy.chassis_position[0], data.xy.chassis_position[1]);
        const cv::Point b = w2p(data.xy.aim_point[0], data.xy.aim_point[1]);
        cv::line(xy_buf_, a, b, cv::Scalar(0, 0, 220), 2, cv::LINE_AA);
    }

    // ── 自身 chassis 位置（绿方块）──
    {
        const cv::Point p = w2p(data.xy.chassis_position[0], data.xy.chassis_position[1]);
        cv::drawMarker(xy_buf_, p, kChassisColor, cv::MARKER_SQUARE, 24, 2, cv::LINE_AA);
        cv::putText(xy_buf_, "chassis", p + cv::Point(10, -12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, kChassisColor, 1, cv::LINE_AA);
    }

    // ── 车体中心（红实心圆 + 外圈）──
    if (data.filtered_pose.valid) {
        const cv::Point p = w2p((float)data.filtered_pose.pos[0],
                                (float)data.filtered_pose.pos[1]);
        cv::circle(xy_buf_, p, 7, kCenterColor, -1, cv::LINE_AA);
        cv::circle(xy_buf_, p, 12, kCenterColor, 2, cv::LINE_AA);
        cv::putText(xy_buf_, "center", p + cv::Point(13, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, kCenterColor, 1, cv::LINE_AA);
    }

    // ── 装甲板位置（t=0 预测，蓝实心圆 + 序号；CLASS_EKF 目标即四块装甲板）──
    if (data.filtered_pose.valid) {
        for (size_t i = 0; i < data.filtered_pose.pred_center_points.size(); ++i) {
            const auto& ap = data.filtered_pose.pred_center_points[i];
            const cv::Point p = w2p(ap.x, ap.y);
            cv::circle(xy_buf_, p, 6, kArmorColor, -1, cv::LINE_AA);
            cv::circle(xy_buf_, p, 10, kArmorColor, 2, cv::LINE_AA);
            cv::putText(xy_buf_, cv::format("A%d", (int)i), p + cv::Point(9, -9),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, kArmorColor, 1, cv::LINE_AA);
        }
    }

    // ── 瞄准目标位置（品红叉 + 圆）──
    if (data.xy.aim_valid) {
        const cv::Point p = w2p(data.xy.aim_point[0], data.xy.aim_point[1]);
        cv::drawMarker(xy_buf_, p, kAimColor, cv::MARKER_CROSS, 28, 2, cv::LINE_AA);
        cv::circle(xy_buf_, p, 12, kAimColor, 2, cv::LINE_AA);
        cv::putText(xy_buf_, "aim", p + cv::Point(13, -13),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, kAimColor, 1, cv::LINE_AA);
    }

    // ── 标题 / 比例尺 ──
    cv::putText(xy_buf_, "Armor XY Plane (armor positions: predictor t=0)",
                cv::Point(14, 26), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                kTextColor, 2, cv::LINE_AA);
    cv::putText(xy_buf_, cv::format("grid: %.2f m", grid_m),
                cv::Point(14, kSize - 18), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                kTextColor, 1, cv::LINE_AA);

    cv::imshow(XY_WINDOW_NAME, xy_buf_);
}

// ============================================================================
// drawDetectionResults（复制自 test/main.cpp，原 RobotDetectionModel 绘制逻辑）
// ============================================================================

void ArmorVisualizer::drawDetectionResults(cv::Mat& image,
                                             const std::vector<ArmorDetect::Object>& objects) {
    static const cv::Scalar COLOR_RED   = cv::Scalar(0, 0, 255);
    static const cv::Scalar COLOR_BLUE  = cv::Scalar(255, 0, 0);
    static const cv::Scalar COLOR_GREEN = cv::Scalar(0, 255, 0);
    static const cv::Scalar COLOR_WHITE = cv::Scalar(255, 255, 255);
    static const cv::Scalar COLOR_GRAY  = cv::Scalar(200, 200, 200);

    int img_h = image.rows, img_w = image.cols;

    for (const auto& obj : objects) {
        if (obj.rect.width <= 0 || obj.rect.height <= 0 ||
            obj.rect.x >= img_w || obj.rect.y >= img_h ||
            obj.rect.x + obj.rect.width <= 0 || obj.rect.y + obj.rect.height <= 0)
            continue;

        cv::Scalar color = (obj.color == 1) ? COLOR_RED :
                                (obj.color == 0) ? COLOR_BLUE : COLOR_GREEN;
        cv::Scalar color_dim = (obj.color == 1) ? cv::Scalar(0, 0, 180) :
                                    (obj.color == 0) ? cv::Scalar(180, 0, 0) : cv::Scalar(0, 180, 0);

        // 检测框 + 角标
        cv::rectangle(image, obj.rect, color, 2);
        int cl = std::max((int)(std::min(obj.rect.width, obj.rect.height) / 4), 10);
        cv::line(image, cv::Point(obj.rect.x, obj.rect.y + cl), cv::Point(obj.rect.x, obj.rect.y), color, 2);
        cv::line(image, cv::Point(obj.rect.x, obj.rect.y), cv::Point(obj.rect.x + cl, obj.rect.y), color, 2);
        cv::line(image, cv::Point(obj.rect.br().x - cl, obj.rect.y), cv::Point(obj.rect.br().x, obj.rect.y), color, 2);
        cv::line(image, cv::Point(obj.rect.br().x, obj.rect.y), cv::Point(obj.rect.br().x, obj.rect.y + cl), color, 2);
        cv::line(image, cv::Point(obj.rect.x, obj.rect.br().y - cl), cv::Point(obj.rect.x, obj.rect.br().y), color, 2);
        cv::line(image, cv::Point(obj.rect.x, obj.rect.br().y), cv::Point(obj.rect.x + cl, obj.rect.br().y), color, 2);
        cv::line(image, cv::Point(obj.rect.br().x - cl, obj.rect.br().y), cv::Point(obj.rect.br().x, obj.rect.br().y), color, 2);
        cv::line(image, cv::Point(obj.rect.br().x, obj.rect.br().y), cv::Point(obj.rect.br().x, obj.rect.br().y - cl), color, 2);

        // 关键点连线 左上->左下->右下->右上
        cv::Point2f pts[4] = {
            cv::Point2f(obj.landmarks[0], obj.landmarks[1]),
            cv::Point2f(obj.landmarks[2], obj.landmarks[3]),
            cv::Point2f(obj.landmarks[4], obj.landmarks[5]),
            cv::Point2f(obj.landmarks[6], obj.landmarks[7])
        };
        for (int i = 0; i < 4; ++i) {
            cv::line(image, pts[i], pts[(i+1)%4], color_dim, 1);
            cv::circle(image, pts[i], 3, color, -1);
            cv::putText(image, std::to_string(i), cv::Point(pts[i].x + 5, pts[i].y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1);
        }

        // 标签
        static const char* CLS[] = {"Sentry","1","2","3","4","5","Armor","Base","BigBase"};
        const char* cname = (obj.label >= 0 && obj.label < 9) ? CLS[obj.label] : "?";
        std::string lbl = cv::format("%s %.1f%%", cname, obj.prob * 100.f);
        int baseline;
        cv::Size lsz = cv::getTextSize(lbl, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        cv::Rect lbg(obj.rect.x, obj.rect.y - lsz.height - 8, lsz.width + 12, lsz.height + 6);
        if (lbg.y < 0) lbg.y = obj.rect.y + obj.rect.height + 8;

        cv::Rect lbgc = lbg & cv::Rect(0, 0, img_w, img_h);
        if (lbgc.width > 0 && lbgc.height > 0) {
            cv::Mat roi = image(lbgc);
            cv::Mat ovl = roi.clone();
            cv::rectangle(ovl, cv::Rect(0, 0, lbgc.width, lbgc.height), color_dim, -1);
            cv::addWeighted(ovl, 0.7, roi, 0.3, 0, roi);
        }
        cv::rectangle(image, lbg, color, 1);
        cv::putText(image, lbl, cv::Point(lbg.x + 6, lbg.y + lsz.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2);
        cv::putText(image, lbl, cv::Point(lbg.x + 6, lbg.y + lsz.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 1);

        // 颜色 + 编号
        std::string tag = (obj.color == 0) ? "RED" : (obj.color == 1) ? "BLUE" : "?";
        std::string ids = cv::format("[%s] #%d", tag.c_str(), obj.label);
        cv::putText(image, ids, cv::Point(obj.rect.x + 5, obj.rect.y + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1);
        cv::putText(image, ids, cv::Point(obj.rect.x + 5, obj.rect.y + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1);

        // 尺寸
        int iy = obj.rect.y + obj.rect.height + 15;
        if (lbg.y > obj.rect.y) iy = lbg.y + lbg.height + 5;
        cv::putText(image, cv::format("%.0f x %.0f", obj.rect.width, obj.rect.height),
                    cv::Point(obj.rect.x, iy), cv::FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1);
        cv::putText(image, cv::format("R:%.2f L:%.0f W:%.0f", obj.ratio, obj.length, obj.width),
                    cv::Point(obj.rect.x, iy + 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1);

        // 进度条
        int bw = std::min((int)obj.rect.width, 80), bh = 4;
        int bx = obj.rect.x, by = obj.rect.y + obj.rect.height + 35;
        if (by < 0) by = obj.rect.y - 15;
        by = std::max(0, std::min(by, img_h - bh));
        bx = std::max(0, std::min(bx, img_w - bw));
        if (by + bh <= img_h && bx + bw <= img_w) {
            cv::rectangle(image, cv::Rect(bx, by, bw, bh), cv::Scalar(100, 100, 100), -1);
            int fw = (int)(bw * obj.prob);
            if (fw > 0) cv::rectangle(image, cv::Rect(bx, by, fw, bh), color, -1);
        }
        cv::putText(image, cv::format("%.1f%%", obj.prob * 100.f),
                    cv::Point(bx + bw + 4, by + bh), cv::FONT_HERSHEY_SIMPLEX, 0.35, color, 1);
    }
}
