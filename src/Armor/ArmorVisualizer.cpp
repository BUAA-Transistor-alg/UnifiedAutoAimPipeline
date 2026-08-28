// ArmorVisualizer.cpp — 装甲板自瞄可视化实现（全部绘制逻辑从 test/main.cpp 抽取）
#include "Armor/ArmorVisualizer.h"

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
