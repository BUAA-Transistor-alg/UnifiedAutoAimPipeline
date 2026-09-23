#include "PowerRune/PowerRuneVisualizer.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <cmath>

using namespace cv;
using namespace std;

// ==================== 静态颜色常量定义 ====================
const Scalar PowerRuneVisualizer::COLOR_WHITE = Scalar(255, 255, 255);
const Scalar PowerRuneVisualizer::COLOR_BLACK = Scalar(0, 0, 0);
const Scalar PowerRuneVisualizer::COLOR_GRAY  = Scalar(128, 128, 128);

// 6 种高饱和颜色 (BGR)
const Scalar PowerRuneVisualizer::OBJECT_PALETTE[] = {
    Scalar(0, 0, 255),      // 红
    Scalar(0, 255, 0),      // 绿
    Scalar(255, 0, 0),      // 蓝
    Scalar(0, 255, 255),    // 黄
    Scalar(255, 0, 255),    // 品红
    Scalar(255, 255, 0),    // 青
};

// ==================== 五边形顶点(局部坐标) ====================
vector<cv::Point3f> PowerRuneVisualizer::generatePentagonPoints() {
    constexpr float PI = 3.14159265358979323846f;
    constexpr float radius = 0.7f;
    constexpr float angle_step = 2.0f * PI / 5.0f;
    vector<cv::Point3f> pts;
    for (int i = 0; i < 5; ++i) {
        float angle = i * angle_step;
        pts.emplace_back(radius * sin(angle), 0.0f, radius * cos(angle));
    }
    return pts;
}

const vector<cv::Point3f> PowerRuneVisualizer::pentagon_local_ =
    PowerRuneVisualizer::generatePentagonPoints();

// ==================== worldToImage ====================
cv::Point2f PowerRuneVisualizer::worldToImage(
    const cv::Vec3f& p_world,
    const RobotTfTree& tf_tree,
    const CameraProjection& camera_proj) const
{
    // world → cam（经 tf 树）→ image（相机投影，内部自动 PnP 系转换）
    cv::Vec3f p_cam = tf_tree.transformPoint(RobotTfTree::WORLD, RobotTfTree::CAMERA, p_world);
    vector<cv::Point2f> proj;
    camera_proj.projectPoints_Cam({
        cv::Point3f(p_cam[0], p_cam[1], p_cam[2])
    }, proj);
    return proj.empty() ? cv::Point2f(-1, -1) : proj[0];
}

// ==================== render ====================
void PowerRuneVisualizer::render(Mat& image,
                                  const PowerRuneVisualizationData& data,
                                  const RobotTfTree& tf_tree,
                                  const CameraProjection& camera_proj,
                                  const PowerRuneVisualizationOptions& options) const {
    if (options.detections) drawDetections(image, data.detection.detections, options.details);

    // 4. 绘制滤波后的位姿：白色五边形 + CMY 三轴(粗线)
    if (options.filtered_pose && !data.filtered_pose.filtered_R.empty()) {
        drawPoseAxes(
            image, data.filtered_pose.filtered_pos, data.filtered_pose.filtered_R,
            COLOR_WHITE, 3,                             // 白色五边形, 线宽3
            {Scalar(0, 255, 255),
             Scalar(255, 0, 255),
             Scalar(255, 255, 0)}, 5,                    // CMY 三轴, 线宽5
            COLOR_WHITE, 7,                               // 白色中心点, 半径7
            tf_tree, camera_proj
        );
    }

    // 6. 绘制滤波前的原始位姿：灰色五边形 + RGB 三轴(细线)
    if (options.raw_pose && !data.raw_pose.world_rot_mat.empty()) {
        drawPoseAxes(
            image, data.raw_pose.world_pos, data.raw_pose.world_rot_mat,
            COLOR_GRAY, 2,                               // 灰色五边形, 线宽2
            {Scalar(255, 0, 0),
                Scalar(0, 255, 0),
                Scalar(0, 0, 255)}, 2,                      // RGB 三轴, 线宽2
            COLOR_GRAY, 5,                                // 灰色中心点, 半径5
            tf_tree, camera_proj
        );
    }

    if (options.predicted_pose && data.roll_predictor.fit_valid &&
        !data.roll_predictor.predictor_prediction.second.empty()) {
        drawPoseAxes(
            image, data.roll_predictor.predictor_prediction.first, data.roll_predictor.predictor_prediction.second,
            Scalar(0, 255, 0), 2,                           // 绿色五边形, 线宽2
            {Scalar(255, 127, 127),
                Scalar(127, 255, 127),
                Scalar(127, 127, 255)}, 2,                      // 低饱和度高亮度 RGB 三轴, 线宽2
            COLOR_GRAY, 0,                                // 不画中心点
            tf_tree, camera_proj
        );
    }

    // 6. 绘制目标位置可视化点
    // 滤波位姿下的目标位置（黄色圆点，半径10）
    if (options.filtered_pose) {
        for (const auto& pt : data.filtered_target_points) {
            cv::Point2f img_pt = worldToImage(pt, tf_tree, camera_proj);
            if (img_pt.x >= 0) {
                cv::circle(image, img_pt, 10, Scalar(0, 255, 255), -1);
            }
        }
    }
    // 预测位姿下的目标位置（天蓝色圆点，半径10）：被屏蔽的靶点（本帧不存在）
    // 不绘制——预测列表恒含全部 5 个靶点，屏蔽索引与列表下标一一对应
    if (options.predicted_pose) {
        for (size_t i = 0; i < data.predictor_target_points.size(); ++i) {
            if (std::find(data.predictor_masked_indices.begin(),
                          data.predictor_masked_indices.end(), static_cast<int>(i)) !=
                data.predictor_masked_indices.end()) {
                continue;
            }
            cv::Point2f img_pt = worldToImage(data.predictor_target_points[i], tf_tree, camera_proj);
            if (img_pt.x >= 0) {
                cv::circle(image, img_pt, 10, Scalar(255, 255, 0), -1);
            }
        }
    }

}

// ==================== Separate fit / status window ====================
bool PowerRuneVisualizer::fitWindowExists() const {
    // GTK does not support WND_PROP_VISIBLE; use the same probe as the controls panel.
    try { return cv::getWindowProperty(fit_window_name_, cv::WND_PROP_AUTOSIZE) >= 0; }
    catch (const cv::Exception&) { return false; }
}

bool PowerRuneVisualizer::syncFitWindow(bool active) {
    if (!active) {
        closeFitWindow();
        return true;
    }
    if (fit_window_open_) {
        if (fitWindowExists()) return true;
        fit_window_open_ = false;
        return false;  // Let the controls panel reflect a manual window close.
    }
    cv::namedWindow(fit_window_name_, cv::WINDOW_AUTOSIZE);
    fit_window_open_ = true;
    // Show useful empty-state text even before the first PowerRune frame arrives.
    cv::imshow(fit_window_name_, makeFitPanel(PowerRuneVisualizationData{}));
    return true;
}

void PowerRuneVisualizer::closeFitWindow() {
    if (fit_window_open_ && fitWindowExists()) cv::destroyWindow(fit_window_name_);
    fit_window_open_ = false;
}

void PowerRuneVisualizer::renderFitWindow(const PowerRuneVisualizationData& data) const {
    if (fit_window_open_ && fitWindowExists())
        cv::imshow(fit_window_name_, makeFitPanel(data));
}

cv::Mat PowerRuneVisualizer::makeFitPanel(const PowerRuneVisualizationData& data) const {
    cv::Mat panel(740, 820, CV_8UC3, cv::Scalar(25, 25, 25));
    const cv::Scalar white(225, 225, 225), gray(155, 155, 155);
    const cv::Scalar green(0, 230, 100), cyan(255, 255, 0);
    const auto text = [&](int y, const std::string& value, cv::Scalar color) {
        cv::putText(panel, value, {20, y}, cv::FONT_HERSHEY_SIMPLEX,
                    .52, color, 1, cv::LINE_AA);
    };
    text(28, "PowerRune | Fit curve and pose status", white);
    const bool raw_valid = !data.raw_pose.world_rot_mat.empty();
    text(60, raw_valid ? cv::format("Raw position [m]: %.3f  %.3f  %.3f",
         data.raw_pose.world_pos[0], data.raw_pose.world_pos[1], data.raw_pose.world_pos[2])
         : "Raw position [m]: N/A", gray);
    text(84, raw_valid ? cv::format("Raw Euler [rad]: %.3f  %.3f  %.3f",
         data.raw_pose.world_euler[0], data.raw_pose.world_euler[1], data.raw_pose.world_euler[2])
         : "Raw Euler [rad]: N/A", gray);

    const bool filtered_valid = !data.filtered_pose.filtered_R.empty();
    text(116, filtered_valid ? cv::format("Filtered position [m]: %.3f  %.3f  %.3f",
         data.filtered_pose.filtered_pos[0], data.filtered_pose.filtered_pos[1],
         data.filtered_pose.filtered_pos[2]) : "Filtered position [m]: N/A", white);
    if (filtered_valid) {
        const auto euler = CoordinateTransform::rotationMatrixToEuler(data.filtered_pose.filtered_R);
        text(140, cv::format("Filtered Euler [rad]: %.3f  %.3f  %.3f",
             euler[0], euler[1], euler[2]), white);
        text(164, cv::format("Omega [rad/s]: %.3f   jump_a: %d   flip: %s",
             data.filtered_pose.filtered_omega, data.filtered_pose.jump_a,
             data.filtered_pose.flip ? "true" : "false"), white);
    } else {
        text(140, "Filtered Euler [rad]: N/A", white);
        text(164, "Omega / jump_a / flip: N/A", white);
    }
    std::string targets = "Observed target indices:";
    for (int index : data.filtered_pose.target_rotation_counts)
        targets += " " + std::to_string(index);
    if (data.filtered_pose.target_rotation_counts.empty()) targets += " none";
    text(188, targets, white);

    const auto& fit = data.roll_predictor;
    if (fit.fit_valid) {
        text(222, cv::format("Fit: valid   method: %s   direction: %+d",
             fit.fit_method.c_str(), fit.direction), green);
        if (fit.fit_method == "big") {
            text(246, cv::format("a: %.3f   omega: %.3f   o_t: %.3f",
                 fit.big_params.a, fit.big_params.omega, fit.big_params.o_t), green);
        } else {
            text(246, cv::format("o_t: %.3f", fit.small_params.o_t), green);
        }
        text(270, cv::format("Correction bias [rad]: %.3f", fit.correction_bias), green);
    } else {
        text(222, "Fit: invalid / waiting for observations", gray);
        text(246, "Fit parameters: N/A", gray);
        text(270, "Correction bias: N/A", gray);
    }
    text(304, "Cyan: observations     Green: fitted curve", white);
    text(328, "Roll [rad] vs sample time [s] (automatic scale)", gray);
    constexpr int x1 = 75, y1 = 360, x2 = 790, y2 = 680;
    cv::rectangle(panel, {x1, y1}, {x2, y2}, gray, 1);
    const auto finite = [](const auto& point) {
        return std::isfinite(point.first) && std::isfinite(point.second);
    };
    float tmin = std::numeric_limits<float>::max(), tmax = -tmin;
    float rmin = tmin, rmax = -tmin;
    bool have_points = false;
    const auto bounds = [&](const auto& points) {
        for (const auto& point : points) {
            if (!finite(point)) continue;
            have_points = true;
            tmin = std::min(tmin, point.first); tmax = std::max(tmax, point.first);
            rmin = std::min(rmin, point.second); rmax = std::max(rmax, point.second);
        }
    };
    bounds(fit.raw_points);
    if (fit.fit_valid) bounds(fit.fitted_curve);
    if (!have_points) {
        text(400, "No curve samples available", gray);
        return panel;
    }
    if (tmax - tmin < 1e-5f) { tmax += .5f; tmin -= .5f; }
    if (rmax - rmin < 1e-5f) { rmax += .5f; rmin -= .5f; }
    const auto pixel = [&](const auto& point) {
        return cv::Point(x1 + cvRound((point.first-tmin)/(tmax-tmin)*(x2-x1)),
                         y2 - cvRound((point.second-rmin)/(rmax-rmin)*(y2-y1)));
    };
    for (const auto& point : fit.raw_points)
        if (finite(point)) cv::circle(panel, pixel(point), 2, cyan, -1);
    if (fit.fit_valid) {
        for (size_t i = 1; i < fit.fitted_curve.size(); ++i)
            if (finite(fit.fitted_curve[i-1]) && finite(fit.fitted_curve[i]))
                cv::line(panel, pixel(fit.fitted_curve[i-1]), pixel(fit.fitted_curve[i]), green, 2);
    }
    cv::putText(panel, cv::format("%.2f", rmax), {5, y1+5}, 0, .42, gray, 1, cv::LINE_AA);
    cv::putText(panel, cv::format("%.2f", rmin), {5, y2}, 0, .42, gray, 1, cv::LINE_AA);
    cv::putText(panel, cv::format("%.2f", tmin), {x1, y2+25}, 0, .45, gray, 1, cv::LINE_AA);
    cv::putText(panel, cv::format("%.2f", tmax), {x2-65, y2+25}, 0, .45, gray, 1, cv::LINE_AA);
    return panel;
}

// ==================== drawDetections ====================
void PowerRuneVisualizer::drawDetections(Mat& image,
                                          const vector<PoseDetection>& detections, bool details) {
    int obj_index = 0;
    for (const auto& det : detections) {
        Scalar obj_color = OBJECT_PALETTE[obj_index % OBJECT_PALETTE_SIZE];
        ++obj_index;
        const string class_name = (det.class_id >= 0 && det.class_id <= 7)
            ? CLASS_NAMES[det.class_id]
            : ("Unknown:" + to_string(det.class_id));

        rectangle(image, det.rect, obj_color, 2);

        if (!details) continue;

        string label = format("%s %.1f%%", class_name.c_str(), det.confidence * 100.0f);
        int baseline;
        Size label_sz = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        Rect label_bg(det.rect.x, det.rect.y - label_sz.height - 5,
                      label_sz.width + 10, label_sz.height + 5);
        if (label_bg.y < 0)
            label_bg.y = det.rect.y + det.rect.height + 5;
        rectangle(image, label_bg, obj_color, FILLED);
        putText(image, label, Point(det.rect.x + 5, label_bg.y + label_sz.height - 2),
                FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1);

        for (const auto& kpt : det.keypoints) {
            if (kpt.z < 0.5f) continue;
            circle(image, Point2f(kpt.x, kpt.y), 4, obj_color, -1);
            circle(image, Point2f(kpt.x, kpt.y), 2, COLOR_BLACK, 1);
        }
    }
}

// ==================== drawPoseAxes ====================
void PowerRuneVisualizer::drawPoseAxes(
    Mat& image,
    const cv::Vec3f& world_pos,
    const cv::Mat& world_rot_mat,
    const cv::Scalar& pentagon_color,
    int pentagon_thickness,
    const array<cv::Scalar, 3>& axis_colors,
    int axis_thickness,
    const cv::Scalar& center_color,
    int center_radius,
    const RobotTfTree& tf_tree,
    const CameraProjection& camera_proj)
{
    // 将世界坐标点投影到图像平面的辅助函数
    auto worldToImage = [&](const cv::Vec3f& p_world) -> cv::Point2f {
        cv::Vec3f p_cam = tf_tree.transformPoint(RobotTfTree::WORLD, RobotTfTree::CAMERA, p_world);
        vector<cv::Point2f> proj;
        camera_proj.projectPoints_Cam({
            cv::Point3f(p_cam[0], p_cam[1], p_cam[2])
        }, proj);
        return proj.empty() ? cv::Point2f(-1, -1) : proj[0];
    };

    // 绘制五边形（世界坐标）
    if (pentagon_thickness) {
        vector<cv::Point2f> pentagon_img;
        for (const auto& pt_local : pentagon_local_) {
            cv::Mat world_vec = world_rot_mat * cv::Mat(cv::Vec3f(pt_local.x, pt_local.y, pt_local.z));
            cv::Vec3f world_pt(
                world_pos[0] + world_vec.at<float>(0, 0),
                world_pos[1] + world_vec.at<float>(1, 0),
                world_pos[2] + world_vec.at<float>(2, 0)
            );
            cv::Point2f img_pt = worldToImage(world_pt);
            pentagon_img.push_back(img_pt);
        }
        for (size_t j = 0; j < pentagon_img.size(); ++j) {
            size_t next = (j + 1) % pentagon_img.size();
            cv::line(image, pentagon_img[j], pentagon_img[next], pentagon_color, pentagon_thickness);
        }
    }

    // 绘制中心点（世界坐标原点 = world_pos）
    if (center_radius) {
        cv::Point2f center_img = worldToImage(world_pos);
        if (center_img.x >= 0) {
            cv::circle(image, center_img, center_radius, center_color, -1);
        }
    }

    // 绘制三轴
    static const array<cv::Vec3f, 3> axis_vectors = {
        cv::Vec3f(1.0f, 0.0f, 0.0f),
        cv::Vec3f(0.0f, 1.0f, 0.0f),
        cv::Vec3f(0.0f, 0.0f, 1.0f)
    };

    if (axis_thickness)
    for (int i = 0; i < 3; ++i) {
        // 起点: world_pos
        cv::Point2f img_start = worldToImage(world_pos);

        // 终点: world_pos + R * axis
        cv::Mat world_vec = world_rot_mat * cv::Mat(axis_vectors[i]);
        cv::Vec3f world_end(
            world_pos[0] + world_vec.at<float>(0, 0),
            world_pos[1] + world_vec.at<float>(1, 0),
            world_pos[2] + world_vec.at<float>(2, 0)
        );
        cv::Point2f img_end = worldToImage(world_end);

        if (img_start.x >= 0 && img_end.x >= 0) {
            cv::line(image, img_start, img_end, axis_colors[i], axis_thickness);
        }
    }
}