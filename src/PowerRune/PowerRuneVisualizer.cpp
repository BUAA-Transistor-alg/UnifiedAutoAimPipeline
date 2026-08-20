#include "PowerRune/PowerRuneVisualizer.h"
#include <iostream>
#include <limits>

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
                                  const CameraProjection& camera_proj) const {
    // 1. 绘制检测框
    // drawDetections(image, data.detection.detections);
    // return;

    // 2. 绘制世界坐标和欧拉角文字
    string label = format("%.3f   %.3f   %.3f", data.raw_pose.world_pos[0], data.raw_pose.world_pos[1], data.raw_pose.world_pos[2]);
    putText(image, label, Point(620, 80),
        FONT_HERSHEY_SIMPLEX, 1, COLOR_GRAY, 2);
    label = format("%.3f   %.3f   %.3f", data.raw_pose.world_euler[0], data.raw_pose.world_euler[1], data.raw_pose.world_euler[2]);
    putText(image, label, Point(620, 120),
        FONT_HERSHEY_SIMPLEX, 1, COLOR_GRAY, 2);

    // 3. 绘制角速度和跳变信息
    label = format("%.3f", data.filtered_pose.filtered_omega);
    putText(image, label, Point(620, 160),
        FONT_HERSHEY_SIMPLEX, 1, COLOR_WHITE, 2);
    label = format("%d", data.filtered_pose.jump_a);
    putText(image, label, Point(800, 160),
        FONT_HERSHEY_SIMPLEX, 1, data.filtered_pose.flip ? Scalar(255, 0, 255) : Scalar(0, 255, 0), 2);

    // 3.5. 绘制滤波后的世界坐标和欧拉角
    if (!data.filtered_pose.filtered_R.empty()) {
        cv::Vec3f filtered_euler = CoordinateTransform::rotationMatrixToEuler(data.filtered_pose.filtered_R);
        label = format("%.3f  %.3f  %.3f",
                       data.filtered_pose.filtered_pos[0],
                       data.filtered_pose.filtered_pos[1],
                       data.filtered_pose.filtered_pos[2]);
        putText(image, label, Point(620, 200),
            FONT_HERSHEY_SIMPLEX, 1, COLOR_WHITE, 2);
        label = format("%.3f  %.3f  %.3f",
                       filtered_euler[0], filtered_euler[1], filtered_euler[2]);
        putText(image, label, Point(620, 240),
            FONT_HERSHEY_SIMPLEX, 1, COLOR_WHITE, 2);
    }

    // 3.6. 绘制 Roll 预测器拟合参数和方向（在 filtered_euler 下方，绿色，拟合无效时不显示）
    if (data.roll_predictor.fit_valid) {
        if (data.roll_predictor.fit_method == "big") {
            label = format("%.3f  %.3f  %.3f  bias:%.3f  dir:%+d",
                           data.roll_predictor.big_params.a,
                           data.roll_predictor.big_params.omega,
                           data.roll_predictor.big_params.o_t,
                           data.roll_predictor.correction_bias,
                           data.roll_predictor.direction);
        } else {
            label = format("%.3f  bias:%.3f  dir:%+d",
                           data.roll_predictor.small_params.o_t,
                           data.roll_predictor.correction_bias,
                           data.roll_predictor.direction);
        }
        putText(image, label, Point(620, 280),
            FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    }

    // 3.8. 绘制 target rotation counts（红色，空格分隔）
    if (!data.filtered_pose.target_rotation_counts.empty()) {
        string target_label;
        for (size_t i = 0; i < data.filtered_pose.target_rotation_counts.size(); ++i) {
            if (i > 0) target_label += " ";
            target_label += std::to_string(data.filtered_pose.target_rotation_counts[i]);
        }
        putText(image, target_label, Point(620, 660),
            FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    }

    // 3.9. 绘制当前使用的拟合方法（在 target rotation counts 下方，青色）
    if (data.roll_predictor.fit_valid && !data.roll_predictor.fit_method.empty()) {
        string method_label = "method: " + data.roll_predictor.fit_method;
        putText(image, method_label, Point(620, 700),
            FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 2);
    }

    // 3.7. 绘制 Roll 预测器拟合曲线图（仅在拟合有效时绘制）
    if (data.roll_predictor.fit_valid &&
        !data.roll_predictor.fitted_curve.empty() &&
        !data.roll_predictor.raw_points.empty()) {

        // 绘图区域边界
        constexpr int PLOT_X1 = 620, PLOT_Y1 = 320;
        constexpr int PLOT_X2 = 920, PLOT_Y2 = 620;
        const cv::Point plot_tl(PLOT_X1, PLOT_Y1);
        const cv::Point plot_br(PLOT_X2, PLOT_Y2);

        // 绘制边框
        rectangle(image, plot_tl, plot_br, Scalar(255, 255, 255), 1);

        // 计算两组数据合并后的最值
        float t_min = std::numeric_limits<float>::max();
        float t_max = std::numeric_limits<float>::lowest();
        float r_min = std::numeric_limits<float>::max();
        float r_max = std::numeric_limits<float>::lowest();

        auto updateBounds = [&](const std::vector<std::pair<float, float>>& pts) {
            for (const auto& p : pts) {
                if (p.first  < t_min) t_min = p.first;
                if (p.first  > t_max) t_max = p.first;
                if (p.second < r_min) r_min = p.second;
                if (p.second > r_max) r_max = p.second;
            }
        };

        updateBounds(data.roll_predictor.fitted_curve);
        updateBounds(data.roll_predictor.raw_points);

        // 防止除零（数据为常量时扩展一定范围）
        constexpr float EPS = 1e-5f;
        if (t_max - t_min < EPS) { t_max += 0.5f; t_min -= 0.5f; }
        if (r_max - r_min < EPS) { r_max += 0.5f; r_min -= 0.5f; }

        // 数据坐标到像素坐标的线性映射
        // t: [t_min, t_max] → 像素 x [plot_tl.x, plot_br.x]
        // r: [r_min, r_max] → 像素 y [plot_br.y, plot_tl.y]（r 越大越靠上）
        float scale_t = static_cast<float>(PLOT_X2 - PLOT_X1) / (t_max - t_min);
        float scale_r = static_cast<float>(PLOT_Y2 - PLOT_Y1) / (r_max - r_min);

        auto dataToPixel = [&](float t, float r) -> cv::Point {
            int px = static_cast<int>(PLOT_X1 + (t - t_min) * scale_t + 0.5f);
            int py = static_cast<int>(PLOT_Y2 - (r - r_min) * scale_r + 0.5f);
            return cv::Point(px, py);
        };

        // 绘制原始观测点（青色小点）
        for (const auto& p : data.roll_predictor.raw_points) {
            cv::Point pt = dataToPixel(p.first, p.second);
            circle(image, pt, 2, Scalar(255, 255, 0), -1);
        }

        // 绘制拟合曲线（绿色折线段）
        for (size_t i = 1; i < data.roll_predictor.fitted_curve.size(); ++i) {
            cv::Point pt1 = dataToPixel(
                data.roll_predictor.fitted_curve[i - 1].first,
                data.roll_predictor.fitted_curve[i - 1].second);
            cv::Point pt2 = dataToPixel(
                data.roll_predictor.fitted_curve[i].first,
                data.roll_predictor.fitted_curve[i].second);
            line(image, pt1, pt2, Scalar(0, 255, 0), 2);
        }
    }

    // 4. 绘制滤波后的位姿：白色五边形 + CMY 三轴(粗线)
    if (!data.filtered_pose.filtered_R.empty()) {
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

    // // 5. 绘制滤波器预测位姿：白色五边形(不绘制) + CMY 三轴(细线)
    // if (!data.filtered_pose.filter_predicted_R.empty()) {
    //     drawPoseAxes(
    //         image, data.filtered_pose.filtered_pos, data.filtered_pose.filter_predicted_R,
    //         COLOR_WHITE, 0,                             // 白色五边形, 线宽0(不绘制)
    //         {Scalar(0, 255, 255),
    //          Scalar(255, 0, 255),
    //          Scalar(255, 255, 0)}, 2,                    // CMY 三轴, 线宽2
    //         COLOR_WHITE, 0,                               // 白色中心点, 半径0(不绘制)
    //         coordinate_transform, pose_solver
    //     );
    // }

    // 6. 绘制滤波前的原始位姿：灰色五边形 + RGB 三轴(细线)
    if (!data.raw_pose.world_rot_mat.empty()) {
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

    if (data.roll_predictor.fit_valid) {
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
    for (const auto& pt : data.filtered_target_points) {
        cv::Point2f img_pt = worldToImage(pt, tf_tree, camera_proj);
        if (img_pt.x >= 0) {
            cv::circle(image, img_pt, 10, Scalar(0, 255, 255), -1);
        }
    }
    // 预测位姿下的目标位置（天蓝色圆点，半径10）
    for (const auto& pt : data.predictor_target_points) {
        cv::Point2f img_pt = worldToImage(pt, tf_tree, camera_proj);
        if (img_pt.x >= 0) {
            cv::circle(image, img_pt, 10, Scalar(255, 255, 0), -1);
        }
    }
}

// ==================== drawDetections ====================
void PowerRuneVisualizer::drawDetections(Mat& image,
                                          const vector<PoseDetection>& detections) {
    int obj_index = 0;
    for (const auto& det : detections) {
        Scalar obj_color = OBJECT_PALETTE[obj_index % OBJECT_PALETTE_SIZE];
        ++obj_index;
        const char* class_name = (det.class_id >= 0 && det.class_id <= 7)
            ? CLASS_NAMES[det.class_id]
            : ("Unknown:" + to_string(det.class_id)).c_str();

        rectangle(image, det.rect, obj_color, 2);

        string label = format("%s %.1f%%", class_name, det.confidence * 100.0f);
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