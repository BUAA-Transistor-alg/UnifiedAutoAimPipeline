// VisualizeOutput.cpp — 可视化输出模式实现
#include "common/Output/VisualizeOutput.h"
#include "common/RobotConfig.h"
#include "Outpost/OutpostModel.h"
#include "PowerRune/TargetPositionCalculator.h"

#include <cstdio>
#include <iostream>
#include <cmath>

namespace {

// 预测瞄准点绘制（品红圆点 + 十字 + 预测时间文字），两种流水线模式统一
void drawAimPointOverlay(cv::Mat& img, const cv::Vec3f& aim_world, double aim_t,
                         const RobotTfTree& tree, const CameraProjection& proj) {
    cv::Vec3f aim_cam = tree.transformPoint(RobotTfTree::WORLD, RobotTfTree::CAMERA, aim_world);
    std::vector<cv::Point2f> pts;
    proj.projectPoints_Cam({cv::Point3f(aim_cam[0], aim_cam[1], aim_cam[2])}, pts);
    if (pts.empty()) return;

    const cv::Point2f pt = pts[0];
    const cv::Scalar color(255, 0, 255);   // 品红
    cv::circle(img, pt, 6, color, -1);
    cv::circle(img, pt, 10, color, 2);
    cv::line(img, cv::Point(pt.x - 14, pt.y), cv::Point(pt.x + 14, pt.y), color, 1);
    cv::line(img, cv::Point(pt.x, pt.y - 14), cv::Point(pt.x, pt.y + 14), color, 1);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Aim t=%.2fs", aim_t);
    cv::putText(img, buf, cv::Point(pt.x + 12, pt.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
}

} // namespace

VisualizeOutput::VisualizeOutput(std::shared_ptr<CameraProjection> camera_proj,
                                 AimPredictor& aim)
    : camera_proj_(std::move(camera_proj)),
      aim_(aim) {}

void VisualizeOutput::setMode(PipelineMode mode)
{
    mode_ = mode;   // 相机投影与输入模式绑定，不随流水线切换
}

void VisualizeOutput::syncTree(const ExtraInputInfo& info)
{
    RobotTfTree& tree = tree_;
    tree.unlock();
    tree.setChassisPosition((float)info.chassis_x, (float)info.chassis_y, (float)info.chassis_z);
    tree.setChassisEuler((float)info.chassis_yaw, (float)info.chassis_pitch, (float)info.chassis_roll);
    tree.setYaw((float)info.yaw_pos);
    tree.setPitch((float)info.pitch_angle);
    tree.lockAndComputeCache();
}

void VisualizeOutput::update(const PipelineResult& result, RobotController* rc)
{
    if (!result.valid) return;   // 无新帧时不重绘

    syncTree(result.extra_info);
    fps_.tick();

    if (mode_ == PipelineMode::OUTPOST) {
        renderOutpost(result, rc);
    } else {
        renderPowerRune(result, rc);
    }

    // 预测瞄准点：取自 AimPredictor 瞄准点序列的第一个值（两种模式统一绘制；
    // 无有效瞄准点时不画）
    if (!display_.empty()) {
        const AimPredictor::Result seq = aim_.latest();
        if (seq.valid) {
            drawAimPointOverlay(display_, seq.first_point, seq.first_predict_time,
                                tree_, *camera_proj_);
        }
    }
}

void VisualizeOutput::renderOutpost(const PipelineResult& result, RobotController* rc)
{
    const OutpostPerception& p = result.outpost;

    OutpostVisualizationData vis;
    vis.detection.objects = p.objects;

    vis.raw_pose.world_positions   = p.world_positions;
    vis.raw_pose.world_eulers      = p.world_eulers;
    vis.raw_pose.reprojected_points = p.reprojected_points;
    vis.raw_pose.valid = !p.objects.empty();

    vis.filtered_pose.valid = p.esekf_initialized;
    vis.filtered_pose.pos = p.ekf_pos;
    vis.filtered_pose.R64 = p.ekf_R64;
    vis.filtered_pose.world_points = p.ekf_world_points;
    vis.filtered_pose.pred_center_points = p.pred_center_points;

    // 瞄准点由 GimbalOutput 计算；本模式不计算，仅绘制 ESEKF 预测中心
    vis.aim.auto_aim_enable = false;

    vis.robot_state = rc ? rc->getState() : RobotController::State{};

    vis.fps = fps_.fps();
    vis.frame_timestamp = result.frame_timestamp;

    cv::Mat display = result.frame.clone();
    outpost_vis_.render(display, vis, tree_, *camera_proj_);
    display_ = std::move(display);
}

void VisualizeOutput::renderPowerRune(const PipelineResult& result, RobotController* rc)
{
    const PowerRunePerception& p = result.power_rune;

    PowerRuneVisualizationData vis;
    vis.detection.detections = p.detections;

    if (p.pose_valid) {
        vis.raw_pose.world_pos     = p.pr_world_posi;
        vis.raw_pose.world_euler   = p.pr_world_euler;
        vis.raw_pose.world_rot_mat = p.pr_world_rot_mat;
    }

    vis.filtered_pose.filtered_omega = p.filtered_omega;
    vis.filtered_pose.jump_a = p.jump_a;
    vis.filtered_pose.flip = p.flip;
    vis.filtered_pose.filtered_pos = p.filtered_pos;
    vis.filtered_pose.filtered_R = p.filtered_R;
    vis.filtered_pose.target_rotation_counts = p.filtered_rotation_counts;

    if (p.pose_valid && !p.filtered_rotation_counts.empty()) {
        vis.filtered_target_points = TargetPositionCalculator::calculate(
            p.filtered_pos, p.filtered_R, p.filtered_rotation_counts);
    }

    vis.roll_predictor.fit_valid = p.fit_valid;
    vis.roll_predictor.big_params = p.big_params;
    vis.roll_predictor.small_params = p.small_params;
    vis.roll_predictor.direction = p.direction;
    vis.roll_predictor.fit_method = p.fit_method;
    vis.roll_predictor.correction_bias = p.correction_bias;
    vis.roll_predictor.fitted_curve = p.fitted_curve;
    vis.roll_predictor.raw_points = p.raw_points;

    if (p.fit_valid && p.predictor_lambda) {
        vis.roll_predictor.predictor_prediction = (*p.predictor_lambda)(0.3f);
    }
    if (p.fit_valid && p.target_predictor) {
        // target_predictor 已为统一签名 std::vector<cv::Point3f>(double)；
        // 可视化数据仍用 Vec3f，逐个转换
        const auto pts = (*p.target_predictor)(0.3);
        vis.predictor_target_points.clear();
        vis.predictor_target_points.reserve(pts.size());
        for (const auto& pt : pts) {
            vis.predictor_target_points.emplace_back(pt.x, pt.y, pt.z);
        }
    }

    cv::Mat display = result.frame.clone();
    power_rune_vis_.render(display, vis, tree_, *camera_proj_);
    display_ = std::move(display);
}
