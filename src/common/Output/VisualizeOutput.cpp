// VisualizeOutput.cpp — 可视化输出模式实现
#include "Output/VisualizeOutput.h"
#include "RobotConfig.h"
#include "OutpostModel.h"
#include "TargetPositionCalculator.h"

#include <iostream>
#include <cmath>

VisualizeOutput::VisualizeOutput()
    : camera_proj_(std::make_shared<CameraProjection>(
          RobotConfig::instance().camera.cameraMatrix,
          RobotConfig::instance().camera.distCoeffs,
          ImageResolution{RobotConfig::instance().camera.width,
                          RobotConfig::instance().camera.height})) {}

void VisualizeOutput::setMode(PipelineMode mode)
{
    mode_ = mode;
    const RobotConfig& cfg = RobotConfig::instance();
    if (mode == PipelineMode::OUTPOST) {
        camera_proj_ = std::make_shared<CameraProjection>(
            cfg.camera.cameraMatrix, cfg.camera.distCoeffs,
            ImageResolution{cfg.camera.width, cfg.camera.height});
    } else {
        camera_proj_ = std::make_shared<CameraProjection>(
            cfg.powerRune.cameraMatrix, cfg.powerRune.distCoeffs,
            ImageResolution{cfg.powerRune.width, cfg.powerRune.height});
    }
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
        vis.predictor_target_points = (*p.target_predictor)(0.3f);
    }

    cv::Mat display = result.frame.clone();
    power_rune_vis_.render(display, vis, tree_, *camera_proj_);
    display_ = std::move(display);
}
