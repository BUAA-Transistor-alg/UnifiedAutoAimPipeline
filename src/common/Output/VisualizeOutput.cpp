// VisualizeOutput.cpp — 可视化输出模式实现
#include "common/Output/VisualizeOutput.h"
#include "common/RobotConfig.h"
#include "Armor/ArmorModel.h"
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

// cam 系 (0,1,0) 点（相机 y 轴正方向 1m 处的点）在当前图像上的投影叉丝绘制。
// world_pt 为某云台位姿下该点（= 该位姿相机原点 + 相机 y 轴方向）的 world 系坐标，
// 经当前树（cur_tree，已同步并上锁）投影到当前画面。
// full_cross 为 true 时绘制四角括弧形叉丝（四个 L 形角括弧，中间行/列留空、互不相连），
// 否则绘制普通十字（均无文字）。
void drawCamYAxisCrosshair(cv::Mat& img, const cv::Vec3f& world_pt,
                           const RobotTfTree& cur_tree, const CameraProjection& proj,
                           const cv::Scalar& color, bool full_cross) {
    cv::Vec3f cam_pt = cur_tree.transformPoint(RobotTfTree::WORLD, RobotTfTree::CAMERA, world_pt);
    std::vector<cv::Point2f> pts;
    proj.projectPoints_Cam({cv::Point3f(cam_pt[0], cam_pt[1], cam_pt[2])}, pts);
    if (pts.empty()) return;
    const cv::Point2f p = pts[0];
    if (!std::isfinite(p.x) || !std::isfinite(p.y)) return;

    const int arm = 20;   // 半臂长（像素）
    if (full_cross) {
        // 四角括弧形叉丝：中间行/列留空，四条臂互不相连（对应字符画黑色部分为线条的样式）
        const int gap = 6;    // 括弧角点到中心的距离（中心空腔半宽，约为总半长的 1/3）
        const int len = arm - gap;   // 每段臂长（从角点向外延伸）
        // 左上角 (cx-gap, cy-gap)：竖臂向上、横臂向左
        cv::line(img, cv::Point((int)p.x - gap, (int)(p.y - gap)), cv::Point((int)p.x - gap, (int)(p.y - gap - len)), color, 2);
        cv::line(img, cv::Point((int)p.x - gap, (int)(p.y - gap)), cv::Point((int)(p.x - gap - len), (int)p.y - gap), color, 2);
        // 右上角 (cx+gap, cy-gap)：竖臂向上、横臂向右
        cv::line(img, cv::Point((int)p.x + gap, (int)(p.y - gap)), cv::Point((int)p.x + gap, (int)(p.y - gap - len)), color, 2);
        cv::line(img, cv::Point((int)p.x + gap, (int)(p.y - gap)), cv::Point((int)(p.x + gap + len), (int)p.y - gap), color, 2);
        // 左下角 (cx-gap, cy+gap)：竖臂向下、横臂向左
        cv::line(img, cv::Point((int)p.x - gap, (int)(p.y + gap)), cv::Point((int)p.x - gap, (int)(p.y + gap + len)), color, 2);
        cv::line(img, cv::Point((int)p.x - gap, (int)(p.y + gap)), cv::Point((int)(p.x - gap - len), (int)p.y + gap), color, 2);
        // 右下角 (cx+gap, cy+gap)：竖臂向下、横臂向右
        cv::line(img, cv::Point((int)p.x + gap, (int)(p.y + gap)), cv::Point((int)p.x + gap, (int)(p.y + gap + len)), color, 2);
        cv::line(img, cv::Point((int)p.x + gap, (int)(p.y + gap)), cv::Point((int)(p.x + gap + len), (int)p.y + gap), color, 2);
    } else {
        // 普通十字
        cv::line(img, cv::Point((int)(p.x - arm), (int)p.y), cv::Point((int)(p.x + arm), (int)p.y), color, 2);
        cv::line(img, cv::Point((int)p.x, (int)(p.y - arm)), cv::Point((int)p.x, (int)(p.y + arm)), color, 2);
    }
}

// 与瞄准点对应的云台位姿可视化（两种流水线模式统一）：
//   - 当前位姿：cam 系 (0,1,0) 点在 world 系的坐标在图像上的投影（绿色普通十字，恒为图像中心）；
//   - 需要的位姿：解算出的"需要的云台位姿"下，cam 系 (0,1,0) 点在 world 系的坐标在图像上的
//     投影（井形叉丝，与当前位姿以形状区分；gimbal 模式开启时按 ctx.fire_out 首元素着色：
//     false → 红，true → 绿，未开启时保持绿色）。
// req_tree 为独立复现需要的云台位姿的树（底盘沿用当前，yaw/pitch 用解算原始关节角）。
void drawGimbalYAxisOverlays(cv::Mat& img, const SequencePredictor::Result& seq,
                             const RobotTfTree& cur_tree, RobotTfTree& req_tree,
                             const CameraProjection& proj, const OutputContext& ctx) {
    if (seq.items.empty()) return;
    const SequencePredictor::Item& item = seq.items.front();

    // 当前位姿：cam 系 (0,1,0) → world → 投影（往返恒为 (0,1,0) → 图像中心）
    const cv::Vec3f cur_world = cur_tree.transformPoint(RobotTfTree::CAMERA, RobotTfTree::WORLD,
                                                       cv::Vec3f(0.0f, 1.0f, 0.0f));
    drawCamYAxisCrosshair(img, cur_world, cur_tree, proj, cv::Scalar(0, 255, 0), false);

    // 需要的云台位姿：底盘沿用当前树状态，yaw/pitch 关节角 = 解算原始关节角
    req_tree.unlock();
    {
        const RobotTfTree::State st = cur_tree.saveState();
        req_tree.setChassisPosition(st.chassisPosition[0], st.chassisPosition[1], st.chassisPosition[2]);
        req_tree.setChassisEuler(st.chassisEuler[0], st.chassisEuler[1], st.chassisEuler[2]);
        req_tree.setYaw(item.gimbal_yaw);
        req_tree.setPitch(item.gimbal_pitch);
    }
    req_tree.lockAndComputeCache();
    const cv::Vec3f req_world = req_tree.transformPoint(RobotTfTree::CAMERA, RobotTfTree::WORLD,
                                                       cv::Vec3f(0.0f, 1.0f, 0.0f));
    // 需要的位姿井形叉丝颜色：gimbal 模式开启时取 fire_out 首元素
    // （false → 红，true → 绿）；未开启或 fire_out 为空时保持绿色
    cv::Scalar req_color(0, 255, 0);
    if (ctx.gimbal_enabled && (!ctx.fire_out.empty()) && (!ctx.fire_out.front())) {
        req_color = cv::Scalar(0, 0, 255);
    }
    drawCamYAxisCrosshair(img, req_world, cur_tree, proj, req_color, true);
}

} // namespace

VisualizeOutput::VisualizeOutput(std::shared_ptr<CameraProjection> camera_proj,
                                 SequencePredictor& aim)
    : camera_proj_(std::move(camera_proj)),
      aim_(aim) {}

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

void VisualizeOutput::update(const PipelineResult& result, RobotController* rc,
                             OutputContext& ctx)
{
    if (!result.valid) return;   // 无新帧时不重绘

    syncTree(result.extra_info);
    fps_.tick();

    // 渲染当前流水线模式的画面（写入成员缓冲 render_buf_，复用以免每帧分配）
    const PipelineMode mode = mode_.load(std::memory_order_relaxed);
    if (mode == PipelineMode::ARMOR)
        renderArmor(result, rc);
    else
        renderPowerRune(result, rc);

    // 预测瞄准点：取自 SequencePredictor 瞄准点序列的第一个值（两种模式统一绘制；
    // 无有效瞄准点时不画）。display_ 由可视化线程写入、主线程读取，加锁保护。
    std::lock_guard<std::mutex> lock(display_mtx_);
    display_ = render_buf_;
    if (!display_.empty()) {
        const SequencePredictor::Result seq = aim_.latest();
        if (seq.valid) {
            drawAimPointOverlay(display_, seq.first_point, seq.first_predict_time,
                                tree_, *camera_proj_);
            // 与瞄准点对应的云台位姿下 cam 系 (0,1,0) 点投影（紫）/ 当前位姿（绿）
            drawGimbalYAxisOverlays(display_, seq, tree_, required_pose_tree_, *camera_proj_, ctx);
        }
    }
}

cv::Mat VisualizeOutput::display() const
{
    std::lock_guard<std::mutex> lock(display_mtx_);
    return display_;   // 浅拷贝：共享像素数据（引用计数原子安全）
}

void VisualizeOutput::renderArmor(const PipelineResult& result, RobotController* rc)
{
    const ArmorPerception& p = result.armor;

    ArmorVisualizationData vis;
    vis.detection.objects = p.objects;

    vis.raw_pose.world_positions   = p.world_positions;
    vis.raw_pose.world_eulers      = p.world_eulers;
    vis.raw_pose.reprojected_points = p.reprojected_points;
    vis.raw_pose.valid = !p.objects.empty();

    vis.filtered_pose.valid = p.target_valid;
    vis.filtered_pose.pos = p.target_pos;
    vis.filtered_pose.R64 = p.target_R64;
    vis.filtered_pose.world_points = p.target_world_points;
    vis.filtered_pose.pred_center_points = p.target_pred_center_points;

    // 瞄准点由 GimbalOutput 计算；本模式不计算，仅绘制目标滤波预测中心
    vis.aim.auto_aim_enable = false;

    // XY 平面窗口数据：自身底盘位置（取帧时刻 ExtraInputInfo 快照）+ 瞄准目标
    // （SequencePredictor 瞄准点序列第一个值，与主画面瞄准点绘制同源）
    vis.xy.chassis_position = cv::Vec3f((float)result.extra_info.chassis_x,
                                        (float)result.extra_info.chassis_y,
                                        (float)result.extra_info.chassis_z);
    const SequencePredictor::Result seq = aim_.latest();
    vis.xy.aim_valid = seq.valid;
    vis.xy.aim_point = seq.first_point;

    vis.robot_state = rc ? rc->getState() : RobotController::State{};

    vis.fps = fps_.fps();
    vis.frame_timestamp = result.frame_timestamp;

    // 成员缓冲复用：尺寸/类型不变时 create 不重新分配，仅 copyTo 拷贝像素
    render_buf_.create(result.frame.size(), result.frame.type());
    result.frame.copyTo(render_buf_);
    armor_vis_.render(render_buf_, vis, tree_, *camera_proj_);

    // XY 平面窗口每帧刷新（窗口未开启时内部直接返回；仅可视化线程调用）
    armor_vis_.renderXY(vis);
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

    // 成员缓冲复用：尺寸/类型不变时 create 不重新分配，仅 copyTo 拷贝像素
    render_buf_.create(result.frame.size(), result.frame.type());
    result.frame.copyTo(render_buf_);
    power_rune_vis_.render(render_buf_, vis, tree_, *camera_proj_);
}
