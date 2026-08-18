// OutpostPipeline.cpp — 前哨站感知流水线实现（5 阶段，输出 PipelineResult）
#include "OutpostPipeline.h"
#include "PathResolver.h"
#include "RobotConfig.h"

#include <string>
#include <iostream>
#include <cmath>

// ==================== 阶段上下文构造 ====================

OutpostPipeline::Stage4Ctx::Stage4Ctx(const RobotConfig::CameraParams& camera)
    : camera_proj(std::make_shared<CameraProjection>(
          camera.cameraMatrix, camera.distCoeffs,
          ImageResolution{camera.width, camera.height})) {}

OutpostPipeline::Stage5Ctx::Stage5Ctx(const RobotConfig::CameraParams& camera)
    : tree(std::make_shared<RobotTfTree>()),
      camera_proj(std::make_shared<CameraProjection>(
          camera.cameraMatrix, camera.distCoeffs,
          ImageResolution{camera.width, camera.height})),
      esekf(std::make_unique<ESEKF>(tree, camera_proj,
                                    OutpostModel::OUTPOST_POINTS_3D_LIST,
                                    OutpostModel::OUTPOST_TARGET_CENTER_3D_LIST)) {}

// ==================== 构造 ====================

OutpostPipeline::OutpostPipeline(const std::array<int, NUM_QUEUES>& queue_max_sizes,
                                 float max_delay_seconds,
                                 const RobotConfig::CameraParams& camera)
    : queue_max_sizes_(queue_max_sizes)
    , max_delay_seconds_(max_delay_seconds)
    , s4_(camera)
    , s5_(camera)
{
    const RobotConfig& cfg = RobotConfig::instance();

    // 模型路径（相对项目根目录，经 PathResolver 解析；以 / 开头为绝对路径）
    std::string model_path = (!cfg.outpost.modelPath.empty() && cfg.outpost.modelPath[0] == '/')
        ? cfg.outpost.modelPath
        : PathResolver::resolvePath(cfg.outpost.modelPath);

    s2_.infer_p = std::make_unique<OutpostDetect::OutpostInfer>(
        model_path, cfg.outpost.device, MAX_INFERENCE_BATCH);
    s3_.postprocessor = std::make_unique<OutpostDetect::OutpostPostprocessor>();

    std::cout << "========================================" << std::endl;
    std::cout << "Outpost Pipeline (5 stages)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "    Model: " << model_path << std::endl;
    std::cout << "    Device: " << cfg.outpost.device << std::endl;
    std::cout << "    Confidence threshold: " << conf_threshold_ << std::endl;
    std::cout << "    NMS threshold: " << nms_threshold_ << std::endl;
    std::cout << "    Max delay: " << max_delay_seconds_ << "s" << std::endl;
    std::cout << "========================================" << std::endl;

    // ---- 阶段1：预处理 ----
    {
        PipelineStage<DataDeque>::Config cfg;
        cfg.input_queue  = &input_queue_;
        cfg.output_queue = &inter_queues_[0];
        cfg.input_mtx    = &input_mtx_;
        cfg.input_cv     = &input_cv_;
        cfg.max_batch    = MAX_PREPROCESS_BATCH;
        cfg.output_max   = queue_max_sizes_[1];
        cfg.process_fn   = [this](DataDeque& data) { processStage1(data); };
        cfg.on_done      = [this]() { wakeScheduler(); };
        stage1_.launch(cfg);
    }

    // ---- 阶段2：推理 ----
    {
        PipelineStage<DataDeque>::Config cfg;
        cfg.input_queue  = &inter_queues_[0];
        cfg.output_queue = &inter_queues_[1];
        cfg.max_batch    = MAX_INFERENCE_BATCH;
        cfg.output_max   = queue_max_sizes_[2];
        cfg.process_fn   = [this](DataDeque& data) { processStage2(data); };
        cfg.on_done      = [this]() { wakeScheduler(); };
        stage2_.launch(cfg);
    }

    // ---- 阶段3：后处理 ----
    {
        PipelineStage<DataDeque>::Config cfg;
        cfg.input_queue  = &inter_queues_[1];
        cfg.output_queue = &inter_queues_[2];
        cfg.max_batch    = MAX_POSTPROCESS_BATCH;
        cfg.output_max   = queue_max_sizes_[3];
        cfg.process_fn   = [this](DataDeque& data) { processStage3(data); };
        cfg.on_done      = [this]() { wakeScheduler(); };
        stage3_.launch(cfg);
    }

    // ---- 阶段4：PnP + 坐标转换 ----
    {
        PipelineStage<DataDeque>::Config cfg;
        cfg.input_queue  = &inter_queues_[2];
        cfg.output_queue = &inter_queues_[3];
        cfg.max_batch    = 1;
        cfg.output_max   = queue_max_sizes_[4];
        cfg.process_fn   = [this](DataDeque& data) { processStage4(data); };
        cfg.on_done      = [this]() { wakeScheduler(); };
        stage4_.launch(cfg);
    }

    // ---- 阶段5：ESEKF ----
    {
        PipelineStage<DataDeque>::Config cfg;
        cfg.input_queue  = &inter_queues_[3];
        cfg.output_queue = &output_queue_;
        cfg.output_mtx   = &output_mtx_;
        cfg.max_batch    = 1;
        cfg.output_max   = queue_max_sizes_[5];
        cfg.process_fn   = [this](DataDeque& data) { processStage5(data); };
        cfg.on_done      = [this]() { wakeScheduler(); };
        stage5_.launch(cfg);
    }

    // 阶段线程启动后再启动调度器
    scheduler_thread_ = std::thread(&OutpostPipeline::schedulerLoop, this);
}

// ==================== 析构 ====================

OutpostPipeline::~OutpostPipeline()
{
    scheduler_exit_.store(true);
    {
        std::lock_guard<std::mutex> lock(scheduler_mtx_);
        scheduler_should_check_ = true;
    }
    scheduler_cv_.notify_one();

    stage1_.shutdown();
    stage2_.shutdown();
    stage3_.shutdown();
    stage4_.shutdown();
    stage5_.shutdown();

    if (scheduler_thread_.joinable()) scheduler_thread_.join();
}

// ==================== 阶段处理函数 ====================

void OutpostPipeline::processStage1(DataDeque& data)
{
    std::vector<OutpostPipelineData*> ptrs;
    ptrs.reserve(data.size());
    for (auto& d : data) ptrs.push_back(d.get());

    std::vector<const cv::Mat*> frame_ptrs;
    frame_ptrs.reserve(ptrs.size());
    for (auto* d : ptrs) frame_ptrs.push_back(&d->initial.frame);

    std::vector<cv::Mat*> preprocessed_ptrs;
    preprocessed_ptrs.reserve(ptrs.size());
    for (auto* d : ptrs) preprocessed_ptrs.push_back(&d->stage1.frame);

    s1_.preprocessor.preprocess(frame_ptrs, preprocessed_ptrs);
}

void OutpostPipeline::processStage2(DataDeque& data)
{
    std::vector<OutpostPipelineData*> ptrs;
    ptrs.reserve(data.size());
    for (auto& d : data) ptrs.push_back(d.get());

    std::vector<const cv::Mat*> preprocessed_ptrs;
    preprocessed_ptrs.reserve(ptrs.size());
    for (auto* d : ptrs)
        preprocessed_ptrs.push_back(&d->stage1.frame);

    auto outputs = s2_.infer_p->runInference(preprocessed_ptrs);
    for (size_t i = 0; i < ptrs.size(); ++i) {
        ptrs[i]->stage2.output_tensor = outputs[i].first;
        ptrs[i]->stage2.batch_index   = outputs[i].second;
    }
}

void OutpostPipeline::processStage3(DataDeque& data)
{
    std::vector<OutpostPipelineData*> ptrs;
    ptrs.reserve(data.size());
    for (auto& d : data) ptrs.push_back(d.get());

    std::vector<std::shared_ptr<ov::Tensor>> tensors;
    std::vector<int> batch_indices, orig_ws, orig_hs;
    tensors.reserve(ptrs.size());
    batch_indices.reserve(ptrs.size());
    orig_ws.reserve(ptrs.size());
    orig_hs.reserve(ptrs.size());
    for (auto* d : ptrs) {
        tensors.push_back(d->stage2.output_tensor);
        batch_indices.push_back(d->stage2.batch_index);
        orig_ws.push_back(d->initial.frame.cols);
        orig_hs.push_back(d->initial.frame.rows);
    }

    std::vector<std::vector<OutpostDetect::Object>> results;
    s3_.postprocessor->postprocessBatch(tensors, batch_indices, orig_ws, orig_hs,
                                        /*detect_color=*/2,
                                        conf_threshold_, nms_threshold_, results);

    for (size_t i = 0; i < ptrs.size(); ++i)
        ptrs[i]->stage3.objects = std::move(results[i]);
}

void OutpostPipeline::processStage4(DataDeque& data)
{
    if (data.empty()) return;
    OutpostPipelineData* d = data.front().get();
    const ExtraInputInfo& info = d->initial.extra_info;

    // 同步本阶段独立变换树（ExtraInputInfo = StrictPose + 底盘 xyz）
    RobotTfTree& tree = s4_.tree;
    tree.unlock();
    tree.setChassisPosition((float)info.chassis_x, (float)info.chassis_y, (float)info.chassis_z);
    tree.setChassisEuler((float)info.chassis_yaw, (float)info.chassis_pitch, (float)info.chassis_roll);
    tree.setYaw((float)info.yaw_pos);
    tree.setPitch((float)info.pitch_angle);
    tree.lockAndComputeCache();

    const auto& objects = d->stage3.objects;
    auto& world_positions  = d->stage4.world_positions;
    auto& world_eulers     = d->stage4.world_eulers;
    auto& reprojected      = d->stage4.reprojected_points;

    for (size_t i = 0; i < objects.size(); ++i) {
        const auto& obj = objects[i];

        // 提取 4 个关键点作为 2D 图像点（已在原图坐标系）
        std::vector<cv::Point2f> image_points = {
            cv::Point2f(obj.landmarks[0], obj.landmarks[1]),  // 左上
            cv::Point2f(obj.landmarks[2], obj.landmarks[3]),  // 左下
            cv::Point2f(obj.landmarks[4], obj.landmarks[5]),  // 右下
            cv::Point2f(obj.landmarks[6], obj.landmarks[7])   // 右上
        };

        cv::Vec3f position_cam, euler_cam;
        bool pnp_ok = s4_.camera_proj->solvePnP_Cam(
            OutpostModel::OBJECT_POINTS_3D_LOCAL, image_points,
            {cv::SOLVEPNP_IPPE, cv::SOLVEPNP_ITERATIVE},
            position_cam, euler_cam);

        cv::Vec3f world_pos(0, 0, 0);
        cv::Vec3f world_euler(0, 0, 0);
        if (pnp_ok) {
            world_pos = tree.transformPoint(RobotTfTree::CAMERA, RobotTfTree::WORLD, position_cam);
            world_euler = tree.transformEuler(RobotTfTree::CAMERA, RobotTfTree::WORLD, euler_cam);

            // 3D 角点重投影（图像坐标，供可视化绘制序号）
            std::vector<cv::Point2f> reprojected_pts;
            std::vector<cv::Point3f> reprojected_pts_3d_cam =
                CoordinateTransform::transformPoints(position_cam, euler_cam,
                                                     OutpostModel::OBJECT_POINTS_3D_LOCAL);
            s4_.camera_proj->projectPoints_Cam(reprojected_pts_3d_cam, reprojected_pts);
            reprojected.push_back(std::move(reprojected_pts));
        }
        world_positions.push_back(world_pos);
        world_eulers.push_back(world_euler);

        // 第 0 物体：保存观测关键点 + 初始化 PnP（面 0 模型），供 ESEKF 阶段使用
        if (i == 0) {
            d->stage4.first_image_points = image_points;

            cv::Vec3f position_c, euler_c;
            bool init_pnp_ok = s4_.camera_proj->solvePnP_Cam(
                OutpostModel::OUTPOST_POINTS_3D_LIST[0], image_points,
                {cv::SOLVEPNP_IPPE, cv::SOLVEPNP_ITERATIVE},
                position_c, euler_c);
            if (init_pnp_ok) {
                cv::Vec3f pos_center = tree.transformPoint(RobotTfTree::CAMERA, RobotTfTree::WORLD, position_c);
                cv::Vec3f euler_center = tree.transformEuler(RobotTfTree::CAMERA, RobotTfTree::WORLD, euler_c);
                cv::Mat R_center = CoordinateTransform::eulerToRotationMatrix(euler_center);
                d->stage4.init_pnp_ok = true;
                d->stage4.init_pos    = pos_center;
                d->stage4.init_R      = R_center;
            }
        }
    }
}

void OutpostPipeline::processStage5(DataDeque& data)
{
    if (data.empty()) return;
    OutpostPipelineData* d = data.front().get();
    const ExtraInputInfo& info = d->initial.extra_info;
    const auto& ts = d->initial.frame_timestamp;

    // 同步本阶段独立变换树（ESEKF 内部投影依赖）
    RobotTfTree& tree = *s5_.tree;
    tree.unlock();
    tree.setChassisPosition((float)info.chassis_x, (float)info.chassis_y, (float)info.chassis_z);
    tree.setChassisEuler((float)info.chassis_yaw, (float)info.chassis_pitch, (float)info.chassis_roll);
    tree.setYaw((float)info.yaw_pos);
    tree.setPitch((float)info.pitch_angle);
    tree.lockAndComputeCache();

    // ── 观测丢失计时：连续超过阈值未观测到物体则重置 ESEKF ──
    const bool has_obs = !d->stage4.first_image_points.empty();
    if (has_obs) {
        s5_.last_observation_time = ts;
        s5_.has_observation_time = true;
    }
    const double timeout = RobotConfig::instance().outpost.observationLostTimeoutSec;
    if (s5_.has_observation_time &&
        std::chrono::duration<double>(ts - s5_.last_observation_time).count() > timeout) {
        s5_.esekf->reset();
        s5_.esekf_initialized = false;
        s5_.has_observation_time = false;
    }

    // ── ESEKF：初始化 / 更新 / 仅预测 ──
    if (!s5_.esekf_initialized) {
        if (d->stage4.init_pnp_ok) {
            s5_.esekf->init(cv::Vec3d(d->stage4.init_pos[0], d->stage4.init_pos[1], d->stage4.init_pos[2]),
                            d->stage4.init_R, ts);
            s5_.esekf_initialized = true;
        }
    } else {
        if (has_obs) {
            s5_.esekf->update(std::vector<std::vector<cv::Point2f>>{d->stage4.first_image_points}, ts);
        } else {
            s5_.esekf->predict(ts);   // 无观测，仅推进运动模型
        }
    }

    // ── 输出本帧滤波结果（供输出模式使用） ──
    d->stage5.esekf_initialized = s5_.esekf_initialized;
    if (s5_.esekf_initialized) {
        d->stage5.ekf_world_points = s5_.esekf->getWorldPoints();
        d->stage5.ekf_pos = s5_.esekf->getPosition();
        d->stage5.ekf_R64 = s5_.esekf->getRotationMatrix();
        d->stage5.predictor = s5_.esekf->capturePosePredictor();
        d->stage5.pred_center_points = (*d->stage5.predictor)(0.0);
    }
}

// ==================== 事件驱动调度器 ====================

void OutpostPipeline::wakeScheduler()
{
    {
        std::lock_guard<std::mutex> lock(scheduler_mtx_);
        scheduler_should_check_ = true;
    }
    scheduler_cv_.notify_one();
}

void OutpostPipeline::tryAdvanceStages()
{
    stage1_.tryAdvance();
    stage2_.tryAdvance();
    stage3_.tryAdvance();
    stage4_.tryAdvance();
    stage5_.tryAdvance();

    {
        std::lock_guard<std::mutex> lk(input_mtx_);
        queue_sizes_[0].store(static_cast<int>(input_queue_.size()));
    }
    queue_sizes_[1].store(static_cast<int>(inter_queues_[0].size()));
    queue_sizes_[2].store(static_cast<int>(inter_queues_[1].size()));
    queue_sizes_[3].store(static_cast<int>(inter_queues_[2].size()));
    queue_sizes_[4].store(static_cast<int>(inter_queues_[3].size()));
    {
        std::lock_guard<std::mutex> lk(output_mtx_);
        queue_sizes_[5].store(static_cast<int>(output_queue_.size()));
    }
}

void OutpostPipeline::schedulerLoop()
{
    while (true) {
        {
            std::unique_lock<std::mutex> lock(scheduler_mtx_);
            scheduler_cv_.wait(lock, [this]() {
                return scheduler_should_check_ || scheduler_exit_.load();
            });
            if (scheduler_exit_.load()) return;
            scheduler_should_check_ = false;
        }
        tryAdvanceStages();
    }
}

// ==================== 外部接口 ====================

void OutpostPipeline::addFrame(cv::Mat frame,
                               const std::chrono::steady_clock::time_point& frame_timestamp,
                               const ExtraInputInfo& extra_info)
{
    auto data = std::make_unique<OutpostPipelineData>();
    data->initial.frame = std::move(frame);
    data->initial.frame_timestamp = frame_timestamp;
    data->initial.extra_info = extra_info;

    // 输入缓冲满时直接丢弃新帧（不阻塞），避免输入线程被流水线处理速度拖住
    {
        std::lock_guard<std::mutex> lock(input_mtx_);
        if ((int)input_queue_.size() >= queue_max_sizes_[0]) {
            return;
        }
        input_queue_.push_back(std::move(data));
    }

    wakeScheduler();
}

void OutpostPipeline::fillPerception(OutpostPipelineData* d, OutpostPerception& out)
{
    out.objects = d->stage3.objects;
    out.world_positions = d->stage4.world_positions;
    out.world_eulers = d->stage4.world_eulers;
    out.reprojected_points = d->stage4.reprojected_points;

    out.esekf_initialized = d->stage5.esekf_initialized;
    out.ekf_pos = d->stage5.ekf_pos;
    out.ekf_R64 = d->stage5.ekf_R64;
    out.ekf_world_points = d->stage5.ekf_world_points;
    out.pred_center_points = d->stage5.pred_center_points;
    if (d->stage5.predictor) {
        out.predictor = std::move(d->stage5.predictor);
    }
    out.detection_count = d->stage3.objects.size();
    out.valid = true;
}

PipelineResult OutpostPipeline::tryPopFrame(const std::chrono::steady_clock::time_point& timestamp)
{
    PipelineResult result;

    result.queue_sizes.input   = queue_sizes_[0].load();
    result.queue_sizes.inter0  = queue_sizes_[1].load();
    result.queue_sizes.inter1  = queue_sizes_[2].load();
    result.queue_sizes.inter2  = queue_sizes_[3].load();
    result.queue_sizes.inter3  = queue_sizes_[4].load();
    result.queue_sizes.output  = queue_sizes_[5].load();

    std::lock_guard<std::mutex> lock(output_mtx_);
    if (output_queue_.empty()) return result;

    auto& front = output_queue_.front();
    float diff = std::chrono::duration<float>(
        timestamp - front->initial.frame_timestamp).count();
    if (diff >= max_delay_seconds_) {
        result.frame_timestamp = front->initial.frame_timestamp;
        result.extra_info = front->initial.extra_info;
        result.frame = std::move(front->initial.frame);
        fillPerception(front.get(), result.outpost);
        result.valid = true;
        output_queue_.pop_front();
        // 输出队列腾出空间：唤醒调度器推进各阶段（尤其输出队列满导致阶段5停顿时）
        wakeScheduler();
    }

    return result;
}

void OutpostPipeline::clear()
{
    // 清空输入/输出队列（阶段间队列仅调度器访问，随输入清空后自然排空）
    {
        std::lock_guard<std::mutex> lk(input_mtx_);
        input_queue_.clear();
    }
    {
        std::lock_guard<std::mutex> lk(output_mtx_);
        output_queue_.clear();
    }

    // 重置 ESEKF 与观测计时状态
    s5_.esekf->reset();
    s5_.esekf_initialized = false;
    s5_.has_observation_time = false;

    // 唤醒调度器推进（排空各阶段 in-flight 数据）
    wakeScheduler();
}
