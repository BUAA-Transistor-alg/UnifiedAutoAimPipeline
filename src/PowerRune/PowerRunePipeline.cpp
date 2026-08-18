#include "PowerRunePipeline.h"
#include "PathResolver.h"
#include "RobotConfig.h"
#include <string>
#include <iostream>

// ==================== 构造 ====================

PowerRunePipeline::Stage4Ctx::Stage4Ctx()
    : tree(std::make_shared<RobotTfTree>()),
      camera_proj(std::make_shared<CameraProjection>(
          RobotConfig::instance().powerRune.cameraMatrix,
          RobotConfig::instance().powerRune.distCoeffs,
          ImageResolution{RobotConfig::instance().powerRune.width,
                          RobotConfig::instance().powerRune.height})),
      pose_solver(camera_proj) {}

PowerRunePipeline::PowerRunePipeline(const std::array<int, NUM_QUEUES>& queue_max_sizes,
                                     float max_delay_seconds)
    : queue_max_sizes_(queue_max_sizes)
    , max_delay_seconds_(max_delay_seconds)
{
    const RobotConfig& cfg = RobotConfig::instance();

    // 模型路径（相对项目根目录，经 PathResolver 解析；以 / 开头为绝对路径）
    std::string model_path = (!cfg.powerRune.modelPath.empty() && cfg.powerRune.modelPath[0] == '/')
        ? cfg.powerRune.modelPath
        : PathResolver::resolvePath(cfg.powerRune.modelPath);

    conf_threshold_ = cfg.powerRune.confThreshold;
    s3_.postprocessor = std::make_unique<YoloPose::YoloPosePostprocessor>(cfg.powerRune.manualNms);
    s2_.infer_p = std::make_unique<YoloPose::YoloPoseInfer>(
        model_path, cfg.powerRune.device, cfg.powerRune.maxBatch);

    std::cout << "========================================" << std::endl;
    std::cout << "PowerRune Pipeline (5 stages)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "    Model: " << model_path << std::endl;
    std::cout << "    Device: " << cfg.powerRune.device << std::endl;
    std::cout << "    Manual NMS: " << (cfg.powerRune.manualNms ? "true" : "false") << std::endl;
    std::cout << "    Confidence threshold: " << conf_threshold_ << std::endl;
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

    // ---- 阶段4：位姿解算 + 坐标转换 ----
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

    // ---- 阶段5：滤波 + 预测 ----
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
    scheduler_thread_ = std::thread(&PowerRunePipeline::schedulerLoop, this);
}

// ==================== 析构 ====================

PowerRunePipeline::~PowerRunePipeline()
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

void PowerRunePipeline::processStage1(DataDeque& data)
{
    std::vector<PowerRunePipelineData*> ptrs;
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

void PowerRunePipeline::processStage2(DataDeque& data)
{
    std::vector<PowerRunePipelineData*> ptrs;
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

void PowerRunePipeline::processStage3(DataDeque& data)
{
    std::vector<PowerRunePipelineData*> ptrs;
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

    std::vector<std::vector<PoseDetection>> results;
    s3_.postprocessor->postprocessBatch(tensors, batch_indices, orig_ws, orig_hs,
                                        conf_threshold_, results);

    for (size_t i = 0; i < ptrs.size(); ++i)
        ptrs[i]->stage3.detections = std::move(results[i]);
}

void PowerRunePipeline::processStage4(DataDeque& data)
{
    if (data.empty()) return;
    PowerRunePipelineData* d = data.front().get();
    const ExtraInputInfo& info = d->initial.extra_info;

    // 同步本阶段独立变换树（PowerRune 无真实云台：chassis 欧拉 = imu_euler = 相机欧拉，
    // chassis 坐标 = 相机坐标，yaw_pos/pitch_angle = 0）
    RobotTfTree& tree = *s4_.tree;
    tree.unlock();
    tree.setChassisPosition((float)info.chassis_x, (float)info.chassis_y, (float)info.chassis_z);
    tree.setChassisEuler((float)info.chassis_yaw, (float)info.chassis_pitch, (float)info.chassis_roll);
    tree.setYaw((float)info.yaw_pos);
    tree.setPitch((float)info.pitch_angle);
    tree.lockAndComputeCache();

    d->stage4.combined_pose = s4_.pose_solver.estimateCombinedPose(d->stage3.detections);
    d->stage4.rotation_counts = d->stage4.combined_pose.rotation_counts;

    if (d->stage4.combined_pose.valid()) {
        d->stage4.pose_valid = true;
        // PnP 系 → Cam 系 → world 系（经 tf 树）
        cv::Vec3f position_cam = CameraProjection::pnpTvecToCamPosi(d->stage4.combined_pose.tvec);
        cv::Vec3f euler_cam    = CameraProjection::pnpRvecToEuler(d->stage4.combined_pose.rvec);
        d->stage4.pr_world_posi  = tree.transformPoint(RobotTfTree::CAMERA, RobotTfTree::WORLD, position_cam);
        d->stage4.pr_world_euler = tree.transformEuler(RobotTfTree::CAMERA, RobotTfTree::WORLD, euler_cam);
        d->stage4.pr_world_rot_mat = CoordinateTransform::eulerToRotationMatrix(d->stage4.pr_world_euler);
    } else {
        d->stage4.pose_valid = false;
    }
}

void PowerRunePipeline::processStage5(DataDeque& data)
{
    if (data.empty()) return;
    PowerRunePipelineData* d = data.front().get();

    // 计算帧间 dt（用于 RollPredictor 预测），每帧更新 last_frame_timestamp
    float inter_frame_dt = std::chrono::duration<float>(
        d->initial.frame_timestamp - s5_.last_frame_timestamp).count();

    if (d->stage4.pose_valid) {
        float time_diff = std::chrono::duration<float>(
            d->initial.frame_timestamp - s5_.last_valid_timestamp).count();
        bool is_continuous = time_diff < 0.1f; // pi/5/2.09 = 0.30063, 必须小于这个值

        // 当 RollPredictor 拟合有效 且 非连续帧时，使用 RollPredictor 的预测旋转矩阵
        cv::Mat roll_predictor_R;
        if ((!is_continuous) && s5_.roll_predictor.isValid()) {
            if (s5_.roll_predictor.computeRMSE() < 0.3f) {
                auto predicted = s5_.roll_predictor.predict_posi_and_R(inter_frame_dt);
                roll_predictor_R = predicted.second;
            }
        }

        s5_.y_axis_filter.update(
            d->stage4.pr_world_posi,
            d->stage4.pr_world_rot_mat,
            d->initial.frame_timestamp,
            is_continuous,
            roll_predictor_R);
        s5_.last_valid_timestamp = d->initial.frame_timestamp;

        d->stage5.filtered_omega = s5_.y_axis_filter.getAngularVelocity();
        d->stage5.jump_a         = s5_.y_axis_filter.getJumpA();
        d->stage5.flip           = s5_.y_axis_filter.getFlip();
        d->stage5.filtered_pos   = s5_.y_axis_filter.getPosition();
        d->stage5.filtered_R     = s5_.y_axis_filter.getRotation(true);
        d->stage5.filtered_rotation_counts = s5_.y_axis_filter.processRotationCounts(
            d->stage4.rotation_counts);

        if (d->stage5.filtered_omega > 0.0f) {
            s5_.roll_predictor.setDirection(1);
        } else if (d->stage5.filtered_omega < 0.0f) {
            s5_.roll_predictor.setDirection(-1);
        }

        cv::Vec3f filtered_posi = s5_.y_axis_filter.getPosition();
        cv::Vec3f filtered_euler = CoordinateTransform::rotationMatrixToEuler(
            s5_.y_axis_filter.getRotation(true));
        d->stage5.filtered_euler_roll = filtered_euler[2];
        cv::Mat filtered_y_axis_R = CoordinateTransform::eulerToRotationMatrix(
            filtered_euler[0], filtered_euler[1], 0.0f);
        s5_.roll_predictor.update(filtered_euler[2], d->initial.frame_timestamp,
                                  filtered_posi, filtered_y_axis_R);
    } else {
        float time_since_valid = std::chrono::duration<float>(
            d->initial.frame_timestamp - s5_.last_valid_timestamp).count();
        if (time_since_valid > 3.0f) {
            s5_.y_axis_filter.reset();
            s5_.roll_predictor.reset();
        } else {
            s5_.roll_predictor.updateWithoutData(d->initial.frame_timestamp);
        }
    }

    d->stage5.fit_valid    = s5_.roll_predictor.isValid();
    d->stage5.big_params   = s5_.roll_predictor.getBigParams();
    d->stage5.small_params = s5_.roll_predictor.getSmallParams();
    d->stage5.direction    = s5_.roll_predictor.getDirection();
    d->stage5.fit_method   = s5_.roll_predictor.getFitMethodName();
    d->stage5.correction_bias = s5_.roll_predictor.getCorrectionBias();

    if (s5_.roll_predictor.isValid()) {
        s5_.roll_predictor.getVisualizationPoints(
            d->stage5.fitted_curve, d->stage5.raw_points);
        d->stage5.predictor_lambda = s5_.roll_predictor.capturePredictor();
        if (!d->stage5.filtered_rotation_counts.empty()) {
            // 组合成靶点预测函数：位姿预测 + rotation_counts → 目标世界坐标列表
            // （单独再捕获一份，保留 predictor_lambda 供可视化位姿预测绘制）
            auto pred2 = s5_.roll_predictor.capturePredictor();
            d->stage5.target_predictor = TargetPositionCalculator::compose(
                std::move(pred2), d->stage5.filtered_rotation_counts);
        }
    }

    d->stage5.angular_velocity = s5_.y_axis_filter.getAngularVelocity();

    // 每帧结束时更新 last_frame_timestamp
    s5_.last_frame_timestamp = d->initial.frame_timestamp;
}

// ==================== 事件驱动调度器 ====================

void PowerRunePipeline::wakeScheduler()
{
    {
        std::lock_guard<std::mutex> lock(scheduler_mtx_);
        scheduler_should_check_ = true;
    }
    scheduler_cv_.notify_one();
}

void PowerRunePipeline::tryAdvanceStages()
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

void PowerRunePipeline::schedulerLoop()
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

void PowerRunePipeline::addFrame(cv::Mat frame,
                                 const std::chrono::steady_clock::time_point& frame_timestamp,
                                 const ExtraInputInfo& extra_info)
{
    auto data = std::make_unique<PowerRunePipelineData>();
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

void PowerRunePipeline::fillPerception(PowerRunePipelineData* d, PowerRunePerception& out)
{
    out.detections = d->stage3.detections;

    out.pose_valid = d->stage4.pose_valid;
    out.pr_world_posi = d->stage4.pr_world_posi;
    out.pr_world_euler = d->stage4.pr_world_euler;
    out.pr_world_rot_mat = d->stage4.pr_world_rot_mat;

    out.filtered_pos = d->stage5.filtered_pos;
    out.filtered_R = d->stage5.filtered_R;
    out.filtered_omega = d->stage5.filtered_omega;
    out.jump_a = d->stage5.jump_a;
    out.flip = d->stage5.flip;
    out.filtered_rotation_counts = d->stage5.filtered_rotation_counts;

    out.fit_valid = d->stage5.fit_valid;
    out.big_params = d->stage5.big_params;
    out.small_params = d->stage5.small_params;
    out.direction = d->stage5.direction;
    out.fit_method = d->stage5.fit_method;
    out.correction_bias = d->stage5.correction_bias;
    out.fitted_curve = d->stage5.fitted_curve;
    out.raw_points = d->stage5.raw_points;
    if (d->stage5.predictor_lambda) {
        out.predictor_lambda = std::move(d->stage5.predictor_lambda);
    }
    if (d->stage5.target_predictor) {
        out.target_predictor = std::move(d->stage5.target_predictor);
    }
    out.detection_count = d->stage3.detections.size();
    out.valid = true;
}

PipelineResult PowerRunePipeline::tryPopFrame(const std::chrono::steady_clock::time_point& timestamp)
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
        fillPerception(front.get(), result.power_rune);
        result.valid = true;
        output_queue_.pop_front();
    }

    return result;
}

void PowerRunePipeline::clear()
{
    {
        std::lock_guard<std::mutex> lk(input_mtx_);
        input_queue_.clear();
    }
    {
        std::lock_guard<std::mutex> lk(output_mtx_);
        output_queue_.clear();
    }

    // 重置滤波与预测状态
    s5_.y_axis_filter.reset();
    s5_.roll_predictor.reset();
    s5_.last_valid_timestamp = std::chrono::steady_clock::time_point();
    s5_.last_frame_timestamp = std::chrono::steady_clock::time_point();

    wakeScheduler();
}
