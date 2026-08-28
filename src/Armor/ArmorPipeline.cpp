// ArmorPipeline.cpp — 装甲板感知流水线实现（5 阶段，输出 PipelineResult）
#include "Armor/ArmorPipeline.h"
#include "common/PathResolver.h"
#include "common/RobotConfig.h"

#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>

namespace {

// 配置文件 armor 段 → OutpostESEKF::Params（字段一一对应）
OutpostESEKF::Params makeEsekfParams(const RobotConfig::ArmorParams& p) {
    OutpostESEKF::Params r;
    r.positionNoise        = p.esekf.positionNoise;
    r.rotationNoise        = p.esekf.rotationNoise;
    r.measurementNoise     = p.esekf.measurementNoise;
    r.orientationZRegNoise = p.esekf.orientationZRegNoise;
    r.dzNoise              = p.esekf.dzNoise;
    r.dzSearchRange        = p.esekf.dzSearchRange;
    r.dzLimit              = p.esekf.dzLimit;
    r.initPositionNoise    = p.esekf.initPositionNoise;
    r.initOrientationNoise = p.esekf.initOrientationNoise;
    r.initYawRateNoise     = p.esekf.initYawRateNoise;
    r.initDz2Noise         = p.esekf.initDz2Noise;
    r.initDz3Noise         = p.esekf.initDz3Noise;
    r.observationLostTimeoutSec = p.observationLostTimeoutSec;
    return r;
}

} // namespace

// ==================== 阶段上下文构造 ====================

// 构造本流水线所用的推理器（模型编译 + 预热；由 armor_infer_process 的 main 调用）
std::unique_ptr<ArmorDetect::ArmorInfer> ArmorPipeline::createInfer()
{
    const RobotConfig& cfg = RobotConfig::instance();
    // 模型路径（相对项目根目录，经 PathResolver 解析；以 / 开头为绝对路径）
    std::string model_path = (!cfg.armor.modelPath.empty() && cfg.armor.modelPath[0] == '/')
        ? cfg.armor.modelPath
        : PathResolver::resolvePath(cfg.armor.modelPath);
    // 本进程专属模型缓存目录（ONNX→IR 转换产物存放处，不存在时自动创建）
    std::string cache_dir = PathResolver::resolvePath("cache/armor");
    return std::make_unique<ArmorDetect::ArmorInfer>(
        model_path, cfg.armor.device,
        cfg.armor.inputWidth, cfg.armor.inputHeight, cfg.armor.maxBatch,
        nullptr, cache_dir);
}

ArmorPipeline::Stage4Ctx::Stage4Ctx(const RobotConfig::CameraParams& camera)
    : camera_proj(std::make_shared<CameraProjection>(
          camera.cameraMatrix, camera.distCoeffs,
          ImageResolution{camera.width, camera.height})) {}

ArmorPipeline::Stage5Ctx::Stage5Ctx(const RobotConfig::CameraParams& camera,
                                      const RobotConfig::ArmorParams& armor)
    : tree(std::make_shared<RobotTfTree>()),
      camera_proj(std::make_shared<CameraProjection>(
          camera.cameraMatrix, camera.distCoeffs,
          ImageResolution{camera.width, camera.height})),
      esekf(std::make_unique<OutpostESEKF>(tree, camera_proj,
                                    ArmorModel::OUTPOST_POINTS_3D_LIST,
                                    ArmorModel::OUTPOST_TARGET_CENTER_3D_LIST,
                                    makeEsekfParams(armor))) {
    // label 0~5 每类一个封装 EKF：无观测超时自动销毁阈值统一复用
    // armor.observation_lost_timeout（与 OutpostESEKF 同一配置）
    sp_ekf::ClassEKF::Params class_params;
    class_params.observationLostTimeoutSec = armor.observationLostTimeoutSec;
    for (auto& class_ekf : class_ekfs) {
        class_ekf = sp_ekf::ClassEKF(class_params);
    }
    // label 7~8 各一个首物体位姿保持：重置时间同样复用 observation_lost_timeout
    sp_ekf::FirstObjectTracker::Params first_object_params;
    first_object_params.resetTimeoutSec = armor.observationLostTimeoutSec;
    for (auto& tracker : first_object_trackers) {
        tracker = sp_ekf::FirstObjectTracker(first_object_params);
    }
}

// ==================== 构造 ====================

ArmorPipeline::ArmorPipeline(const std::array<int, NUM_QUEUES>& queue_max_sizes,
                                 float min_delay_seconds,
                                 const RobotConfig::CameraParams& camera)
    : queue_max_sizes_(queue_max_sizes)
    , min_delay_seconds_(min_delay_seconds)
    , s1_(RobotConfig::instance().armor.inputWidth,
          RobotConfig::instance().armor.inputHeight)
    , s4_(camera)
    , s5_(camera, RobotConfig::instance().armor)
{
    const RobotConfig& cfg = RobotConfig::instance();
    const RobotConfig::PipelineParams& pipe = cfg.armor.pipeline;

    // 推理器在独立进程 armor_infer_process 中（仅推理一步；预处理/后处理在本
    // 进程），本阶段经共享内存通信调用，推理进程未启动时阻塞等待。
    s2_.client = std::make_unique<Infer::InferShmClient>(cfg.armor.shmKey);
    s3_.postprocessor = std::make_unique<ArmorDetect::ArmorPostprocessor>(
        cfg.armor.inputWidth, cfg.armor.inputHeight);

    // 模型路径（仅用于打印 banner；推理器构造见 createInfer，由 main 调用）
    std::string model_path = (!cfg.armor.modelPath.empty() && cfg.armor.modelPath[0] == '/')
        ? cfg.armor.modelPath
        : PathResolver::resolvePath(cfg.armor.modelPath);

    std::cout << "========================================" << std::endl;
    std::cout << "Armor Pipeline (5 stages)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "    Model: " << model_path << std::endl;
    std::cout << "    Device: " << cfg.armor.device << std::endl;
    std::cout << "    Input resolution: " << cfg.armor.inputWidth << "x" << cfg.armor.inputHeight << std::endl;
    std::cout << "    Inference model max batch: " << cfg.armor.maxBatch << std::endl;
    std::cout << "    Stage batch (preprocess/infer/postprocess): "
              << pipe.preprocessBatch << "/" << pipe.inferenceBatch << "/"
              << pipe.postprocessBatch << std::endl;
    std::cout << "    Queue sizes (input/inter0..3/output):";
    for (int q : pipe.queueMaxSizes) std::cout << " " << q;
    std::cout << std::endl;
    std::cout << "    Confidence threshold: " << conf_threshold_ << std::endl;
    std::cout << "    NMS threshold: " << nms_threshold_ << std::endl;
    std::cout << "    Min delay: " << min_delay_seconds_ << "s" << std::endl;
    std::cout << "========================================" << std::endl;

    // ---- 阶段1：预处理 ----
    {
        PipelineStage<DataDeque>::Config cfg;
        cfg.input_queue  = &input_queue_;
        cfg.output_queue = &inter_queues_[0];
        cfg.input_mtx    = &input_mtx_;
        cfg.input_cv     = &input_cv_;
        cfg.max_batch    = pipe.preprocessBatch;
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
        cfg.max_batch    = pipe.inferenceBatch;
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
        cfg.max_batch    = pipe.postprocessBatch;
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

    // ---- 阶段5：OutpostESEKF ----
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
    scheduler_thread_ = std::thread(&ArmorPipeline::schedulerLoop, this);
}

// ==================== 析构 ====================

ArmorPipeline::~ArmorPipeline()
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

void ArmorPipeline::processStage1(DataDeque& data)
{
    std::vector<ArmorPipelineData*> ptrs;
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

void ArmorPipeline::processStage2(DataDeque& data)
{
    std::vector<ArmorPipelineData*> ptrs;
    ptrs.reserve(data.size());
    for (auto& d : data) ptrs.push_back(d.get());

    // 已预处理的图像（阶段1 resize 到模型输入尺寸）交给推理客户端
    std::vector<const cv::Mat*> preprocessed_ptrs;
    preprocessed_ptrs.reserve(ptrs.size());
    for (auto* d : ptrs)
        preprocessed_ptrs.push_back(&d->stage1.frame);

    // 推理输出由客户端直接 memcpy 进各数据的输出缓冲（PipelineData 自持）
    std::vector<Infer::OutputBuffer*> outs;
    outs.reserve(ptrs.size());
    for (auto* d : ptrs)
        outs.push_back(&d->stage2.output);

    // 推理器在独立进程（armor_infer_process）中：经共享内存通信调用，
    // 推理进程未启动时此处阻塞等待
    if (!s2_.client->runInference(preprocessed_ptrs, outs))
        return;   // 通信失败：丢弃本批
}

void ArmorPipeline::processStage3(DataDeque& data)
{
    std::vector<ArmorPipelineData*> ptrs;
    ptrs.reserve(data.size());
    for (auto& d : data) ptrs.push_back(d.get());

    std::vector<ArmorDetect::BatchOutput> outputs;
    std::vector<int> orig_ws, orig_hs;
    outputs.reserve(ptrs.size());
    orig_ws.reserve(ptrs.size());
    orig_hs.reserve(ptrs.size());
    for (auto* d : ptrs) {
        const Infer::OutputBuffer& out = d->stage2.output;
        outputs.push_back({out.data.data(), out.rows, out.cols});
        orig_ws.push_back(d->initial.frame.cols);
        orig_hs.push_back(d->initial.frame.rows);
    }

    std::vector<std::vector<ArmorDetect::Object>> results;
    s3_.postprocessor->postprocessBatch(outputs, orig_ws, orig_hs,
                                        /*detect_color=*/2,
                                        conf_threshold_, nms_threshold_, results);

    for (size_t i = 0; i < ptrs.size(); ++i)
        ptrs[i]->stage3.objects = std::move(results[i]);
}

void ArmorPipeline::processStage4(DataDeque& data)
{
    if (data.empty()) return;
    ArmorPipelineData* d = data.front().get();
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

    // 按装甲板种类（类别 label 0~8）组织 stage4 数据：外层 vector 索引 = 类别，
    // 每帧重置为 NUM_CLASSES 个槽位（槽内为默认值：空观测、初始化 PnP 未成功）
    d->stage4.categories.clear();
    d->stage4.categories.resize(ArmorDetect::NUM_CLASSES);

    for (size_t i = 0; i < objects.size(); ++i) {
        const auto& obj = objects[i];

        // 提取 4 个关键点作为 2D 图像点（已在原图坐标系）
        std::vector<cv::Point2f> image_points = {
            cv::Point2f(obj.landmarks[0], obj.landmarks[1]),  // 左上
            cv::Point2f(obj.landmarks[2], obj.landmarks[3]),  // 左下
            cv::Point2f(obj.landmarks[4], obj.landmarks[5]),  // 右下
            cv::Point2f(obj.landmarks[6], obj.landmarks[7])   // 右上
        };
        // 按装甲板种类分类：该物体的观测关键点与世界位姿写入对应类别
        // （obj.label 0~8）的槽位，供 OutpostESEKF 等下游阶段按类别使用
        auto& cat = d->stage4.categories[obj.label];
        cat.all_image_points.push_back(image_points);

        cv::Vec3f position_cam, euler_cam;
        bool pnp_ok = s4_.camera_proj->solvePnP_Cam(
            ArmorModel::SMALL_ARMOR_POINTS_3D_LOCAL, image_points,
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
                                                     ArmorModel::SMALL_ARMOR_POINTS_3D_LOCAL);
            s4_.camera_proj->projectPoints_Cam(reprojected_pts_3d_cam, reprojected_pts);
            reprojected.push_back(std::move(reprojected_pts));
        }
        world_positions.push_back(world_pos);
        world_eulers.push_back(world_euler);
        cat.world_pos.push_back(world_pos);      // 按类别存储：与 cat.all_image_points 一一对应
        cat.world_euler.push_back(world_euler);
    }
}

void ArmorPipeline::processStage5(DataDeque& data)
{
    if (data.empty()) return;
    ArmorPipelineData* d = data.front().get();
    const ExtraInputInfo& info = d->initial.extra_info;
    const auto& ts = d->initial.frame_timestamp;

    // 同步本阶段独立变换树（OutpostESEKF 内部投影依赖）
    RobotTfTree& tree = *s5_.tree;
    tree.unlock();
    tree.setChassisPosition((float)info.chassis_x, (float)info.chassis_y, (float)info.chassis_z);
    tree.setChassisEuler((float)info.chassis_yaw, (float)info.chassis_pitch, (float)info.chassis_roll);
    tree.setYaw((float)info.yaw_pos);
    tree.setPitch((float)info.pitch_angle);
    tree.lockAndComputeCache();

    // ── OutpostESEKF 统一帧处理（label 6 装甲板）──
    // 观测超时重置 / 初始化 / 更新（观测截断）/ 无观测仅预测均由
    // OutpostESEKF::processFrame 内部自动分派
    const auto& armor_cat = d->stage4.categories[ArmorDetect::ARMOR_CLASS];
    const bool esekf_initialized = s5_.esekf->processFrame(armor_cat.all_image_points, ts);

    // ── 移植 SuperPower EKF：label 0~5 每类一个 ──
    // 每类使用其对应的数据（stage4.categories[label]）调用一次其对应的封装实例
    // （s5_.class_ekfs[label]），初始化 / predict+update / 超时自动销毁 均由
    // ClassEKF::processFrame 内部自行判断，此处不再展开任何更新逻辑。
    for (int label = 0; label < NUM_CLASS_EKF; ++label) {
        const auto& cat = d->stage4.categories[label];
        s5_.class_ekfs[label].processFrame(cat.world_pos, cat.world_euler, ts);
    }

    // ── 首物体位姿保持：label 7~8（基地/基地大装甲）──
    for (int label = ArmorDetect::BASE_CLASS; label <= ArmorDetect::BASE_LARGE_CLASS; ++label) {
        const auto& cat = d->stage4.categories[label];
        s5_.first_object_trackers[label - ArmorDetect::BASE_CLASS]
            .processFrame(cat.world_pos, cat.world_euler, ts);
    }

    // ── 选择本帧使用的目标滤波结果 ──
    // 有效候选：OutpostESEKF（esekf_initialized）、label 0~5 移植 EKF
    // （state_available_）、label 7~8 首物体（valid()）；取车体中心距底盘系原点
    // （world 系，chassis_x/y/z）最近的那一类，stage5 输出全部使用其结果。
    const cv::Vec3d chassis_origin(info.chassis_x, info.chassis_y, info.chassis_z);
    int best_label = -1;
    double best_dist = std::numeric_limits<double>::infinity();
    if (esekf_initialized) {
        const double dist = cv::norm(s5_.esekf->getPosition() - chassis_origin);
        if (dist < best_dist) {
            best_dist = dist;
            best_label = ArmorDetect::ARMOR_CLASS;   // 6：OutpostESEKF
        }
    }
    for (int label = 0; label < NUM_CLASS_EKF; ++label) {
        if (!s5_.class_ekfs[label].stateAvailable()) continue;
        const double dist = cv::norm(s5_.class_ekfs[label].getPosition() - chassis_origin);
        if (dist < best_dist) {
            best_dist = dist;
            best_label = label;
        }
    }
    for (int label = ArmorDetect::BASE_CLASS; label <= ArmorDetect::BASE_LARGE_CLASS; ++label) {
        const sp_ekf::FirstObjectTracker& tracker =
            s5_.first_object_trackers[label - ArmorDetect::BASE_CLASS];
        if (!tracker.valid()) continue;
        const double dist = cv::norm(tracker.getPosition() - chassis_origin);
        if (dist < best_dist) {
            best_dist = dist;
            best_label = label;
        }
    }

    d->stage5.target_valid = (best_label >= 0);
    d->stage5.target_label = best_label;
    d->stage5.target_filter_type = TargetFilterType::NONE;
    if (best_label == ArmorDetect::ARMOR_CLASS) {
        // OutpostESEKF（label 6 装甲板）结果
        d->stage5.target_filter_type = TargetFilterType::OUTPOST_ESEKF;
        d->stage5.target_world_points = s5_.esekf->getWorldPoints();
        d->stage5.target_pos = s5_.esekf->getPosition();
        d->stage5.target_R64 = s5_.esekf->getRotationMatrix();
        d->stage5.target_predictor = s5_.esekf->capturePosePredictor();
        d->stage5.target_predictor_timestamp = ts;   // 快照的 dt 零点 = 本帧时间戳
        d->stage5.target_pred_center_points = (*d->stage5.target_predictor)(0.0);
    } else if (best_label >= 0 && best_label < NUM_CLASS_EKF) {
        // 移植 EKF（label 0~5）结果：车体中心 / 旋转矩阵 / 4 块装甲位置预测器
        d->stage5.target_filter_type = TargetFilterType::CLASS_EKF;
        const sp_ekf::ClassEKF& class_ekf = s5_.class_ekfs[best_label];
        d->stage5.target_pos = class_ekf.getPosition();
        d->stage5.target_R64 = class_ekf.getRotationMatrix();
        d->stage5.target_predictor = class_ekf.capturePosePredictor();  // state 可用时非空
        d->stage5.target_predictor_timestamp = ts;
        if (d->stage5.target_predictor) {
            const std::vector<cv::Point3f> pts = (*d->stage5.target_predictor)(0.0);
            d->stage5.target_pred_center_points = pts;
            d->stage5.target_world_points = pts;   // 4 块装甲位置（world，米）
        }
    } else if (best_label >= ArmorDetect::BASE_CLASS &&
               best_label <= ArmorDetect::BASE_LARGE_CLASS) {
        // 首物体位姿保持（label 7~8）结果：位置 / 旋转矩阵 / 1 点预测器
        d->stage5.target_filter_type = TargetFilterType::FIRST_OBJECT;
        const sp_ekf::FirstObjectTracker& tracker =
            s5_.first_object_trackers[best_label - ArmorDetect::BASE_CLASS];
        d->stage5.target_pos = tracker.getPosition();
        d->stage5.target_R64 = tracker.getRotationMatrix();
        d->stage5.target_predictor = tracker.capturePosePredictor();  // valid() 时非空
        d->stage5.target_predictor_timestamp = ts;
        if (d->stage5.target_predictor) {
            const std::vector<cv::Point3f> pts = (*d->stage5.target_predictor)(0.0);
            d->stage5.target_pred_center_points = pts;
            d->stage5.target_world_points = pts;   // 1 个点（world，米）
        }
    }
}

// ==================== 事件驱动调度器 ====================

void ArmorPipeline::wakeScheduler()
{
    {
        std::lock_guard<std::mutex> lock(scheduler_mtx_);
        scheduler_should_check_ = true;
    }
    scheduler_cv_.notify_one();
}

void ArmorPipeline::tryAdvanceStages()
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

void ArmorPipeline::schedulerLoop()
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
        // 推进阶段与 clear() 互斥：clear() 持 advance_mtx_ 时本循环阻塞，
        // 保证清空期间不会有 tryAdvance 拉新批或动中段队列。
        std::lock_guard<std::mutex> advance_lock(advance_mtx_);
        tryAdvanceStages();
    }
}

// ==================== 外部接口 ====================

bool ArmorPipeline::addFrame(cv::Mat frame,
                               const std::chrono::steady_clock::time_point& frame_timestamp,
                               const ExtraInputInfo& extra_info)
{
    auto data = std::make_unique<ArmorPipelineData>();
    data->initial.frame = std::move(frame);
    data->initial.frame_timestamp = frame_timestamp;
    data->initial.extra_info = extra_info;

    // 输入缓冲满时直接丢弃新帧（不阻塞），避免输入线程被流水线处理速度拖住
    {
        std::lock_guard<std::mutex> lock(input_mtx_);
        if ((int)input_queue_.size() >= queue_max_sizes_[0]) {
            return false;   // 队列满：抛弃该帧
        }
        input_queue_.push_back(std::move(data));
    }

    wakeScheduler();
    return true;            // 成功加入输入缓冲队列
}

void ArmorPipeline::fillPerception(ArmorPipelineData* d, ArmorPerception& out)
{
    out.objects = d->stage3.objects;
    out.world_positions = d->stage4.world_positions;
    out.world_eulers = d->stage4.world_eulers;
    out.reprojected_points = d->stage4.reprojected_points;

    out.target_valid = d->stage5.target_valid;
    out.target_label = d->stage5.target_label;
    out.target_filter_type = d->stage5.target_filter_type;
    out.target_pos = d->stage5.target_pos;
    out.target_R64 = d->stage5.target_R64;
    out.target_world_points = d->stage5.target_world_points;
    out.target_pred_center_points = d->stage5.target_pred_center_points;
    if (d->stage5.target_predictor) {
        out.target_predictor = std::move(d->stage5.target_predictor);
    }
    out.target_predictor_timestamp = d->stage5.target_predictor_timestamp;
    out.detection_count = d->stage3.objects.size();
    out.valid = true;
}

PipelineResult ArmorPipeline::tryPopFrame(const std::chrono::steady_clock::time_point& timestamp)
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
    if (diff >= min_delay_seconds_) {
        result.frame_timestamp = front->initial.frame_timestamp;
        result.extra_info = front->initial.extra_info;
        result.frame = std::move(front->initial.frame);
        fillPerception(front.get(), result.armor);
        result.valid = true;
        output_queue_.pop_front();
        // 输出队列腾出空间：唤醒调度器推进各阶段（尤其输出队列满导致阶段5停顿时）
        wakeScheduler();
    }

    return result;
}

void ArmorPipeline::clear()
{
    {
        // 与调度器推进互斥：期间不拉新批、不动中段队列；worker 仍可跑完当前批。
        std::lock_guard<std::mutex> advance_lock(advance_mtx_);

        // 等待全部阶段完成当前批处理（stage2 在途推理最多 2s 超时，其余毫秒级）
        stage1_.waitIdle();
        stage2_.waitIdle();
        stage3_.waitIdle();
        stage4_.waitIdle();
        stage5_.waitIdle();

        // 丢弃各阶段在途批（此刻全部空闲，无并发访问）
        stage1_.discardInFlight();
        stage2_.discardInFlight();
        stage3_.discardInFlight();
        stage4_.discardInFlight();
        stage5_.discardInFlight();

        // 清空全部缓冲队列：输入/输出（持锁），中段队列（仅调度器访问，已互斥）
        {
            std::lock_guard<std::mutex> lk(input_mtx_);
            input_queue_.clear();
        }
        for (auto& q : inter_queues_) q.clear();
        {
            std::lock_guard<std::mutex> lk(output_mtx_);
            output_queue_.clear();
        }

        // 重置 OutpostESEKF 与观测计时状态（stage5 已空闲，无竞争；
        // 初始化标志 / 观测计时状态随 reset() 一并复位）
        s5_.esekf->reset();
        // 一并销毁所有移植 EKF 封装（label 0~5），下次观测重新初始化
        for (auto& class_ekf : s5_.class_ekfs) {
            class_ekf.reset();
        }
        // 一并重置首物体位姿保持（label 7~8）
        for (auto& tracker : s5_.first_object_trackers) {
            tracker.reset();
        }

        // 队列计数归零
        for (auto& qs : queue_sizes_) qs.store(0);
    }

    // 恢复调度（advance_mtx_ 已释放，调度器可立即推进）
    wakeScheduler();
}
