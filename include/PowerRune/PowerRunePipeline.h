#ifndef POWER_RUNE_PIPELINE_H
#define POWER_RUNE_PIPELINE_H

#include "common/PipelineStage.h"
#include "PowerRune/YoloPoseInfer.h"
#include "PowerRune/PowerRunePoseSolver.h"
#include "PowerRune/YAxisFilter.h"
#include "PowerRune/RollPredictor.h"
#include "PowerRune/TargetPositionCalculator.h"
#include "common/TransformTree/RobotTfTree.h"
#include "common/Input/IInputMode.h"
#include "common/IPipeline.h"
#include "common/RobotConfig.h"
#include "common/Infer/InferShmClient.h"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <functional>
#include <memory>
#include <vector>
#include <deque>
#include <array>
#include <utility>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <stdexcept>

/**
 * @brief 能量机关感知流水线数据：贯穿所有处理阶段。
 */
struct PowerRunePipelineData {
    // ==================== 初始数据（处理前外部传入） ====================
    struct InitialData {
        cv::Mat frame;
        std::chrono::steady_clock::time_point frame_timestamp;
        ExtraInputInfo extra_info;   // StrictPose + 底盘 xyz（视频/相机/交互模式按规则填充）
    } initial;

    // ==================== 阶段1：预处理 ====================
    struct Stage1Data {
        cv::Mat frame;  // 预处理后图像（power_rune.inference.resolution 指定尺寸）
    } stage1;

    // ==================== 阶段2：推理 ====================
    struct Stage2Data {
        std::shared_ptr<ov::Tensor> output_tensor;  // 该图所在 batch 的输出张量
        int batch_index = 0;                         // 该图在此 batch 中的索引
    } stage2;

    // ==================== 阶段3：后处理 ====================
    struct Stage3Data {
        std::vector<PoseDetection> detections;
    } stage3;

    // ==================== 阶段4：位姿解算 + 坐标转换 ====================
    struct Stage4Data {
        CombinedPoseResult combined_pose;
        bool pose_valid = false;
        cv::Vec3f pr_world_posi   = cv::Vec3f(0, 0, 0);
        cv::Vec3f pr_world_euler  = cv::Vec3f(0, 0, 0);
        cv::Mat   pr_world_rot_mat;
        std::vector<int> rotation_counts;
    } stage4;

    // ==================== 阶段5：滤波 + 预测 ====================
    struct Stage5Data {
        float filtered_omega  = 0.0f;
        int   jump_a          = 0;
        bool  flip            = false;
        cv::Vec3f filtered_pos   = cv::Vec3f(0, 0, 0);
        cv::Mat   filtered_R;
        float filtered_euler_roll = 0.0f;
        float angular_velocity    = 0.0f;
        bool  fit_valid    = false;
        RollPredictor::BigParams   big_params;
        RollPredictor::SmallParams small_params;
        int   direction    = 1;
        std::string fit_method;   // "big" 或 "small"
        float correction_bias = 0.0f;
        std::vector<std::pair<float, float>> fitted_curve;
        std::vector<std::pair<float, float>> raw_points;
        std::vector<int> filtered_rotation_counts;
        // 位姿预测函数快照（RollPredictor::capturePredictor）
        std::unique_ptr<std::function<std::pair<cv::Vec3f, cv::Mat>(float)>> predictor_lambda;
        // 靶点预测函数快照：std::vector<cv::Point3f>(double dt)（world 系；
        // 已为 AimPredictor 统一签名，无需外部包装）
        std::unique_ptr<std::function<std::vector<cv::Point3f>(double)>> target_predictor;
    } stage5;
};

/**
 * @brief 能量机关感知流水线（事件驱动调度器版本，实现 IPipeline）
 *
 * 五阶段（各有独立工作线程，由 PipelineStage 模板管理）：
 *   1. 预处理 / 2. 推理 / 3. 后处理 / 4. 联合 PnP 位姿解算 + world 转换 /
 *   5. YAxisFilter 滤波 + RollPredictor 拟合预测（含靶点预测函数快照）
 *
 * 可视化与预测目标输出均为输出模式（common/Output/）职责。
 */
class PowerRunePipeline : public IPipeline {
public:
    static constexpr int NUM_STAGES = 5;
    static constexpr int NUM_QUEUES = NUM_STAGES + 1;  // 6 个缓冲队列
    static constexpr int MAX_PREPROCESS_BATCH = 4;
    static constexpr int MAX_INFERENCE_BATCH = 4;
    static constexpr int MAX_POSTPROCESS_BATCH = 4;

    using DataDeque = std::deque<std::unique_ptr<PowerRunePipelineData>>;

    /**
     * @param queue_max_sizes  各缓冲队列最大长度数组 [input, inter0..inter3, output]
     * @param max_delay_seconds 提取帧的最大延迟秒数（由 config common.max_delay_seconds 提供）
     * @param camera           相机参数（内参/畸变/分辨率，由输入模式选择后传入）
     */
    PowerRunePipeline(const std::array<int, NUM_QUEUES>& queue_max_sizes,
                      float max_delay_seconds,
                      const RobotConfig::CameraParams& camera);
    ~PowerRunePipeline();

    /// 构造本流水线所用的推理器（模型编译 + 预热；由独立推理进程
    /// power_rune_infer_process 的 main 调用，见 InferShmServer）
    static std::unique_ptr<YoloPose::YoloPoseInfer> createInfer();

    // ---- IPipeline ----
    PipelineMode mode() const override { return PipelineMode::POWER_RUNE; }
    std::string name() const override { return "PowerRune"; }

    /// 输入帧：队列满时抛弃新帧。
    /// @return true 成功加入输入缓冲队列；false 队列满被抛弃（未进入流水线）
    bool addFrame(cv::Mat frame,
                  const std::chrono::steady_clock::time_point& frame_timestamp,
                  const ExtraInputInfo& extra_info) override;

    PipelineResult tryPopFrame(const std::chrono::steady_clock::time_point& timestamp) override;

    /// 清空所有缓冲队列并重置滤波/预测状态
    void clear() override;

private:
    using StageType = PipelineStage<DataDeque>;

    // ==================== 配置 ====================
    std::array<int, NUM_QUEUES> queue_max_sizes_;
    float max_delay_seconds_;
    float conf_threshold_ = 0.5f;

    // ==================== 缓冲队列 ====================
    DataDeque input_queue_;
    std::mutex input_mtx_;
    std::condition_variable input_cv_;

    DataDeque inter_queues_[NUM_STAGES - 1];   // 仅调度器访问

    DataDeque output_queue_;
    std::mutex output_mtx_;

    std::atomic<int> queue_sizes_[NUM_QUEUES];

    // ==================== 阶段上下文 ====================

    /// 阶段1上下文：预处理
    struct Stage1Ctx {
        YoloPose::YoloPosePreprocessor preprocessor;
        explicit Stage1Ctx(int input_width, int input_height)
            : preprocessor(input_width, input_height) {}
    } s1_;

    /// 阶段2上下文：推理（推理器在独立进程 power_rune_infer_process 中，
    /// 本阶段经 InferShmClient 通信调用；推理进程未启动时此处阻塞等待）
    struct Stage2Ctx {
        std::unique_ptr<Infer::InferShmClient> client;
    } s2_;

    /// 阶段3上下文：后处理
    struct Stage3Ctx {
        std::unique_ptr<YoloPose::YoloPosePostprocessor> postprocessor;
    } s3_;

    /// 阶段4上下文：位姿解算 + 坐标转换（独立变换树）
    struct Stage4Ctx {
        std::shared_ptr<RobotTfTree> tree;
        std::shared_ptr<CameraProjection> camera_proj;
        PowerRunePoseSolver pose_solver;
        explicit Stage4Ctx(const RobotConfig::CameraParams& camera);
    } s4_;

    /// 阶段5上下文：滤波 + 预测
    struct Stage5Ctx {
        YAxisFilter   y_axis_filter;
        RollPredictor roll_predictor;
        std::chrono::steady_clock::time_point last_valid_timestamp;
        std::chrono::steady_clock::time_point last_frame_timestamp;  // 用于计算帧间 dt，每帧更新
    } s5_;

    // ==================== PipelineStage 实例 ====================
    StageType stage1_;
    StageType stage2_;
    StageType stage3_;
    StageType stage4_;
    StageType stage5_;

    // ==================== 事件驱动调度器 ====================
    std::thread scheduler_thread_;
    std::atomic<bool> scheduler_exit_{false};
    std::mutex scheduler_mtx_;
    std::condition_variable scheduler_cv_;
    bool scheduler_should_check_{false};

    // ==================== 各阶段处理函数 ====================
    void processStage1(DataDeque& data);
    void processStage2(DataDeque& data);
    void processStage3(DataDeque& data);
    void processStage4(DataDeque& data);
    void processStage5(DataDeque& data);

    void schedulerLoop();
    void wakeScheduler();
    void tryAdvanceStages();

    /// 从 PipelineData 填充 PowerRunePerception
    static void fillPerception(PowerRunePipelineData* d, PowerRunePerception& out);
};

#endif // POWER_RUNE_PIPELINE_H
