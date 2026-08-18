// OutpostPipeline.h — 前哨站感知流水线（5 阶段，统一 IPipeline 接口）
//
// 五阶段（各有独立工作线程，由 PipelineStage 模板管理）：
//   1. 预处理          — OutpostPreprocessor 并行 resize 640×640
//   2. 推理            — OutpostInfer 动态 batch 1..N（编译 + 预热，公共 InferCore）
//   3. 后处理          — OutpostPostprocessor 并行解码 + NMS
//   4. PnP + 坐标转换  — 每物体 solvePnP + world 转换 + 重投影；初始化 PnP（面 0）
//   5. ESEKF           — 误差状态卡尔曼滤波（初始化 / update / predict / 观测超时重置）
//
// 弹道解算、控制序列生成与可视化均已移出流水线，由输出模式（common/Output/）
// 负责；本流水线只输出感知结果（PipelineResult::outpost）。
//
// 时间戳在输入侧（IInputMode）取帧时打上，并附上同时刻的 ExtraInputInfo
// （StrictPose + 底盘 xyz）；每阶段用自己的 RobotTfTree，由 ExtraInputInfo 同步。
#ifndef OUTPOST_PIPELINE_H
#define OUTPOST_PIPELINE_H

#include "PipelineStage.h"
#include "OpenvinoInfer.h"
#include "ESEKF.h"
#include "CameraProjection.h"
#include "TransformTree/RobotTfTree.h"
#include "OutpostModel.h"
#include "IInputMode.h"
#include "IPipeline.h"

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
 * @brief 流水线数据：贯穿所有处理阶段，各阶段只读之前阶段的数据，写入自己阶段的数据。
 */
struct OutpostPipelineData {
    // ==================== 初始数据（处理前外部传入） ====================
    struct InitialData {
        cv::Mat frame;
        std::chrono::steady_clock::time_point frame_timestamp;  // 取帧时打上
        ExtraInputInfo extra_info;                              // 取帧同时刻（StrictPose + 底盘 xyz）
    } initial;

    // ==================== 阶段1：预处理 ====================
    struct Stage1Data {
        cv::Mat frame;  // 预处理后图像 (640×640)
    } stage1;

    // ==================== 阶段2：推理 ====================
    struct Stage2Data {
        std::shared_ptr<ov::Tensor> output_tensor;  // 该图所在 batch 的输出张量
        int batch_index = 0;                         // 该图在此 batch 中的索引
    } stage2;

    // ==================== 阶段3：后处理 ====================
    struct Stage3Data {
        std::vector<OutpostDetect::Object> objects;  // 检测结果（已缩放至原图坐标）
    } stage3;

    // ==================== 阶段4：PnP + 坐标转换 ====================
    struct Stage4Data {
        std::vector<cv::Vec3f> world_positions;
        std::vector<cv::Vec3f> world_eulers;
        std::vector<std::vector<cv::Point2f>> reprojected_points;
        std::vector<cv::Point2f> first_image_points;   // 第 0 物体 4 个 2D 关键点（原图坐标）
        bool init_pnp_ok = false;
        cv::Vec3f init_pos = cv::Vec3f(0, 0, 0);
        cv::Mat   init_R;   // 3x3 CV_32F（world 系）
    } stage4;

    // ==================== 阶段5：ESEKF ====================
    struct Stage5Data {
        bool esekf_initialized = false;
        std::vector<cv::Point3f> ekf_world_points;   // 12 个世界关键点（getWorldPoints）
        cv::Vec3d ekf_pos = cv::Vec3d(0, 0, 0);
        cv::Mat   ekf_R64;                            // CV_64F
        std::vector<cv::Point3f> pred_center_points; // 预测目标中心关键点 t+0（world）
        std::unique_ptr<std::function<std::vector<cv::Point3f>(double)>> predictor;  // 快照
    } stage5;
};

/**
 * @brief 前哨站感知流水线（事件驱动调度器版本，实现 IPipeline）
 */
class OutpostPipeline : public IPipeline {
public:
    static constexpr int NUM_STAGES = 5;
    static constexpr int NUM_QUEUES = NUM_STAGES + 1;  // 6 个缓冲队列
    static constexpr int MAX_PREPROCESS_BATCH = 4;
    static constexpr int MAX_INFERENCE_BATCH  = 4;
    static constexpr int MAX_POSTPROCESS_BATCH = 4;

    using DataDeque = std::deque<std::unique_ptr<OutpostPipelineData>>;

    /**
     * @param queue_max_sizes  各缓冲队列最大长度数组 [input, inter0..inter3, output]
     * @param max_delay_seconds 提取帧的最大延迟秒数，默认 0.05s
     */
    OutpostPipeline(const std::array<int, NUM_QUEUES>& queue_max_sizes = {10, 10, 10, 10, 10, 10},
                    float max_delay_seconds = 0.2f);
    ~OutpostPipeline();

    // ---- IPipeline ----
    PipelineMode mode() const override { return PipelineMode::OUTPOST; }
    std::string name() const override { return "Outpost"; }

    void addFrame(cv::Mat frame,
                  const std::chrono::steady_clock::time_point& frame_timestamp,
                  const ExtraInputInfo& extra_info) override;

    PipelineResult tryPopFrame(const std::chrono::steady_clock::time_point& timestamp) override;

    /// 清空所有缓冲队列并重置 ESEKF / 观测计时状态
    void clear() override;

private:
    using StageType = PipelineStage<DataDeque>;

    // ==================== 配置 ====================
    std::array<int, NUM_QUEUES> queue_max_sizes_;
    float max_delay_seconds_;
    float conf_threshold_ = 0.65f;
    float nms_threshold_   = 0.45f;

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
        OutpostDetect::OutpostPreprocessor preprocessor;
    } s1_;

    /// 阶段2上下文：推理
    struct Stage2Ctx {
        std::unique_ptr<OutpostDetect::OutpostInfer> infer_p;
    } s2_;

    /// 阶段3上下文：后处理
    struct Stage3Ctx {
        std::unique_ptr<OutpostDetect::OutpostPostprocessor> postprocessor;
    } s3_;

    /// 阶段4上下文：PnP + 坐标转换（独立变换树）
    struct Stage4Ctx {
        RobotTfTree tree;
        std::shared_ptr<CameraProjection> camera_proj;
        Stage4Ctx();
    } s4_;

    /// 阶段5上下文：ESEKF（独立变换树 + 相机投影 + 跨帧滤波状态）
    struct Stage5Ctx {
        std::shared_ptr<RobotTfTree> tree;
        std::shared_ptr<CameraProjection> camera_proj;
        std::unique_ptr<ESEKF> esekf;
        bool esekf_initialized = false;
        std::chrono::steady_clock::time_point last_observation_time;
        bool has_observation_time = false;
        Stage5Ctx();
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

    /// 从 PipelineData 填充 OutpostPerception
    static void fillPerception(OutpostPipelineData* d, OutpostPerception& out);
};

#endif // OUTPOST_PIPELINE_H
