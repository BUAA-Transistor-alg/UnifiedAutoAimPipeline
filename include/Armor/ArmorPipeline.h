// ArmorPipeline.h — 装甲板感知流水线（5 阶段，统一 IPipeline 接口）
//
// 五阶段（各有独立工作线程，由 PipelineStage 模板管理）：
//   1. 预处理          — ArmorPreprocessor 并行 resize（分辨率取 config armor.inference.resolution）
//   2. 推理            — InferShmClient 经共享内存调用独立推理进程（动态 batch）
//   3. 后处理          — ArmorPostprocessor 并行解码 + NMS（读取 PipelineData 自持输出缓冲）
//   4. PnP + 坐标转换  — 每物体按装甲板种类（类别 label 0~8）分类：solvePnP + world
//                        转换 + 重投影；结果按类别打包存入 vector<CategoryData>
//                        （索引 = 类别，含该类别观测关键点与 world_pos/world_euler）
//   5. OutpostESEKF           — 误差状态卡尔曼滤波（初始化 / update / predict / 观测超时重置）；
//                               初始化传入 label 6 首个装甲板的图像关键点（位姿在 init
//                               内部解算），观测仅使用 label 6 类别的关键点
//
// 弹道解算、控制序列生成与可视化均已移出流水线，由输出模式（common/Output/）
// 负责；本流水线只输出感知结果（PipelineResult::armor）。
//
// 时间戳在输入侧（IInputMode）取帧时打上，并附上同时刻的 ExtraInputInfo
// （StrictPose + 底盘 xyz）；每阶段用自己的 RobotTfTree，由 ExtraInputInfo 同步。
#ifndef ARMOR_PIPELINE_H
#define ARMOR_PIPELINE_H

#include "common/PipelineStage.h"
#include "Armor/ArmorInfer.h"
#include "Armor/OutpostESEKF.h"
#include "common/CameraProjection.h"
#include "common/TransformTree/RobotTfTree.h"
#include "Armor/ArmorModel.h"
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
 * @brief 流水线数据：贯穿所有处理阶段，各阶段只读之前阶段的数据，写入自己阶段的数据。
 */
struct ArmorPipelineData {
    // ==================== 初始数据（处理前外部传入） ====================
    struct InitialData {
        cv::Mat frame;
        std::chrono::steady_clock::time_point frame_timestamp;  // 取帧时打上
        ExtraInputInfo extra_info;                              // 取帧同时刻（StrictPose + 底盘 xyz）
    } initial;

    // ==================== 阶段1：预处理 ====================
    struct Stage1Data {
        cv::Mat frame;  // 预处理后图像（armor.inference.resolution 指定尺寸）
    } stage1;

    // ==================== 阶段2：推理 ====================
    struct Stage2Data {
        // 推理输出张量（客户端直接 memcpy 填充，data 行优先 f32，
        // rows/cols 为张量 shape[1]/shape[2]，如 armor 的 [num_anchors, 22]）
        Infer::OutputBuffer output;
    } stage2;

    // ==================== 阶段3：后处理 ====================
    struct Stage3Data {
        std::vector<ArmorDetect::Object> objects;  // 检测结果（已缩放至原图坐标）
    } stage3;

    // ==================== 阶段4：PnP + 坐标转换 ====================
    struct Stage4Data {
        // 单个装甲板种类（类别 label）的分类数据
        struct CategoryData {
            std::vector<std::vector<cv::Point2f>> all_image_points;  // 该类别所有物体的 4 个 2D 关键点（原图坐标）
            std::vector<cv::Vec3f> world_pos;    // 与 all_image_points 一一对应：该类别各物体的世界坐标
            std::vector<cv::Vec3f> world_euler;  // 与 all_image_points 一一对应：该类别各物体的世界欧拉角
        };
        // 按装甲板种类分类的数据：外层 vector 索引 = 类别 label 0~8，每帧重置为
        // NUM_CLASSES 个槽位；槽内为该类别自己的观测关键点与世界位姿
        std::vector<CategoryData> categories;

        // 每个物体的世界坐标/欧拉角/重投影（与 objects 一一对应，供输出与可视化）
        std::vector<cv::Vec3f> world_positions;
        std::vector<cv::Vec3f> world_eulers;
        std::vector<std::vector<cv::Point2f>> reprojected_points;
    } stage4;

    // ==================== 阶段5：OutpostESEKF ====================
    struct Stage5Data {
        bool esekf_initialized = false;
        std::vector<cv::Point3f> ekf_world_points;   // 12 个世界关键点（getWorldPoints）
        cv::Vec3d ekf_pos = cv::Vec3d(0, 0, 0);
        cv::Mat   ekf_R64;                            // CV_64F
        std::vector<cv::Point3f> pred_center_points; // 预测目标中心关键点 t+0（world）
        std::unique_ptr<std::function<std::vector<cv::Point3f>(double)>> predictor;  // 快照
        std::chrono::steady_clock::time_point predictor_timestamp;  // 快照对应帧的时间戳（dt 零点）
    } stage5;
};

/**
 * @brief 装甲板感知流水线（事件驱动调度器版本，实现 IPipeline）
 */
class ArmorPipeline : public IPipeline {
public:
    static constexpr int NUM_STAGES = 5;
    static constexpr int NUM_QUEUES = NUM_STAGES + 1;  // 6 个缓冲队列
    // 各阶段批量不再在此硬编码，改由 config 提供：
    //   armor.pipeline.preprocess_batch   阶段1 预处理最大批量
    //   armor.pipeline.inference_batch    阶段2 推理最大批量（≤ inference.max_batch）
    //   armor.pipeline.postprocess_batch  阶段3 后处理最大批量
    // 阶段4/5（PnP / OutpostESEKF）为单帧处理，批量固定 1。

    using DataDeque = std::deque<std::unique_ptr<ArmorPipelineData>>;

    /**
     * @param queue_max_sizes  各缓冲队列最大长度数组 [input, inter0..inter3, output]
     *                         （取自 config armor.pipeline.queue_max_sizes）
     * @param min_delay_seconds 提取帧的最小延迟秒数（输出队列帧龄须达到该值后
     *                          tryPopFrame 才返回；由 config common.min_delay_seconds 提供）
     * @param camera           相机参数（内参/畸变/分辨率，由输入模式选择后传入）
     */
    ArmorPipeline(const std::array<int, NUM_QUEUES>& queue_max_sizes,
                    float min_delay_seconds,
                    const RobotConfig::CameraParams& camera);
    ~ArmorPipeline();

    /// 构造本流水线所用的推理器（模型编译 + 预热；由独立推理进程
    /// armor_infer_process 的 main 调用，见 InferShmServer）
    static std::unique_ptr<ArmorDetect::ArmorInfer> createInfer();

    /// 推理进程（重新）启动后重连 InferShmClient 信号量。
    /// lazy 模式（common.infer_process_lazy=true）下推理进程按需启停，服务端启动时
    /// 会 sem_unlink 重建信号量，本客户端先于服务端附加则需重连（见 InferShmClient::reconnect）。
    void reconnectInferClient() { s2_.client->reconnect(); }

    // ---- IPipeline ----
    PipelineMode mode() const override { return PipelineMode::ARMOR; }
    std::string name() const override { return "Armor"; }

    /// 输入帧：队列满时抛弃新帧。
    /// @return true 成功加入输入缓冲队列；false 队列满被抛弃（未进入流水线）
    bool addFrame(cv::Mat frame,
                  const std::chrono::steady_clock::time_point& frame_timestamp,
                  const ExtraInputInfo& extra_info) override;

    PipelineResult tryPopFrame(const std::chrono::steady_clock::time_point& timestamp) override;

    /// 清空所有缓冲队列并重置 OutpostESEKF / 观测计时状态
    void clear() override;

private:
    using StageType = PipelineStage<DataDeque>;

    // ==================== 配置 ====================
    std::array<int, NUM_QUEUES> queue_max_sizes_;
    float min_delay_seconds_;
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
        ArmorDetect::ArmorPreprocessor preprocessor;
        explicit Stage1Ctx(int input_width, int input_height)
            : preprocessor(input_width, input_height) {}
    } s1_;

    /// 阶段2上下文：推理（推理器在独立进程 armor_infer_process 中，
    /// 本阶段经 InferShmClient 通信调用；推理进程未启动时此处阻塞等待）
    struct Stage2Ctx {
        std::unique_ptr<Infer::InferShmClient> client;
    } s2_;

    /// 阶段3上下文：后处理
    struct Stage3Ctx {
        std::unique_ptr<ArmorDetect::ArmorPostprocessor> postprocessor;
    } s3_;

    /// 阶段4上下文：PnP + 坐标转换（独立变换树）
    struct Stage4Ctx {
        RobotTfTree tree;
        std::shared_ptr<CameraProjection> camera_proj;
        explicit Stage4Ctx(const RobotConfig::CameraParams& camera);
    } s4_;

    /// 阶段5上下文：OutpostESEKF（独立变换树 + 相机投影 + 跨帧滤波状态）
    struct Stage5Ctx {
        std::shared_ptr<RobotTfTree> tree;
        std::shared_ptr<CameraProjection> camera_proj;
        std::unique_ptr<OutpostESEKF> esekf;
        bool esekf_initialized = false;
        std::chrono::steady_clock::time_point last_observation_time;
        bool has_observation_time = false;
        explicit Stage5Ctx(const RobotConfig::CameraParams& camera,
                           const RobotConfig::ArmorParams::EsekfParams& esekf_params);
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
    // 调度器推进（tryAdvanceStages）与 clear() 全量清空互斥：
    // clear() 持锁期间调度器不得推进（不拉新批、不动中段队列），
    // 各阶段 worker 仍可跑完当前批，之后由 clear() 统一丢弃。
    std::mutex advance_mtx_;

    // ==================== 各阶段处理函数 ====================
    void processStage1(DataDeque& data);
    void processStage2(DataDeque& data);
    void processStage3(DataDeque& data);
    void processStage4(DataDeque& data);
    void processStage5(DataDeque& data);

    void schedulerLoop();
    void wakeScheduler();
    void tryAdvanceStages();

    /// 从 PipelineData 填充 ArmorPerception
    static void fillPerception(ArmorPipelineData* d, ArmorPerception& out);
};

#endif // ARMOR_PIPELINE_H
