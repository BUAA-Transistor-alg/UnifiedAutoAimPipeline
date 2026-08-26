// VisualizeOutput.h — 可视化输出模式
//
// 按当前流水线模式（Outpost / PowerRune）渲染对应画面：
//  - 每帧从 PipelineResult.extra_info 同步自己的 RobotTfTree（world→cam 投影）；
//  - Outpost 用 OutpostVisualizer，PowerRune 用 PowerRuneVisualizer；
//  - 预测瞄准点取自 SequencePredictor 瞄准点序列的第一个值（main 每帧统一预测，
//    两种流水线模式一致绘制，任何输出状态下都可用）。
#ifndef VISUALIZE_OUTPUT_H
#define VISUALIZE_OUTPUT_H

#include "common/Output/IOutputMode.h"
#include "common/Ballistic/SequencePredictor.h"
#include "Outpost/OutpostVisualizer.h"
#include "PowerRune/PowerRuneVisualizer.h"
#include "common/TransformTree/RobotTfTree.h"
#include "common/FrameRateCounter.h"
#include "common/IPipeline.h"

#include <opencv2/opencv.hpp>
#include <atomic>
#include <memory>
#include <mutex>

class VisualizeOutput : public IOutputMode {
public:
    /// @param camera_proj 相机投影（由输入模式选择的相机参数构建）
    /// @param aim         预测瞄准点通用类（读取瞄准点序列第一个值绘制）
    explicit VisualizeOutput(std::shared_ptr<CameraProjection> camera_proj,
                             SequencePredictor& aim);

    /// 切换当前渲染的流水线模式（相机投影与输入模式绑定，不随流水线切换）
    void setMode(PipelineMode mode) { mode_.store(mode, std::memory_order_relaxed); }

    void update(const PipelineResult& result, RobotController* rc) override;

    OutputMode type() const override { return OutputMode::VISUALIZE; }
    std::string getName() const override { return "Visualize"; }

    /// 最新渲染画面（线程安全：可视化线程写入，主线程读取；
    /// 返回浅拷贝，共享像素数据，引用计数原子安全）
    cv::Mat display() const;

private:
    void syncTree(const ExtraInputInfo& info);
    // 渲染当前流水线模式的画面到 render_buf_（仅可视化线程访问；create+copyTo
    // 复用缓冲，避免每帧 clone 的分配/释放。像素拷贝仍需保留：result.frame 另被
    // 主线程保存为原始画面，不可原地绘制）
    void renderOutpost(const PipelineResult& result, RobotController* rc);
    void renderPowerRune(const PipelineResult& result, RobotController* rc);

    // 当前渲染模式（主线程 setMode 写 / 可视化线程 update 读，需原子）
    std::atomic<PipelineMode> mode_{PipelineMode::OUTPOST};
    RobotTfTree tree_;
    std::shared_ptr<CameraProjection> camera_proj_;
    OutpostVisualizer  outpost_vis_;
    PowerRuneVisualizer power_rune_vis_;
    FrameRateCounter fps_;
    cv::Mat render_buf_;              // 渲染目标缓冲（可视化线程独占）
    cv::Mat display_;
    mutable std::mutex display_mtx_;   // 保护 display_（可视化线程写 / 主线程读）
    SequencePredictor& aim_;
};

#endif // VISUALIZE_OUTPUT_H
