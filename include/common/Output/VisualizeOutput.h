// VisualizeOutput.h — 可视化输出模式
//
// 按当前流水线模式（Outpost / PowerRune）渲染对应画面：
//  - 每帧从 PipelineResult.extra_info 同步自己的 RobotTfTree（world→cam 投影）；
//  - Outpost 用 OutpostVisualizer，PowerRune 用 PowerRuneVisualizer；
//  - 预测瞄准点取自 AimPredictor 瞄准点序列的第一个值（main 每帧统一预测，
//    两种流水线模式一致绘制，任何输出状态下都可用）。
#ifndef VISUALIZE_OUTPUT_H
#define VISUALIZE_OUTPUT_H

#include "common/Output/IOutputMode.h"
#include "common/Ballistic/AimPredictor.h"
#include "Outpost/OutpostVisualizer.h"
#include "PowerRune/PowerRuneVisualizer.h"
#include "common/TransformTree/RobotTfTree.h"
#include "common/FrameRateCounter.h"
#include "common/IPipeline.h"

#include <opencv2/opencv.hpp>
#include <memory>

class VisualizeOutput : public IOutputMode {
public:
    /// @param camera_proj 相机投影（由输入模式选择的相机参数构建）
    /// @param aim         预测瞄准点通用类（读取瞄准点序列第一个值绘制）
    explicit VisualizeOutput(std::shared_ptr<CameraProjection> camera_proj,
                             AimPredictor& aim);

    /// 切换当前渲染的流水线模式（相机投影与输入模式绑定，不随流水线切换）
    void setMode(PipelineMode mode);

    void update(const PipelineResult& result, RobotController* rc) override;

    OutputMode type() const override { return OutputMode::VISUALIZE; }
    std::string getName() const override { return "Visualize"; }

    /// 最新渲染画面（主线程 imshow 用）
    const cv::Mat& display() const { return display_; }

private:
    void syncTree(const ExtraInputInfo& info);
    void renderOutpost(const PipelineResult& result, RobotController* rc);
    void renderPowerRune(const PipelineResult& result, RobotController* rc);

    PipelineMode mode_ = PipelineMode::OUTPOST;
    RobotTfTree tree_;
    std::shared_ptr<CameraProjection> camera_proj_;
    OutpostVisualizer  outpost_vis_;
    PowerRuneVisualizer power_rune_vis_;
    FrameRateCounter fps_;
    cv::Mat display_;
    AimPredictor& aim_;
};

#endif // VISUALIZE_OUTPUT_H
