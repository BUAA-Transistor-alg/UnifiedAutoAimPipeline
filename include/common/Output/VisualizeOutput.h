// VisualizeOutput.h — 可视化输出模式
//
// 按当前流水线模式（Outpost / PowerRune）渲染对应画面：
//  - 每帧从 PipelineResult.extra_info 同步自己的 RobotTfTree（world→cam 投影）；
//  - Outpost 用 OutpostVisualizer，PowerRune 用 PowerRuneVisualizer；
//  - 若 RobotController 可用，绘制其通信信息（否则以空状态参与）。
#ifndef VISUALIZE_OUTPUT_H
#define VISUALIZE_OUTPUT_H

#include "Output/IOutputMode.h"
#include "OutpostVisualizer.h"
#include "PowerRuneVisualizer.h"
#include "TransformTree/RobotTfTree.h"
#include "FrameRateCounter.h"
#include "IPipeline.h"

#include <opencv2/opencv.hpp>
#include <memory>

class VisualizeOutput : public IOutputMode {
public:
    VisualizeOutput();

    /// 切换当前渲染的流水线模式（同时重建对应相机投影）
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
};

#endif // VISUALIZE_OUTPUT_H
