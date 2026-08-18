// GimbalOutput.h — 云台控制输出模式
//
// 弹道解算 + 控制序列生成（GimbalSolver + PredictedBallisticSolver +
// InputController）作为输出模式：每帧直接读取 RobotController::getState()
// （串口/MPC 状态不经流水线），结合流水线输出的预测器快照生成并发送
// yaw/pitch/fire 序列。
//  - Outpost 流水线：使用 ESEKF 预测器，目标选择 NEAREST（muzzle 距离最近）；
//  - PowerRune 流水线：使用靶点预测函数（target_predictor），目标选择 LOWEST_Z
//    （world z 最低）；
//  - 预测器不可用时进入保持模式（自瞄关闭）。
#ifndef GIMBAL_OUTPUT_H
#define GIMBAL_OUTPUT_H

#include "Output/IOutputMode.h"
#include "InputController.h"
#include "Ballistic/GimbalSolver.h"

#include <memory>

class GimbalOutput : public IOutputMode {
public:
    explicit GimbalOutput(RobotController& rc);

    /// @param rc 参数为 nullptr（本模式构造时已持有 RobotController 引用）
    void update(const PipelineResult& result, RobotController* rc) override;

    OutputMode type() const override { return OutputMode::GIMBAL; }
    std::string getName() const override { return "Gimbal"; }

    /// 本帧生成的序列输入（供显示/调试）
    const InputController::LastOutput& lastOutput() const { return input_controller_->lastOutput(); }

private:
    RobotController& rc_;
    std::shared_ptr<GimbalSolver> gimbal_;
    std::unique_ptr<InputController> input_controller_;
};

#endif // GIMBAL_OUTPUT_H
