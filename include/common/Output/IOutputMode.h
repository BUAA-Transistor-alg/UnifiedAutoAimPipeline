// IOutputMode.h — 输出模式统一接口
//
// 输出模式消费流水线的感知结果（PipelineResult，经 tryPopFrame 获取），
// 并直接读取 RobotController 串口/MPC 状态（不经流水线）。
// 弹道解算 + 控制序列生成（GimbalOutput）与可视化（VisualizeOutput）
// 都是输出模式；流水线只负责感知。
#ifndef IOUTPUT_MODE_H
#define IOUTPUT_MODE_H

#include <string>

#include "PipelineResult.h"
#include "RobotController.h"

enum class OutputMode { NONE, VISUALIZE, GIMBAL };

class IOutputMode {
public:
    virtual ~IOutputMode() = default;

    /**
     * @brief 每周期调用：处理最新一帧流水线输出。
     * @param result 流水线感知结果（valid 为 false 表示本周期没有到时帧）
     * @param rc     RobotController 指针（可能为 nullptr；串口状态应直接读取）
     */
    virtual void update(const PipelineResult& result, RobotController* rc) = 0;

    virtual OutputMode type() const = 0;
    virtual std::string getName() const = 0;
};

#endif // IOUTPUT_MODE_H
