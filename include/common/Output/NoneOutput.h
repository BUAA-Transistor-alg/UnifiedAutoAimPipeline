// NoneOutput.h — 空输出模式（不消费结果，仅用于关闭输出）
#ifndef NONE_OUTPUT_H
#define NONE_OUTPUT_H

#include "common/Output/IOutputMode.h"

class NoneOutput : public IOutputMode {
public:
    void update(const PipelineResult&, RobotController*, OutputContext&) override {}
    OutputMode type() const override { return OutputMode::NONE; }
    std::string getName() const override { return "None"; }
};

#endif // NONE_OUTPUT_H
