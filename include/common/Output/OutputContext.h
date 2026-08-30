// OutputContext.h — 输出上下文（流水线阶段间传递的附加信息载体）
//
// 每帧由处理线程（process_thread）产生，随 BallisticRequest / GimbalRequest /
// VisualizeRequest 沿"弹道 → 云台 → 可视化"级联逐级转发，最终传给各输出模式
// （IOutputMode::update 的 ctx 参数）。
// 内容由各级写入/消费：
//  - fire_out     云台输出阶段（GimbalOutput::update）计算出的 fire 序列，
//                 可视化阶段取首元素控制井形叉丝颜色；
//  - gimbal_enabled 当前是否开启 gimbal 输出模式（main 云台线程转发时写入）。
#ifndef OUTPUT_CONTEXT_H
#define OUTPUT_CONTEXT_H

#include <vector>

struct OutputContext {
    std::vector<bool> fire_out;   // GimbalOutput::update 计算出的 fire 序列（首元素供可视化绘制）
    bool gimbal_enabled = false;  // 是否开启 gimbal 输出模式
};

#endif // OUTPUT_CONTEXT_H
