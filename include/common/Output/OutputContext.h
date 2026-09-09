// OutputContext.h — 输出上下文（流水线阶段间传递的附加信息载体）
//
// 每帧由处理线程（process_thread）产生，随 BallisticRequest / GimbalRequest /
// VisualizeRequest 沿"弹道 → 云台 → 可视化"级联逐级转发，最终传给各输出模式
// （IOutputMode::update 的 ctx 参数）。
// 内容由各级写入/消费：
//  - predict_result  main 弹道线程每帧经 SequencePredictor::predict 产生的结果
//                    （预测云台控制序列 + 瞄准点序列 + yaw 系原点；预测不可用时
//                    为默认无效结果）。GimbalOutput / VisualizeOutput 直接消费，
//                    不再各自持有 SequencePredictor 引用（瞄准点预测仍由 main
//                    每帧统一调用 SequencePredictor 完成）；
//  - fire_out        云台输出阶段（GimbalOutput::update）计算出的 fire 序列，
//                    可视化阶段取首元素控制井形叉丝颜色；
//  - gimbal_enabled  当前是否开启 gimbal 输出模式（main 云台线程转发时写入）。
#ifndef OUTPUT_CONTEXT_H
#define OUTPUT_CONTEXT_H

#include <vector>

#include "common/Ballistic/SequencePredictor.h"

struct OutputContext {
    // main 弹道线程每帧写入：当帧 SequencePredictor::predict 的结果
    // （无有效预测 / 已被 invalidate 时保持默认无效 Result，输出模式进入保持模式）
    SequencePredictor::Result predict_result;

    std::vector<bool> fire_out;   // GimbalOutput::update 计算出的 fire 序列（首元素供可视化绘制）
    bool gimbal_enabled = false;  // 是否开启 gimbal 输出模式
};

#endif // OUTPUT_CONTEXT_H
