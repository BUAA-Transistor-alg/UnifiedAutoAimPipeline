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
#include "common/BigSmallYaw/RobotStateForBigSmallYaw.h"
#include "common/RobotConfig.h"

struct OutputContext {
    // 当前 yaw 构型（config common.big_small_yaw.mode；两构型的信息包互不混用，
    // 下游据此决定读哪一包 / 是否绘制大小 yaw 叠加）
    YawMode yaw_mode = YawMode::SINGLE;

    // 新构型（大小 yaw）当帧控制器状态快照：**仅** mode = BIG_SMALL 时由 main 弹道线程
    // 填充（供新构型云台输出模式与可视化使用）；单 yaw 构型下保持默认值（valid = false）
    bsy::RobotState bsy_state;

    // main 弹道线程每帧写入：当帧 SequencePredictor::predict 的结果
    // （无有效预测 / 已被 invalidate 时保持默认无效 Result，输出模式进入保持模式）
    SequencePredictor::Result predict_result;

    // 大小 yaw 拆分器诊断（GimbalOutputForBigSmallYaw::update 每帧回写；
    // 单 yaw 构型恒为 valid = false）——供可视化显示“角度/参考/越限标志”
    struct BigSmallSplitDiag {
        bool   valid = false;
        int    jump_count = 0;              // 本帧被修正的点数
        int    unlimited_episodes = 0;      // 本帧无限幅运动段数（连续修正算一段）
        bool   over_limit = false;          // 本帧是否发生越软限位修正
        double theta_small_max_abs = 0.0;   // |θ_small| 最大值（rad）
        double soft_min = 0.0, soft_max = 0.0;   // 小 yaw 软限位边界（rad）
        double big_ref_front = 0.0;         // ψ_big* 序列首元素（世界方位角）
        double small_ref_front = 0.0;       // ψ_small* 序列首元素（世界方位角）
    } split_diag;

    std::vector<bool> fire_out;   // GimbalOutput::update 计算出的 fire 序列（首元素供可视化绘制）
    bool gimbal_enabled = false;  // 是否开启 gimbal 输出模式
};

#endif // OUTPUT_CONTEXT_H
