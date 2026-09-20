// PredictedPointSelector.h — PowerRune 预测点（靶点）选取决策器。
//
// 用途：SequencePredictor 在目标来源为 PowerRune 时用它替换“简单取高度最低点”的
// 选取策略。靶点预测恒输出全部 kBladeCount(5) 个靶点（列表下标 = 旋转计数），
// 本帧不存在的靶点由 Predictor::masked_indices 屏蔽。
//
// ── 每点状态机（初始全部 UNOBSERVED）──
//   UNOBSERVED   未被观测：
//       未被 mask           → ESTABLISHING（计时起点 = 本次 timestamp）
//   ESTABLISHING 观测建立中：
//       mask                → UNOBSERVED（计时清零）
//       连续 establish_duration_s 未被 mask → OBSERVED
//   OBSERVED     已观测到：
//       mask                → TEMPORARILY_LOST（计时起点 = 本次 timestamp）
//   TEMPORARILY_LOST 临时丢失：
//       未被 mask           → OBSERVED（此前已确认，直接恢复，不再重新计时）
//       连续 lost_timeout_s 被 mask → UNOBSERVED
//   特殊规则（每次状态更新、逐点转移完成后判定）：统计**未被 mask** 且处于
//   OBSERVED 的点数，达到或超过 2 时，把所有被 mask 且处于 TEMPORARILY_LOST 的
//   点立即置为 UNOBSERVED。两个计时状态被任何其它转移打断即计时清零；
//   计时一律使用调用方传入的 timestamp（内部不读系统时间）。
//
// ── 选取（每次 select() 调用；每帧一次，见 SequencePredictor::predictImpl）──
//   候选 = 状态为 OBSERVED 或 TEMPORARILY_LOST 的点；
//   上一次选中的目标仍在候选中且本步有可用解算结果 → 继续选它（粘滞）；
//   否则在候选中取**预测高度 z 最低**者；无可用候选 → 返回 -1（并清空粘滞目标）。
//
// ── 解算前预处理 ──
//   preUpdateCandidateIndices() 给出“本次状态更新前”处于 OBSERVED / TEMPORARILY_LOST
//   的点索引：这些点即使本帧被 mask 也应参与弹道解算——已观测点可能在本步被 mask
//   转为临时丢失并被粘滞选中；临时丢失点也可能被继续选中（否则它们在本步的解算
//   序列里只是 masked 占位符，选到后没有可用结果）。调用方据此从解算 mask 中解除
//   对应屏蔽（见 SequencePredictor::predictImpl 的 solve_masked_indices）。
#ifndef PREDICTED_POINT_SELECTOR_H
#define PREDICTED_POINT_SELECTOR_H

#include <chrono>
#include <vector>

#include "common/Ballistic/PredictedBallisticSolver.h"

class PredictedPointSelector {
public:
    using TimePoint = std::chrono::steady_clock::time_point;

    // 单点状态机状态（见文件头说明）
    enum class State {
        UNOBSERVED = 0,    // 未被观测
        ESTABLISHING,      // 观测建立中（计时中：establish_duration_s）
        OBSERVED,          // 已观测到
        TEMPORARILY_LOST,  // 临时丢失（计时中：lost_timeout_s）
    };

    /// 阈值从 RobotConfig 的 power_rune.target_selection 段读取
    /// （establish_duration_s = a，lost_timeout_s = b）
    PredictedPointSelector();

    /// 解算前查询（不改变任何状态）：返回当前（本次状态更新前）处于 OBSERVED 或
    /// TEMPORARILY_LOST 的点索引（升序）。这些点应解除解算屏蔽（见文件头说明）。
    /// 状态表尚未建立（首次调用前）时返回空。
    std::vector<int> preUpdateCandidateIndices() const;

    /// 解算后调用：用本步 masked_indices 更新状态机，并选出本步使用的目标。
    /// @param masked_indices 本步（本帧）被屏蔽的靶点索引（预测函数列表下标）
    /// @param solved         本步解算序列（下标 = 目标索引；被 mask 的点为占位符，
    ///                       见 PredictedBallisticSolver::Result::masked）
    /// @param timestamp      本步时间戳（计时基准；不用系统当前时间）
    /// @return 选中的目标索引（预测函数列表下标）；无可用候选返回 -1。
    int select(const std::vector<int>& masked_indices,
               const std::vector<PredictedBallisticSolver::Result>& solved,
               TimePoint timestamp);

    /// 状态整体重置：全部点回到 UNOBSERVED、粘滞目标清空（状态表长度保留）。
    /// 由 SequencePredictor 在“不使用本决策器”（来源非 PowerRune）与“无目标”
    /// （predictor 无效 → invalidate）时调用。
    void reset();

    /// 单点当前状态（诊断/测试用）
    State stateOf(int index) const;

    /// 上一次选中的目标索引（-1 = 无；诊断/测试用）
    int lastSelectedIndex() const { return last_selected_index_; }

private:
    // 单点状态记录：state_since 仅在 ESTABLISHING / TEMPORARILY_LOST 两个计时状态有效
    struct PointRecord {
        State     state = State::UNOBSERVED;
        TimePoint state_since{};   // 进入当前计时状态的时刻
    };

    /// 状态表长度与解算序列不一致（首次调用 / 靶点数变化）时整体重建为 UNOBSERVED
    void ensureSize(size_t size);

    /// 逐点转移 + 特殊规则（见文件头说明）
    void updateStates(const std::vector<int>& masked_indices, TimePoint timestamp);

    static bool isCandidate(State s) {
        return s == State::OBSERVED || s == State::TEMPORARILY_LOST;
    }
    static bool isMaskedIndex(const std::vector<int>& masked_indices, int index);

    /// 取得 index 在本步解算序列中的可用结果（存在且 success && !masked），
    /// 否则返回 nullptr
    static const PredictedBallisticSolver::Result* usableResult(
        const std::vector<PredictedBallisticSolver::Result>& solved, int index);

    std::vector<PointRecord> states_;
    int    last_selected_index_ = -1;
    double establish_duration_s_ = 0.1;   // a：ESTABLISHING → OBSERVED 的连续时长（秒）
    double lost_timeout_s_      = 0.5;    // b：TEMPORARILY_LOST → UNOBSERVED 的连续时长（秒）
};

#endif // PREDICTED_POINT_SELECTOR_H
