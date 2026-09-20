// PredictedPointSelector.cpp — PowerRune 预测点选取决策器实现（见头文件说明）
#include "PowerRune/PredictedPointSelector.h"

#include <limits>

#include "common/RobotConfig.h"

namespace {
// 两个计时状态已持续的秒数（timestamp 倒退时按 0 处理，避免负计时）
double elapsedSec(PredictedPointSelector::TimePoint since,
                  PredictedPointSelector::TimePoint now) {
    const double dt = std::chrono::duration<double>(now - since).count();
    return dt > 0.0 ? dt : 0.0;
}
} // namespace

PredictedPointSelector::PredictedPointSelector()
    : establish_duration_s_(
          RobotConfig::instance().powerRune.targetSelection.establishDurationSec),
      lost_timeout_s_(RobotConfig::instance().powerRune.targetSelection.lostTimeoutSec) {}

bool PredictedPointSelector::isMaskedIndex(const std::vector<int>& masked_indices, int index) {
    for (int m : masked_indices) {
        if (m == index) return true;
    }
    return false;
}

std::vector<int> PredictedPointSelector::preUpdateCandidateIndices() const {
    std::vector<int> out;
    out.reserve(states_.size());
    for (size_t i = 0; i < states_.size(); ++i) {
        if (isCandidate(states_[i].state)) out.push_back((int)i);
    }
    return out;
}

PredictedPointSelector::State PredictedPointSelector::stateOf(int index) const {
    if (index < 0 || index >= (int)states_.size()) return State::UNOBSERVED;
    return states_[(size_t)index].state;
}

void PredictedPointSelector::ensureSize(size_t size) {
    if (states_.size() == size) return;
    // 点数变化（首次调用 / 预测列表长度改变）：整体重建，全部从未被观测开始
    states_.assign(size, PointRecord{});
    last_selected_index_ = -1;
}

void PredictedPointSelector::updateStates(const std::vector<int>& masked_indices,
                                          TimePoint timestamp) {
    for (size_t i = 0; i < states_.size(); ++i) {
        const bool masked = isMaskedIndex(masked_indices, (int)i);
        PointRecord& rec = states_[i];
        switch (rec.state) {
            case State::UNOBSERVED:
                // 未被观测：一旦未被 mask 即进入“观测建立中”并开始计时
                if (!masked) {
                    rec.state = State::ESTABLISHING;
                    rec.state_since = timestamp;
                }
                break;

            case State::ESTABLISHING:
                if (masked) {
                    // 建立过程被打断：计时清零，回到未被观测
                    rec.state = State::UNOBSERVED;
                    rec.state_since = timestamp;
                } else if (elapsedSec(rec.state_since, timestamp) >= establish_duration_s_) {
                    rec.state = State::OBSERVED;
                    rec.state_since = timestamp;
                }
                break;

            case State::OBSERVED:
                // 已观测到：被 mask → 临时丢失（开始 b 计时）
                if (masked) {
                    rec.state = State::TEMPORARILY_LOST;
                    rec.state_since = timestamp;
                }
                break;

            case State::TEMPORARILY_LOST:
                if (!masked) {
                    // 重新出现：此前已确认过，直接恢复为已观测到（不再重新建立）
                    rec.state = State::OBSERVED;
                    rec.state_since = timestamp;
                } else if (elapsedSec(rec.state_since, timestamp) >= lost_timeout_s_) {
                    // 丢失超时：忘记该点
                    rec.state = State::UNOBSERVED;
                    rec.state_since = timestamp;
                }
                break;
        }
    }

    // ── 特殊规则：未被 mask 的已观测点达到 2 个及以上时，立即清掉被 mask 的
    //    临时丢失点（已有足够确认目标，不再为丢失点保留粘滞窗口）──
    int observed_unmasked = 0;
    for (size_t i = 0; i < states_.size(); ++i) {
        if (states_[i].state == State::OBSERVED &&
            !isMaskedIndex(masked_indices, (int)i)) {
            ++observed_unmasked;
        }
    }
    if (observed_unmasked >= 2) {
        for (size_t i = 0; i < states_.size(); ++i) {
            if (isMaskedIndex(masked_indices, (int)i) &&
                states_[i].state == State::TEMPORARILY_LOST) {
                states_[i].state = State::UNOBSERVED;
                states_[i].state_since = timestamp;
            }
        }
    }
}

const PredictedBallisticSolver::Result* PredictedPointSelector::usableResult(
    const std::vector<PredictedBallisticSolver::Result>& solved, int index) {
    if (index < 0 || index >= (int)solved.size()) return nullptr;
    const PredictedBallisticSolver::Result& r = solved[(size_t)index];
    if (r.masked || !r.success) return nullptr;          // 占位符 / 解算失败
    if (r.target_index >= 0 && r.target_index != index) return nullptr;  // 下标错位保护
    return &r;
}

int PredictedPointSelector::select(const std::vector<int>& masked_indices,
                                   const std::vector<PredictedBallisticSolver::Result>& solved,
                                   TimePoint timestamp) {
    ensureSize(solved.size());
    updateStates(masked_indices, timestamp);

    // 1) 粘滞：上一次选中的目标仍在候选中，且本步有可用解算结果 → 继续选它
    if (last_selected_index_ >= 0 &&
        last_selected_index_ < (int)states_.size() &&
        isCandidate(states_[(size_t)last_selected_index_].state) &&
        usableResult(solved, last_selected_index_) != nullptr) {
        return last_selected_index_;
    }

    // 2) 兜底：候选中预测高度 z 最低且本步有可用结果者
    int    best = -1;
    double best_z = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < states_.size(); ++i) {
        if (!isCandidate(states_[i].state)) continue;
        const PredictedBallisticSolver::Result* r = usableResult(solved, (int)i);
        if (r == nullptr) continue;
        const double z = (double)r->predicted_point[2];
        if (z < best_z) {
            best_z = z;
            best = (int)i;
        }
    }

    last_selected_index_ = best;
    return best;
}

void PredictedPointSelector::reset() {
    for (PointRecord& rec : states_) {
        rec.state = State::UNOBSERVED;
        rec.state_since = TimePoint{};
    }
    last_selected_index_ = -1;
}
