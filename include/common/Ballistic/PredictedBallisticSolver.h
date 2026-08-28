// PredictedBallisticSolver.h — 预测弹道解算器
#ifndef PREDICTED_BALLISTIC_SOLVER_H
#define PREDICTED_BALLISTIC_SOLVER_H

#include <functional>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common/Ballistic/GimbalSolver.h"

// ============================================================================
// 预测弹道解算器：结合"自身位姿预测函数"与 GimbalSolver，对目标关键点做
// 飞行时间迭代的提前量预测并解算云台角度。
//
// 对每个目标点 i 迭代求解：
//   1) 飞行时间首次估计 = |当前 muzzle 系原点(world) - 目标点| / 弹速；
//   2) 预测时间 = 额外预测时间(extra_predict_time，solve 时传入) + 飞行时间；
//   3) 预测点 = 预测函数(预测时间) 返回列表中的第 i 个点；
//   4) 用 GimbalSolver 解算该预测点，得到实际弹道飞行时间 flight_time；
//   5) 时间误差 = |本次用于预测的飞行时间 - flight_time|，小于容差则提前停止，
//      否则以 flight_time 作为下一次迭代的飞行时间；
//   6) 选取迭代中时间误差最小的一次作为该目标点的预测结果。
// 对列表中所有目标点，按目标选择策略选取实际使用目标：
//   - NEAREST（默认）：预测点距离当前 muzzle 系原点最近；
//   - LOWEST_Z：预测点 world z 最低（PowerRune 能量机关模式使用）。
//
// 预测序列由调用方（SequencePredictor）生成：对 i = 1..n，以
// extra_predict_time + i*dt_control 作为额外预测时间分别调用 solve()。
// ============================================================================
class PredictedBallisticSolver {
public:
    // 预测函数签名：传入预测时间（秒），返回该时刻的目标关键点列表（world 系）
    using Predictor = std::function<std::vector<cv::Point3f>(double)>;

    // 目标选择策略
    enum class TargetSelection { NEAREST, LOWEST_Z };

    // 单次求解结果
    struct Result {
        bool   success = false;         // 使用的 GimbalSolver 结果是否有效
        cv::Vec3f predicted_point;      // 使用的预测目标点（world 系）
        double predict_time = 0.0;      // 使用的预测时间 = 额外预测时间 + 飞行时间（秒）
        GimbalSolver::AimResult gimbal; // GimbalSolver 解算结果（含 yaw/pitch/flight_time）
        int    target_index = -1;       // 选中的目标在预测函数返回列表中的索引（目标选择策略结果）
    };

    // gimbal：云台角度解算器（内部树需由调用方同步当前底盘位姿与 yaw/pitch 关节角）。
    // 飞行时间迭代上限与时间误差容差直接从 RobotConfig（机器配置文件的
    // predicted_ballistic 配置段）读取，不在外部传入。
    explicit PredictedBallisticSolver(std::shared_ptr<GimbalSolver> gimbal);

    // 目标选择策略（默认 NEAREST），运行时可切换
    void setTargetSelection(TargetSelection sel) { target_selection_ = sel; }
    TargetSelection targetSelection() const { return target_selection_; }

    // 对预测函数返回列表中的每个目标点迭代求解，按目标选择策略返回最终目标的
    // 预测结果（预测点 / 预测时间 / GimbalSolver 结果）。
    // extra_predict_time：额外预测时间（秒），由调用方按序列元素传入（如 extra + i*dt）。
    Result solve(const Predictor& predictor, double extra_predict_time) const;

private:
    std::shared_ptr<GimbalSolver> gimbal_;
    int    max_iterations_;
    double time_error_tolerance_;
    TargetSelection target_selection_ = TargetSelection::NEAREST;
};

#endif // PREDICTED_BALLISTIC_SOLVER_H
