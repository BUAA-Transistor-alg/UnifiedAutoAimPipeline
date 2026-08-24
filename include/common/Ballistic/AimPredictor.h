// AimPredictor.h — 预测瞄准点通用类（预测云台控制序列 + 瞄准点序列）
//
// 基于 PredictedBallisticSolver：对目标预测函数生成预测云台控制序列与对应的
// 瞄准点序列，并保存最新结果供多个输出模式消费：
//   - GimbalOutput    使用预测云台控制序列（yaw/pitch，已含底盘修正与 yaw/pitch 偏置），
//                     自行计算 fire 序列并截取后发送；
//   - VisualizeOutput 使用瞄准点序列中的第一个值绘制预测瞄准点。
//
// 序列生成（config common.input_controller）：
//   - 只精确解算 prediction_points（M）个实际计算点，时间间隔 K*dt_control；
//   - 相邻实际计算点之间按 interpolation_refine（K）细分，线性插值推出
//     K-1 个插值点；总返回点数 = (M-1)*K + 1，时间间隔恰为 dt_control，
//     第一个和最后一个返回点必为实际计算点；
//   - 相邻实际计算点选取的目标不同时：用左侧点与再上一个点之间的线性差值
//     参数外推；若左侧点与再上一个点也目标不同或没有再上一个点，直接复制
//     左侧点的值。
//
// 任何情况下（无论输出模式）由 main 每帧调用 predict()，保证瞄准点始终可用。
#ifndef AIM_PREDICTOR_H
#define AIM_PREDICTOR_H

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include <opencv2/opencv.hpp>

#include "RobotController.h"
#include "common/TaskPool.h"
#include "common/Ballistic/GimbalSolver.h"
#include "common/Ballistic/PredictedBallisticSolver.h"

class AimPredictor {
public:
    using Predictor = PredictedBallisticSolver::Predictor;

    // 单个序列返回点（云台控制值 + 瞄准点；实际计算点或插值/外推/复制生成）
    struct Item {
        bool   success = false;
        cv::Vec3f predicted_point;   // 瞄准点（world 系）
        double predict_time = 0.0;   // 预测时间 = 额外预测时间 + 飞行时间（秒）
        float  yaw = 0.0f;           // 云台解算 yaw（已叠加底盘修正与 yaw 偏置）
        float  pitch = 0.0f;         // 云台解算 pitch（已叠加 pitch 偏置）
        double flight_time = 0.0;    // 弹道飞行时间（秒）
        int    target_index = -1;    // 该点对应的目标索引（实际计算点为选中目标，插值/外推继承左侧）
    };

    // 预测结果：预测云台控制序列 + 瞄准点序列
    struct Result {
        bool valid = false;          // 首个返回点（实际计算点）解算是否有效
        bool integral_enable = false;  // 本帧 yaw 力矩积分补偿是否启用：
                                       // 仅当 valid 且 st.mcu.auto_aim_switch == 1 时为 true
        std::vector<Item> items;     // 总返回点数 = (M-1)*K + 1
        cv::Vec3f first_point;       // 瞄准点序列第一个值（可视化用）
        double first_predict_time = 0.0;
    };

    /// 构造时创建内部 GimbalSolver，序列/弹道/偏置参数从 RobotConfig common 段读取
    AimPredictor();

    // 目标选择策略透传（Outpost NEAREST / PowerRune LOWEST_Z），作用于全部线程实例
    void setTargetSelection(PredictedBallisticSolver::TargetSelection sel) {
        for (auto& s : solvers_) s.setTargetSelection(sel);
    }

    /// 同步内部树（st.strict + MCU 弹速）并生成预测云台控制序列 + 瞄准点序列。
    /// 实际计算点（solve）经内部线程池并行执行；每个工作线程通过 thread_local
    /// 绑定一个独立的 GimbalSolver（内部 pitch 粗搜索保持并行且互不竞争），
    /// 结果同时写入最新槽（线程安全）。
    ///
    /// @param timestamp          调用时刻（当前帧时间戳）
    /// @param predictor_timestamp 产生 predictor 快照的那一帧的时间戳（dt 的零点）；
    ///                            额外预测时间自动加上 (timestamp - predictor_timestamp)，
    ///                            补偿快照生成到消费之间的延迟
    Result predict(const RobotController::State& st, const Predictor& predictor,
                   const std::chrono::steady_clock::time_point& timestamp,
                   const std::chrono::steady_clock::time_point& predictor_timestamp);

    /// 预测器不可用：使最新结果失效
    void invalidate();

    /// 最新一次预测结果（无有效预测时 valid == false）
    Result latest() const;

    /// 任一内部云台解算器（fire 判定距离等用途；各线程实例树状态相同）
    std::shared_ptr<GimbalSolver> gimbal() const { return gimbals_.front(); }

private:
    // 为每个工作线程准备一个独立的 GimbalSolver（内部 TaskPool 互不竞争）
    std::vector<std::shared_ptr<GimbalSolver>> gimbals_;
    std::vector<PredictedBallisticSolver> solvers_;
    TaskPool pool_;                 // 默认构造（min(硬件核数/2, 4) 线程）
    std::atomic<int> next_gimbal_{0};   // thread_local 绑定：worker 首次执行时领取编号

    // 序列生成参数（构造时从 RobotConfig common 读取）
    double extra_predict_time_;
    double dt_control_;
    double pitch_bias_;
    double yaw_bias_;
    int    prediction_points_;       // M：实际精确解算点数
    int    interpolation_refine_;    // K：插值细化倍数

    // 线性插值：lo + t*(hi - lo)（t ∈ [0,1]）
    static Item lerpItem(const Item& lo, const Item& hi, double t);
    // 沿段 (P, A) 方向外推 s 步长：A + s*(A - P)（P 为 A 的前一实际计算点）
    static Item extrapItem(const Item& A, const Item& P, double s);

    // 最新结果槽
    mutable std::mutex mtx_;
    Result latest_;
};

#endif // AIM_PREDICTOR_H
