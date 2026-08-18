// AimPredictor.h — 预测瞄准点通用类（预测云台控制序列 + 瞄准点序列）
//
// 基于 PredictedBallisticSolver：对目标预测函数生成预测云台控制序列与对应的
// 瞄准点序列（i = 1..n，以 extra_predict_time + i*dt_control 为额外预测时间逐
// 个解算），并保存最新结果供多个输出模式消费：
//   - GimbalOutput    使用预测云台控制序列（yaw/pitch，已含底盘修正与 pitch 偏置），
//                     自行计算 fire 序列并截取后发送；
//   - VisualizeOutput 使用瞄准点序列中的第一个值绘制预测瞄准点。
//
// 任何情况下（无论输出模式）由 main 每帧调用 predict()，保证瞄准点始终可用。
#ifndef AIM_PREDICTOR_H
#define AIM_PREDICTOR_H

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include <opencv2/opencv.hpp>

#include "RobotController.h"
#include "Ballistic/GimbalSolver.h"
#include "Ballistic/PredictedBallisticSolver.h"

class AimPredictor {
public:
    using Predictor = PredictedBallisticSolver::Predictor;

    // 单个序列元素（一次解算的云台控制值 + 瞄准点）
    struct Item {
        bool   success = false;
        cv::Vec3f predicted_point;   // 瞄准点（world 系）
        double predict_time = 0.0;   // 预测时间 = 额外预测时间 + 飞行时间（秒）
        float  yaw = 0.0f;           // 云台解算 yaw（已叠加底盘修正）
        float  pitch = 0.0f;         // 云台解算 pitch（已叠加 pitch 偏置）
        double flight_time = 0.0;    // 弹道飞行时间（秒）
    };

    // 预测结果：预测云台控制序列 + 瞄准点序列
    struct Result {
        bool valid = false;          // 首个序列元素解算是否有效（first_success）
        std::vector<Item> items;     // i = 1..n 的序列元素
        cv::Vec3f first_point;       // 瞄准点序列第一个值（可视化用）
        double first_predict_time = 0.0;
    };

    /// 构造时创建内部 GimbalSolver，序列/弹道/偏置参数从 RobotConfig common 段读取
    AimPredictor();

    // 目标选择策略透传（Outpost NEAREST / PowerRune LOWEST_Z）
    void setTargetSelection(PredictedBallisticSolver::TargetSelection sel) {
        solver_.setTargetSelection(sel);
    }

    /// 同步内部树（st.strict + MCU 弹速）并生成预测云台控制序列 + 瞄准点序列，
    /// 结果同时写入最新槽（线程安全）。
    Result predict(const RobotController::State& st, const Predictor& predictor);

    /// 预测器不可用：使最新结果失效
    void invalidate();

    /// 最新一次预测结果（无有效预测时 valid == false）
    Result latest() const;

    /// 内部云台解算器（fire 判定距离等用途）
    std::shared_ptr<GimbalSolver> gimbal() const { return gimbal_; }

private:
    std::shared_ptr<GimbalSolver> gimbal_;
    PredictedBallisticSolver solver_;

    // 序列生成参数（构造时从 RobotConfig common 读取）
    double extra_predict_time_;
    double dt_control_;
    double pitch_bias_;
    int    prediction_seq_len_;

    // 最新结果槽
    mutable std::mutex mtx_;
    Result latest_;
};

#endif // AIM_PREDICTOR_H
