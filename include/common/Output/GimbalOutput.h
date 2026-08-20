// GimbalOutput.h — 云台控制输出模式
//
// 消费 AimPredictor 的预测云台控制序列（yaw/pitch，已含底盘修正与偏置）：
//  - 计算 fire 序列（MPC ref/pred 逐对判定 + 动态角度阈值）并按配置截取
//    （pitch 截第 m 个之后 / fire 截第 o 个之后）后发送到 RobotController；
//  - 预测不可用时进入保持模式（自瞄关闭，保持当前严格反解位置）。
// 瞄准点预测由 AimPredictor 统一完成（main 每帧调用），本模式不做解算。
#ifndef GIMBAL_OUTPUT_H
#define GIMBAL_OUTPUT_H

#include "common/Output/IOutputMode.h"
#include "common/Ballistic/AimPredictor.h"

#include <memory>
#include <vector>

class GimbalOutput : public IOutputMode {
public:
    GimbalOutput(AimPredictor& aim, RobotController& rc);

    /// @param rc 参数为 nullptr（本模式构造时已持有 RobotController 引用）
    void update(const PipelineResult& result, RobotController* rc) override;

    OutputMode type() const override { return OutputMode::GIMBAL; }
    std::string getName() const override { return "Gimbal"; }

    /// 本帧生成的序列输入（供显示/调试）
    struct LastOutput {
        bool auto_aim_enable = false;
        cv::Vec3f predicted_point = cv::Vec3f(0, 0, 0);
        double predict_time = 0.0;
        std::vector<double> yaw_seq;
        std::vector<double> pitch_seq;
        std::vector<bool>   fire_seq;
        bool mpc_available = false;
        double fire_threshold = 0.0;
    };
    const LastOutput& lastOutput() const { return last_; }

private:
    // 单对 (ref, pred) 的 fire 判定：角度差解缠绕后小于动态阈值
    static bool computeFire(double ref, double pred, double threshold);

    AimPredictor& aim_;
    RobotController& rc_;

    // ── 配置（构造时从 RobotConfig common 读取）──
    int    pitch_seq_lead_;
    int    fire_seq_lead_;
    double fire_angle_lower_limit_;
    double fire_angle_length_;

    LastOutput last_;
};

#endif // GIMBAL_OUTPUT_H
