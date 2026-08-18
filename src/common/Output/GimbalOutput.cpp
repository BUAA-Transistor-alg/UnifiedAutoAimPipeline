// GimbalOutput.cpp — 云台控制输出模式实现
#include "Output/GimbalOutput.h"

#include <functional>
#include <vector>

GimbalOutput::GimbalOutput(RobotController& rc)
    : rc_(rc),
      gimbal_(std::make_shared<GimbalSolver>()),
      input_controller_(std::make_unique<InputController>(rc_, gimbal_)) {}

void GimbalOutput::update(const PipelineResult& result, RobotController*)
{
    // 无新帧时不重发序列，让 McuMpcController 后台 100Hz 线程正常消费已发送序列
    if (!result.valid) return;

    // ── 直接读取串口/MPC 状态（不经流水线）──
    const RobotController::State st = rc_.getState();

    // 同步 GimbalSolver 内部树（预测弹道解算依赖当前 muzzle 原点）
    gimbal_->setChassisPosition(0.0f, 0.0f, 0.0f);
    gimbal_->setChassisEuler((float)st.strict.chassis_yaw, (float)st.strict.chassis_pitch, (float)st.strict.chassis_roll);
    gimbal_->setYaw((float)st.strict.yaw_pos);
    gimbal_->setPitch((float)st.strict.pitch_angle);

    // MCU 数据有效时，用 MCU 传来的弹丸初速覆盖默认值
    if (st.mcu.valid) {
        gimbal_->setBulletVelocity(st.mcu.bullet_velocity);
    }

    if (result.outpost.esekf_initialized && result.outpost.predictor) {
        // ── Outpost：预测弹道解算 + 序列输入控制（目标选择：默认 muzzle 距离最近）──
        input_controller_->setTargetSelection(PredictedBallisticSolver::TargetSelection::NEAREST);
        input_controller_->update(st, *result.outpost.predictor);
    } else if (result.power_rune.target_predictor) {
        // ── PowerRune：靶点预测函数包装为统一签名（float→double, Vec3f→Point3f），
        //    目标选择策略用 z 最低（能量机关）──
        input_controller_->setTargetSelection(PredictedBallisticSolver::TargetSelection::LOWEST_Z);
        const auto& pr_pred = result.power_rune.target_predictor;   // const 引用，update 调用期间有效
        std::function<std::vector<cv::Point3f>(double)> wrapped =
            [&pr_pred](double dt) -> std::vector<cv::Point3f> {
            const auto pts = (*pr_pred)(static_cast<float>(dt));
            std::vector<cv::Point3f> out;
            out.reserve(pts.size());
            for (const auto& p : pts) out.emplace_back(p[0], p[1], p[2]);
            return out;
        };
        input_controller_->update(st, wrapped);
    } else {
        // 预测器不可用：保持模式（自瞄关闭，目标保持当前严格反解位置）。
        // 序列模式下用单元素序列调用序列 set。
        const double hold_yaw   = st.strict.yaw_pos;
        const double hold_pitch = st.strict.pitch_angle;
        rc_.set(/*auto_aim_enable=*/false, /*yaw_torque_only_mode=*/false,
                std::vector<double>{hold_yaw},
                std::vector<double>{hold_pitch},
                std::vector<bool>{false});
    }
}
