// AimPredictor.cpp — 预测瞄准点通用类实现
#include "Ballistic/AimPredictor.h"

#include "RobotConfig.h"

AimPredictor::AimPredictor()
    : gimbal_(std::make_shared<GimbalSolver>()),
      solver_(gimbal_),
      extra_predict_time_(RobotConfig::instance().common.predictedBallistic.extraPredictTime),
      dt_control_(RobotConfig::instance().common.robotController.dtControl),
      pitch_bias_(RobotConfig::instance().common.inputController.pitchBias),
      prediction_seq_len_(RobotConfig::instance().common.inputController.predictionSeqLen) {}

AimPredictor::Result AimPredictor::predict(const RobotController::State& st,
                                           const Predictor& predictor)
{
    // ── 同步内部树（预测弹道解算依赖当前 muzzle 原点与弹速）──
    gimbal_->setChassisPosition(0.0f, 0.0f, 0.0f);
    gimbal_->setChassisEuler((float)st.strict.chassis_yaw, (float)st.strict.chassis_pitch, (float)st.strict.chassis_roll);
    gimbal_->setYaw((float)st.strict.yaw_pos);
    gimbal_->setPitch((float)st.strict.pitch_angle);
    if (st.mcu.valid) {
        gimbal_->setBulletVelocity(st.mcu.bullet_velocity);
    }
    const double chassis_yaw = st.strict.imu_euler_yaw - st.strict.yaw_pos;  // 底盘 yaw 修正

    // ── 预测云台控制序列 + 瞄准点序列：i = 1..n ──
    Result res;
    res.items.reserve((size_t)prediction_seq_len_);
    for (int i = 1; i <= prediction_seq_len_; ++i) {
        const double extra_i = extra_predict_time_ + i * dt_control_;
        const PredictedBallisticSolver::Result r = solver_.solve(predictor, extra_i);

        Item item;
        item.success = r.success;
        item.predicted_point = r.predicted_point;
        item.predict_time = r.predict_time;
        item.yaw = r.gimbal.yaw + (float)chassis_yaw;   // 叠加底盘 yaw 修正
        item.pitch = r.gimbal.pitch + (float)pitch_bias_;   // 叠加 pitch 偏置
        item.flight_time = r.gimbal.flight_time;
        res.items.push_back(item);

        if (i == 1) {   // 首个序列元素：供 fire 距离 / 可视化
            res.valid = r.success;
            res.first_point = r.predicted_point;
            res.first_predict_time = r.predict_time;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mtx_);
        latest_ = res;
    }
    return res;
}

void AimPredictor::invalidate()
{
    std::lock_guard<std::mutex> lock(mtx_);
    latest_ = Result{};
}

AimPredictor::Result AimPredictor::latest() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return latest_;
}
