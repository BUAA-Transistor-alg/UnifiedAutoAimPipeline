// AimPredictor.cpp — 预测瞄准点通用类实现
#include "Ballistic/AimPredictor.h"

#include "RobotConfig.h"

namespace {
// 每个工作线程领取一个固定的 GimbalSolver 编号（持久线程池 + thread_local →
// 同一 worker 始终使用同一个独立 GimbalSolver，内部 TaskPool 互不竞争）。
int workerGimbalIndex(std::atomic<int>& next) {
    thread_local int idx = next.fetch_add(1);
    return idx;
}
} // namespace

AimPredictor::AimPredictor()
    : extra_predict_time_(RobotConfig::instance().common.predictedBallistic.extraPredictTime),
      dt_control_(RobotConfig::instance().common.robotController.dtControl),
      pitch_bias_(RobotConfig::instance().common.inputController.pitchBias),
      prediction_seq_len_(RobotConfig::instance().common.inputController.predictionSeqLen) {
    // 默认线程数 min(硬件核数/2, 4)；为每个工作线程准备一个独立的
    // GimbalSolver + 各自绑定的 PredictedBallisticSolver
    const size_t T = pool_.size();
    gimbals_.reserve(T);
    solvers_.reserve(T);
    for (size_t i = 0; i < T; ++i) {
        gimbals_.push_back(std::make_shared<GimbalSolver>());
        solvers_.emplace_back(gimbals_.back());
    }
}

AimPredictor::Result AimPredictor::predict(const RobotController::State& st,
                                           const Predictor& predictor)
{
    // ── 同步所有线程的独立 GimbalSolver 树（预测弹道解算依赖当前 muzzle 原点与弹速）──
    for (auto& g : gimbals_) {
        g->setChassisPosition(0.0f, 0.0f, 0.0f);
        g->setChassisEuler((float)st.strict.chassis_yaw, (float)st.strict.chassis_pitch, (float)st.strict.chassis_roll);
        g->setYaw((float)st.strict.yaw_pos);
        g->setPitch((float)st.strict.pitch_angle);
        if (st.mcu.valid) {
            g->setBulletVelocity(st.mcu.bullet_velocity);
        }
    }
    const double chassis_yaw = st.strict.imu_euler_yaw - st.strict.yaw_pos;  // 底盘 yaw 修正

    // ── 预测云台控制序列 + 瞄准点序列：i = 1..n，solve 之间并行，每线程独立 GimbalSolver ──
    const int n = prediction_seq_len_;
    const size_t T = gimbals_.size();
    std::vector<PredictedBallisticSolver::Result> solved((size_t)n);
    pool_.run_parallel(n, [&](int idx) {
        const int wid = workerGimbalIndex(next_gimbal_);
        const int i = idx + 1;
        solved[idx] = solvers_[(size_t)(wid % T)].solve(
            predictor, extra_predict_time_ + i * dt_control_);
    });

    Result res;
    res.items.reserve((size_t)n);
    for (int i = 1; i <= n; ++i) {
        const PredictedBallisticSolver::Result& r = solved[(size_t)(i - 1)];

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
