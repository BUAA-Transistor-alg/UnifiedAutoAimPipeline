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
      yaw_bias_(RobotConfig::instance().common.inputController.yawBias),
      prediction_points_(RobotConfig::instance().common.inputController.predictionPoints),
      interpolation_refine_(RobotConfig::instance().common.inputController.interpolationRefine) {
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

AimPredictor::Item AimPredictor::lerpItem(const Item& lo, const Item& hi, double t)
{
    Item r;
    r.success = lo.success;
    r.predicted_point = lo.predicted_point + t * (hi.predicted_point - lo.predicted_point);
    r.predict_time = lo.predict_time + t * (hi.predict_time - lo.predict_time);
    r.yaw = (float)((double)lo.yaw + t * ((double)hi.yaw - (double)lo.yaw));
    r.pitch = (float)((double)lo.pitch + t * ((double)hi.pitch - (double)lo.pitch));
    r.flight_time = lo.flight_time + t * (hi.flight_time - lo.flight_time);
    r.target_index = lo.target_index;
    return r;
}

AimPredictor::Item AimPredictor::extrapItem(const Item& A, const Item& P, double s)
{
    // 沿段 (P, A) 的方向外推 s 个步长：R = A + s*(A - P)
    Item r;
    r.success = A.success;
    r.predicted_point = A.predicted_point + s * (A.predicted_point - P.predicted_point);
    r.predict_time = A.predict_time + s * (A.predict_time - P.predict_time);
    r.yaw = (float)((double)A.yaw + s * ((double)A.yaw - (double)P.yaw));
    r.pitch = (float)((double)A.pitch + s * ((double)A.pitch - (double)P.pitch));
    r.flight_time = A.flight_time + s * (A.flight_time - P.flight_time);
    r.target_index = A.target_index;
    return r;
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

    // ── 1. 只精确解算 M 个实际计算点（索引 (j-1)*K，j=1..M），solve 之间并行 ──
    const int M = prediction_points_;
    const int K = interpolation_refine_;
    const int N = (M - 1) * K + 1;   // 总返回点数
    std::vector<PredictedBallisticSolver::Result> solved((size_t)M);
    const size_t T = gimbals_.size();
    pool_.run_parallel(M, [&](int idx) {
        const int wid = workerGimbalIndex(next_gimbal_);
        const int j = idx + 1;
        const int ret_idx = (j - 1) * K;   // 该实际计算点在返回点序列中的索引
        solved[idx] = solvers_[(size_t)(wid % T)].solve(
            predictor, extra_predict_time_ + (ret_idx + 1) * dt_control_);
    });

    // ── 2. 组装返回点序列（实际计算点 + 插值/外推/复制点）──
    std::vector<Item> items((size_t)N);
    auto makeActual = [&](const PredictedBallisticSolver::Result& r) {
        Item item;
        item.success = r.success;
        item.predicted_point = r.predicted_point;
        item.predict_time = r.predict_time;
        item.yaw = r.gimbal.yaw + (float)chassis_yaw + (float)yaw_bias_;   // 叠加底盘 yaw 修正与 yaw 偏置
        item.pitch = r.gimbal.pitch + (float)pitch_bias_;   // 叠加 pitch 偏置
        item.flight_time = r.gimbal.flight_time;
        item.target_index = r.target_index;
        return item;
    };
    for (int j = 1; j <= M; ++j) {
        items[(size_t)((j - 1) * K)] = makeActual(solved[(size_t)(j - 1)]);
    }

    // ── 3. 相邻实际计算点之间填充插值/外推/复制点 ──
    for (int j = 1; j < M; ++j) {
        const size_t a = (size_t)((j - 1) * K);   // 左侧实际计算点索引
        const size_t b = (size_t)(j * K);         // 右侧实际计算点索引
        const Item& A = items[a];
        const Item& B = items[b];

        // 左侧点 A 的上一个实际计算点 P（索引 a-K），判断是否可用作外推参考
        bool can_extrap = false;
        Item P;
        if (a >= (size_t)K) {
            const Item& Pp = items[a - (size_t)K];
            can_extrap = (Pp.target_index == A.target_index);
            if (can_extrap) P = Pp;
        }

        for (int m = 1; m < K; ++m) {
            const double t = (double)m / K;
            if (A.target_index == B.target_index) {
                // 目标相同：正常线性插值
                items[a + (size_t)m] = lerpItem(A, B, t);
            } else if (can_extrap) {
                // 相邻实际点目标不同：用段 (P, A) 的线性差值参数外推
                items[a + (size_t)m] = extrapItem(A, P, t);
            } else {
                // 左侧点与再上一个点也目标不同或没有再上一个点：复制左侧点
                items[a + (size_t)m] = A;
            }
        }
    }

    // ── 4. 结果 ──
    Result res;
    res.items = std::move(items);
    res.valid = res.items.front().success;             // 第一个返回点 = 实际计算点 1
    res.first_point = res.items.front().predicted_point;
    res.first_predict_time = res.items.front().predict_time;

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
