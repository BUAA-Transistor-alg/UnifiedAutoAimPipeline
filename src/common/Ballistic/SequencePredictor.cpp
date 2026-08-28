// SequencePredictor.cpp — 预测序列通用类实现
#include "common/Ballistic/SequencePredictor.h"

#include <algorithm>

#include "common/RobotConfig.h"

namespace {
// 每个工作线程领取一个固定的 GimbalSolver 编号（持久线程池 + thread_local →
// 同一 worker 始终使用同一个独立 GimbalSolver，内部 TaskPool 互不竞争）。
int workerGimbalIndex(std::atomic<int>& next) {
    thread_local int idx = next.fetch_add(1);
    return idx;
}
} // namespace

SequencePredictor::SequencePredictor()
    : extra_predict_time_(RobotConfig::instance().common.predictedBallistic.extraPredictTime),
      dt_control_(RobotConfig::instance().common.robotController.dtControl),
      pitch_bias_(RobotConfig::instance().common.predictSequence.pitchBias),
      yaw_bias_(RobotConfig::instance().common.predictSequence.yawBias),
      prediction_points_(RobotConfig::instance().common.predictSequence.predictionPoints),
      interpolation_refine_(RobotConfig::instance().common.predictSequence.interpolationRefine),
      exact_lead_points_(RobotConfig::instance().common.predictSequence.exactLeadPoints) {
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

SequencePredictor::Item SequencePredictor::lerpItem(const Item& lo, const Item& hi, double t)
{
    Item r;
    r.success = lo.success;
    r.predicted_point = lo.predicted_point + t * (hi.predicted_point - lo.predicted_point);
    r.predict_time = lo.predict_time + t * (hi.predict_time - lo.predict_time);
    r.yaw = (float)((double)lo.yaw + t * ((double)hi.yaw - (double)lo.yaw));
    r.pitch = (float)((double)lo.pitch + t * ((double)hi.pitch - (double)lo.pitch));
    r.gimbal_yaw = (float)((double)lo.gimbal_yaw + t * ((double)hi.gimbal_yaw - (double)lo.gimbal_yaw));
    r.gimbal_pitch = (float)((double)lo.gimbal_pitch + t * ((double)hi.gimbal_pitch - (double)lo.gimbal_pitch));
    r.flight_time = lo.flight_time + t * (hi.flight_time - lo.flight_time);
    r.target_index = lo.target_index;
    return r;
}

SequencePredictor::Item SequencePredictor::extrapItem(const Item& A, const Item& P, double s)
{
    // 沿段 (P, A) 的方向外推 s 个步长：R = A + s*(A - P)
    Item r;
    r.success = A.success;
    r.predicted_point = A.predicted_point + s * (A.predicted_point - P.predicted_point);
    r.predict_time = A.predict_time + s * (A.predict_time - P.predict_time);
    r.yaw = (float)((double)A.yaw + s * ((double)A.yaw - (double)P.yaw));
    r.pitch = (float)((double)A.pitch + s * ((double)A.pitch - (double)P.pitch));
    r.gimbal_yaw = (float)((double)A.gimbal_yaw + s * ((double)A.gimbal_yaw - (double)P.gimbal_yaw));
    r.gimbal_pitch = (float)((double)A.gimbal_pitch + s * ((double)A.gimbal_pitch - (double)P.gimbal_pitch));
    r.flight_time = A.flight_time + s * (A.flight_time - P.flight_time);
    r.target_index = A.target_index;
    return r;
}

SequencePredictor::Result SequencePredictor::predict(const RobotController::State& st,
                                           const Predictor& predictor,
                                           const std::chrono::steady_clock::time_point& timestamp,
                                           const std::chrono::steady_clock::time_point& predictor_timestamp)
{
    // 快照生成到本次消费之间的延迟：额外预测时间叠加该延迟，补偿 dt 零点（快照帧）
    // 与当前时刻的差值
    const double predictor_age = std::chrono::duration<double>(
        timestamp - predictor_timestamp).count();
    const double extra_predict_time = extra_predict_time_ + predictor_age;
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
    // ── yaw 系原点（world 系）：树已同步，同一线程内计算并加锁缓存，
    //    供输出模式跨线程读取（弹道线程化后不能直接读 GimbalSolver）──
    const cv::Vec3f yaw_origin = gimbals_.front()->yawWorldOrigin();

    // ── 1. 精确解算点集合（solve 之间并行）──
    // 返回序列 = [前 n 个前导精确点（索引 0..n-1）] 后接 [原划分序列
    // （索引 n .. n+(M-1)K）]，总返回点数 = (M-1)*K + 1 + n，
    // 时间间隔全程均匀为 dt_control（索引 i 对应 extra + (i+1)*dt）。
    // 精确解算共 n + M 个点：n 个前导点 + M 个原划分实际计算点（索引 n + j*K, j=0..M-1）。
    const int M = prediction_points_;
    const int K = interpolation_refine_;
    const int N = (M - 1) * K + 1;   // 原划分序列点数（不含前导）
    const int n = std::max(exact_lead_points_, 0);   // 防御性夹取（RobotConfig 已校验 >= 0）
    const int TOTAL = N + n;         // 总返回点数

    std::vector<int> solve_idx;
    solve_idx.reserve((size_t)n + (size_t)M);
    for (int i = 0; i < n; ++i) solve_idx.push_back(i);            // 前导精确点
    for (int j = 0; j < M; ++j) solve_idx.push_back(n + j * K);    // 原划分实际计算点

    const int U = (int)solve_idx.size();
    std::vector<PredictedBallisticSolver::Result> solved((size_t)U);
    const size_t T = gimbals_.size();
    pool_.run_parallel(U, [&](int idx) {
        const int wid = workerGimbalIndex(next_gimbal_);
        const int ret_idx = solve_idx[(size_t)idx];   // 该实际计算点在返回点序列中的索引
        solved[(size_t)idx] = solvers_[(size_t)(wid % T)].solve(
            predictor, extra_predict_time + (ret_idx + 1) * dt_control_);
    });

    // ── 2. 组装返回点序列（实际计算点 + 插值/外推/复制点）──
    std::vector<Item> items((size_t)TOTAL);
    auto makeActual = [&](const PredictedBallisticSolver::Result& r) {
        Item item;
        item.success = r.success;
        item.predicted_point = r.predicted_point;
        item.predict_time = r.predict_time;
        item.yaw = r.gimbal.yaw + (float)chassis_yaw + (float)yaw_bias_;   // 叠加底盘 yaw 修正与 yaw 偏置
        item.pitch = r.gimbal.pitch + (float)pitch_bias_;   // 叠加 pitch 偏置
        item.gimbal_yaw = r.gimbal.yaw;      // 原始关节角（相对底盘），供可视化复现需要的云台位姿
        item.gimbal_pitch = r.gimbal.pitch;
        item.flight_time = r.gimbal.flight_time;
        item.target_index = r.target_index;
        return item;
    };
    for (int u = 0; u < U; ++u) {
        items[(size_t)solve_idx[(size_t)u]] = makeActual(solved[(size_t)u]);
    }

    // ── 3. 原划分序列段间填充插值/外推/复制点 ──
    // 相邻实际计算点 (a, b)（a = n + j*K, b = a + K）之间填 K-1 个点：
    //   同目标 → 线性插值；
    //   目标不同 → 外推参考分两种：
    //     - 首段（j=0，即紧邻窗口 n+1..n+K-1）：用前导精确区最后一个点
    //       items[n-1]（n >= 1 时存在），要求与 A 同目标，否则复制 A；
    //     - 其余段：原规则 items[a-K]（上一实际计算点，须同目标，否则复制 A）。
    for (int j = 0; j < M - 1; ++j) {
        const size_t a = (size_t)(n + j * K);         // 左侧实际计算点索引
        const size_t b = (size_t)(n + (j + 1) * K);   // 右侧实际计算点索引
        const Item& A = items[a];
        const Item& B = items[b];

        // 首段外推参考：前导精确区最后一个点 items[n-1]（n >= 1 时存在，
        // 由短路的 n >= 1 保证下标合法），须与 A 同目标，否则复制 A
        const bool lead_extrap_valid = (j == 0) && (n >= 1) &&
            (items[(size_t)n - 1].target_index == A.target_index);

        // 原规则外推参考：A 的上一个实际计算点 P（索引 a-K），仅非首段使用
        bool can_extrap = false;
        Item P;
        if (j > 0) {
            const Item& Pp = items[a - (size_t)K];
            can_extrap = (Pp.target_index == A.target_index);
            if (can_extrap) P = Pp;
        }

        for (int m = 1; m < K; ++m) {
            const double t = (double)m / K;
            if (A.target_index == B.target_index) {
                // 目标相同：正常线性插值
                items[a + (size_t)m] = lerpItem(A, B, t);
            } else if (lead_extrap_valid) {
                // 首段（紧邻窗口）：以第 n 个精确值 items[n-1] 为参考外推
                items[a + (size_t)m] = extrapItem(A, items[(size_t)n - 1], t);
            } else if (can_extrap) {
                // 相邻实际点目标不同：用段 (P, A) 的线性差值参数外推
                items[a + (size_t)m] = extrapItem(A, P, t);
            } else {
                // 无可用外推参考：复制左侧点
                items[a + (size_t)m] = A;
            }
        }
    }

    // ── 4. 结果 ──
    Result res;
    res.items = std::move(items);
    res.valid = res.items.front().success;             // 第一个返回点 = 前导精确点（实际计算点）
    res.first_point = res.items.front().predicted_point;
    res.first_predict_time = res.items.front().predict_time;
    // 积分补偿开关：仅在预测有效且 MCU 自瞄开关打开时启用
    res.integral_enable = res.valid && (st.mcu.auto_aim_switch == 1);

    {
        std::lock_guard<std::mutex> lock(mtx_);
        latest_ = res;
        yaw_origin_ = yaw_origin;
    }
    return res;
}

void SequencePredictor::invalidate()
{
    std::lock_guard<std::mutex> lock(mtx_);
    latest_ = Result{};
}

SequencePredictor::Result SequencePredictor::latest() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return latest_;
}

cv::Vec3f SequencePredictor::yawWorldOrigin() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return yaw_origin_;
}
