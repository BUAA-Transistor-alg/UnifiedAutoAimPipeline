// SequencePredictor.cpp — 预测序列通用类实现
#include "common/Ballistic/SequencePredictor.h"

#include "common/TransformTree/TfTreeSync.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "common/RobotConfig.h"

namespace {
// 每个工作线程领取一个固定的 GimbalSolver 编号（持久线程池 + thread_local →
// 同一 worker 始终使用同一个独立 GimbalSolver，内部 TaskPool 互不竞争）。
int workerGimbalIndex(std::atomic<int>& next) {
    thread_local int idx = next.fetch_add(1);
    return idx;
}

// 目标选择策略（移入本类：PredictedBallisticSolver::solve 返回全部目标点结果，
// 由 SequencePredictor 在结果之间做实际目标选择）
enum class TargetStrategy { NEAREST, LOWEST_Z };

// 依据目标预测器来源标注自动选择目标策略：PowerRune → LOWEST_Z，其余（Armor）→ NEAREST
TargetStrategy targetStrategyForSource(const SequencePredictor::PredictorSource& source) {
    return (source.kind == SequencePredictor::PredictorSource::Kind::POWER_RUNE)
               ? TargetStrategy::LOWEST_Z
               : TargetStrategy::NEAREST;
}

// 在单个实际计算点解出的全部目标点结果中，按策略选出实际使用的目标：
//   - NEAREST：预测点距离当前 muzzle 原点最近（默认）；
//   - LOWEST_Z：预测点 world z 最低（PowerRune 能量机关模式）。
// predictor.masked_indices 中索引对应的瞄准点（目标）不参与选择。
// 需求3（角速度可用的 Armor 目标且本帧非 fast_target）：center_aim_angles 非空时
// 优先不考虑 |中心瞄准夹角| > π/2 的瞄准点——该指标优先于其它指标（含下面的慢
// 目标粘滞）：先在剩余瞄准点中选，一个都不剩时才退回全部瞄准点。
// 慢目标瞄准点滞回（Armor 目标，本帧判定为慢目标时）：
//   sticky_index >= 0 表示本实际计算点“粘”的瞄准点索引（上一帧序列第一个值
//   选中的点 / 本帧前一个实际计算点选中的点）。粘滞目标若未被屏蔽且存在于候选
//   列表中，则保持它，仅当存在其它未被屏蔽目标比它“明显更好”（criterion 优于
//   sticky 超过 stick_delta，delta = aim_stick_ratio × t=0 全瞄准点到预测车体中心的平均距离）
//   时才切换；sticky 不存在 / 被屏蔽 / 为 -1 → 退回纯策略选最优。
// 结果为空（无可用目标）时返回默认无效 Result（success=false）。
PredictedBallisticSolver::Result selectTargetResult(
    const std::vector<PredictedBallisticSolver::Result>& candidates,
    TargetStrategy strategy, const cv::Vec3f& muzzle_origin,
    const SequencePredictor::Predictor& predictor,
    int sticky_index, double stick_delta,
    const std::vector<double>* center_aim_angles) {
    PredictedBallisticSolver::Result best;
    if (candidates.empty()) return best;

    const bool lowest_z = (strategy == TargetStrategy::LOWEST_Z);
    auto criterionOf = [&](const PredictedBallisticSolver::Result& c) {
        return lowest_z
            ? (double)c.predicted_point[2]                        // world z
            : (double)cv::norm(muzzle_origin - c.predicted_point); // muzzle 距离
    };

    // 候选（未被屏蔽）下标集合
    std::vector<int> pool;
    pool.reserve(candidates.size());
    for (int i = 0; i < (int)candidates.size(); ++i) {
        if (predictor.isIndexMasked(candidates[(size_t)i].target_index)) continue;
        pool.push_back(i);
    }
    // 需求3：优先排除 |中心瞄准夹角| > π/2 的瞄准点（仅当调用方给出夹角时）
    if (center_aim_angles != nullptr) {
        std::vector<int> preferred;
        preferred.reserve(pool.size());
        for (int i : pool) {
            const double a = (*center_aim_angles)[(size_t)i];
            if (std::isfinite(a) && std::fabs(a) <= M_PI / 2.0) preferred.push_back(i);
        }
        if (!preferred.empty()) pool.swap(preferred);   // 全部超出时退回全部候选
    }

    // 全局最优（未被屏蔽）；同时寻找粘滞目标（未被屏蔽）
    double best_criterion = std::numeric_limits<double>::infinity();
    const PredictedBallisticSolver::Result* sticky = nullptr;
    double sticky_criterion = 0.0;
    bool sticky_found = false;
    for (int i : pool) {
        const PredictedBallisticSolver::Result& c = candidates[(size_t)i];
        const double criterion = criterionOf(c);
        if (sticky_index >= 0 && c.target_index == sticky_index) {
            sticky = &c;
            sticky_criterion = criterion;
            sticky_found = true;
        }
        if (criterion < best_criterion) {
            best_criterion = criterion;
            best = c;
        }
    }
    // 粘滞目标存在（未被屏蔽）时：仅当有其它目标比它“明显更好”（好过 stick_delta）
    // 才切换，否则保持粘滞目标（含打平情形）。
    if (sticky_found) {
        const double best_other_criterion =
            (best.target_index == sticky_index) ? std::numeric_limits<double>::infinity()
                                                : best_criterion;
        if (best_other_criterion < sticky_criterion - stick_delta) {
            return best;   // 其它目标确实更优：切换
        }
        return *sticky;    // 保持当前（粘滞）目标
    }
    return best;
}
} // namespace

SequencePredictor::SequencePredictor()
    : extra_predict_time_(RobotConfig::instance().common.predictedBallistic.extraPredictTime),
      dt_control_(RobotConfig::instance().common.dtControl()),
      pitch_bias_(RobotConfig::instance().common.predictSequence.pitchBias),
      yaw_bias_(RobotConfig::instance().common.predictSequence.yawBias),
      prediction_points_(RobotConfig::instance().common.predictSequence.predictionPoints),
      interpolation_refine_(RobotConfig::instance().common.predictSequence.interpolationRefine),
      exact_lead_points_(RobotConfig::instance().common.predictSequence.exactLeadPoints),
      aim_stick_ratio_(RobotConfig::instance().common.predictSequence.aimStickRatio),
      // 慢目标施密特触发阈值：判定已移入本类 predict()，阈值仍取自 Armor 目标选取配置
      slow_w_lower_(RobotConfig::instance().armor.targetSelection.slowAngularVelocityLower),
      slow_w_upper_(RobotConfig::instance().armor.targetSelection.slowAngularVelocityUpper),
      // 快目标（fast_target）施密特触发阈值：同段读取的第二对阈值（均高于慢目标上阈值）
      fast_w_lower_(RobotConfig::instance().armor.targetSelection.fastAngularVelocityLower),
      fast_w_upper_(RobotConfig::instance().armor.targetSelection.fastAngularVelocityUpper),
      // fast_target 火控几何参数：每块板旋转容差角 = max(下限, fire_angle_length / 该板半径)
      fire_angle_length_(RobotConfig::instance().common.predictSequence.fireAngleLength),
      min_rotation_tolerance_angle_(
          RobotConfig::instance().common.predictSequence.minRotationToleranceAngle) {
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

SequencePredictor::Result SequencePredictor::predict(const tcs::RobotController::State& st,
                                           const Predictor& predictor,
                                           const std::chrono::steady_clock::time_point& timestamp)
{
    return predictImpl(snapshotFromSingle(st), predictor, timestamp);
}

SequencePredictor::Result SequencePredictor::predict(const bsy::RobotState& st,
                                           const Predictor& predictor,
                                           const std::chrono::steady_clock::time_point& timestamp)
{
    return predictImpl(snapshotFromBigSmall(st), predictor, timestamp);
}

// 单 yaw 构型的输入快照（旧行为：只读 st.strict + st.mcu）
SequencePredictor::InputSnapshot
SequencePredictor::snapshotFromSingle(const tcs::RobotController::State& st) const {
    InputSnapshot in;
    in.big_small = false;
    in.info.single.yaw_pos         = st.strict.yaw_pos;
    in.info.single.pitch_angle     = st.strict.pitch_angle;
    in.info.single.imu_euler_yaw   = st.strict.imu_euler_yaw;
    in.info.single.imu_euler_pitch = st.strict.imu_euler_pitch;
    in.info.single.imu_euler_roll  = st.strict.imu_euler_roll;
    in.info.chassis_yaw   = st.strict.chassis_yaw;
    in.info.chassis_pitch = st.strict.chassis_pitch;
    in.info.chassis_roll  = st.strict.chassis_roll;
    in.has_bullet_velocity = st.mcu.valid;
    in.bullet_velocity     = st.mcu.bullet_velocity;
    in.auto_aim_switch     = (st.mcu.auto_aim_switch == 1);
    // 底盘 yaw 修正项（原式：imu_euler_yaw − yaw_pos）
    in.chassis_yaw_correction = st.strict.imu_euler_yaw - st.strict.yaw_pos;
    return in;
}

// 大小 yaw 构型的输入快照（适配器状态包 → ExtraInputInfo 的大小 yaw 包 + 弹速/开关 + MPC 预测序列）
SequencePredictor::InputSnapshot
SequencePredictor::snapshotFromBigSmall(const bsy::RobotState& st) const {
    InputSnapshot in;
    in.big_small = true;
    in.info = bsy::toExtraInputInfo(st);
    in.has_bullet_velocity = st.valid && std::isfinite(st.bullet_velocity) && st.bullet_velocity > 0.0;
    in.bullet_velocity     = st.bullet_velocity;
    in.auto_aim_switch     = (st.auto_aim_switch == 1);
    // 底盘 yaw 修正项：严格反解的 chassis 欧拉 yaw（ZXY，与树的 chassis 欧拉角同一约定）
    in.chassis_yaw_correction = st.info_chassis_yaw;
    in.pred_big_azimuth_seq   = st.pred_big_azimuth_seq;
    return in;
}

// 中心瞄准夹角（需求1，rad，(-π, π]）：方向向量 = 中心位置 − 瞄准点位置，
// 中心瞄准向量 = 中心位置 − 自身 yaw 轴旋转中心；两者投影到 xy 平面后，
// 求方向向量相对中心瞄准向量的有向夹角（方向向量逆时针为正，即
// atan2(dir × ref, dir · ref)）。夹角 = 0 表示瞄准点位于 yaw 轴旋转中心与目标
// 中心连线上（近侧、正对射手）；夹角对时间的导数 = 目标角速度 ω（Armor EKF 的
// 角速度符号约定），故夹角与 ω 异号 ⇒ 正在向枪线靠近（对齐时间 |夹角|/|ω|）。
// 两向量 xy 投影退化（长度 ~0）时返回 0（视为已对齐，不产生虚假判据）。
double SequencePredictor::centerAimAngle(const cv::Vec3f& center, const cv::Vec3f& aim_point,
                                         const cv::Vec3f& yaw_origin) {
    const double dir_x = (double)center[0] - (double)aim_point[0];     // 方向向量 xy 投影
    const double dir_y = (double)center[1] - (double)aim_point[1];
    const double ref_x = (double)center[0] - (double)yaw_origin[0];    // 中心瞄准向量 xy 投影
    const double ref_y = (double)center[1] - (double)yaw_origin[1];
    if (std::hypot(dir_x, dir_y) < 1e-9 || std::hypot(ref_x, ref_y) < 1e-9) return 0.0;
    // 有向夹角 = atan2(叉积, 点积)：方向向量相对中心瞄准向量逆时针旋转时为正
    const double cross = ref_x * dir_y - ref_y * dir_x;
    const double dot   = ref_x * dir_x + ref_y * dir_y;
    return std::atan2(cross, dot);
}

// 任意时刻的大 yaw 关节角（仅 BIG_SMALL）：与 buildYawBigSequence 同一时间轴
// （返回点索引 i 对应 (i+1)·dt_control），线性插值 + 超出覆盖区间末值保持；
// MPC 预测序列不可用时退回上一轮解算序列（按索引夹取），再没有用当帧实测 θ_big。
float SequencePredictor::yawBigAtTime(const InputSnapshot& in, double t) const {
    const std::vector<double>& seq = in.pred_big_azimuth_seq;
    if (!seq.empty()) {
        const int nLast = (int)seq.size() - 1;
        double idx_f = t / dt_control_;            // dt_control 为单位
        if (idx_f < 0.0) idx_f = 0.0;
        if (idx_f > (double)nLast) idx_f = (double)nLast;   // 超出覆盖区间：末值保持
        const int i0 = (int)std::floor(idx_f);
        const int i1 = std::min(i0 + 1, nLast);
        const double f = idx_f - (double)i0;
        const double psi = (double)seq[(size_t)i0] * (1.0 - f) + (double)seq[(size_t)i1] * f;
        return (float)(psi - in.info.chassis_yaw);   // 世界方位角 → 关节角
    }
    if (!last_yaw_big_seq_.empty()) {
        const double idx_f = std::max(0.0, t / dt_control_);
        const size_t k = std::min((size_t)idx_f, last_yaw_big_seq_.size() - 1);
        return last_yaw_big_seq_[k];
    }
    return (float)in.info.big_small.yaw_big_pos;
}

// 逐返回点的大 yaw 关节角序列（仅 BIG_SMALL 使用）：
//   θ_big(t) = ψ_big_pred(t) − ψ_chassis（当帧）
//   ψ_big_pred 取 MPC 预测序列（线性插值，超出覆盖区间保持最后一个值）；
//   不可用时退回上一轮解算序列（按索引夹取），再没有则用当帧实测 θ_big。
std::vector<float> SequencePredictor::buildYawBigSequence(const InputSnapshot& in,
                                                          int total_points) const {
    std::vector<float> out((size_t)std::max(0, total_points), 0.0f);
    if (!in.big_small || total_points <= 0) return out;
    // 第 i 个返回点对应的预测时间 = (i+1)·dt_control（与弹道解算时间基准一致）
    for (int i = 0; i < total_points; ++i) {
        out[(size_t)i] = yawBigAtTime(in, (double)(i + 1) * dt_control_);
    }
    return out;
}

SequencePredictor::Result SequencePredictor::predictImpl(const InputSnapshot& in,
                                           const Predictor& predictor,
                                           const std::chrono::steady_clock::time_point& timestamp)
{
    // ── 目标屏蔽检查：predictor.masked_indices 中索引对应的瞄准点不参与目标
    // 选择；但须保证屏蔽后至少还有一个瞄准点可选——若预测函数当前返回的全部
    // 瞄准点都被屏蔽（全被屏蔽），本帧预测器等同不可用：自动转为调用
    // invalidate()（重置自身跨帧状态与来源记录）并返回无效结果，与 main 在
    // "无可用预测器"时直接 invalidate() 的行为一致（输出模式进入保持模式）。
    if (!predictor.masked_indices.empty()) {
        const std::vector<cv::Point3f> aims_now = predictor.function(0.0).second;
        bool any_aim_left = false;
        for (int i = 0; i < (int)aims_now.size(); ++i) {
            if (!predictor.isIndexMasked(i)) {
                any_aim_left = true;
                break;
            }
        }
        if (!any_aim_left) {
            invalidate();
            return Result{};
        }
    }

    // ── 自身跨帧状态：target_predictor 来源切换（含首次从无来源进入）时重置 ──
    // State 存放慢目标施密特锁存与慢目标瞄准点滞回所需的“上一帧序列第一个值
    // 瞄准点索引”；来源切换（如 Armor 目标种类变化 / Armor → PowerRune）时自动
    // 清零（锁存与粘滞点均随总目标切换失效）。
    if (!(active_source_ == predictor.source)) {
        state_ = State{};
        active_source_ = predictor.source;
    }

    // ── 慢目标判定（施密特触发器，无连续帧计数；判定已从 ArmorPipeline
    //    ::processStage5 移入本类，流水线只下发原始角速度及其可用标志）──
    // 仅对“需要判定”的目标判定：Predictor::omega_valid == true 时取
    // |target_omega|（带正负，rad/s）——|ω| < lower → 慢目标；|ω| > upper → 非慢目标；
    // lower <= |ω| <= upper → 保持上一帧判定（锁存 state_.slow_latch）。
    // omega_valid == false（PowerRune、基地 label 7/8 等无角速度属性，或角速度当前
    // 不可用）按原方法处理：锁存复位，本帧不判定为慢目标（不启用瞄准点滞回）。
    // 判定结果仅在本帧内部使用（是否允许瞄准点滞回），不再随预测器/结果下发。
    bool slow_target = false;
    if (predictor.omega_valid) {
        const double omega_abs = std::fabs(predictor.target_omega);
        if (omega_abs < slow_w_lower_) state_.slow_latch = true;
        else if (omega_abs > slow_w_upper_) state_.slow_latch = false;
        // lower <= |ω| <= upper：保持 state_.slow_latch 不变
        slow_target = state_.slow_latch;
    } else {
        state_.slow_latch = false;
    }

    // ── 快目标判定（需求2：第二对施密特触发器阈值，同一套判定方式）──
    // 仅对“角速度可用的 Armor 类目标”判定：|ω| > fast_upper → fast_target；
    // |ω| < fast_lower → 非 fast_target；介于两者之间保持锁存（防抖）。
    // 角速度不可用（PowerRune、基地 label 7/8、EKF 未就绪）或非 Armor 来源：
    // 锁存复位，本帧不进入 fast_target 分支（行为与改造前一致）。
    const bool armor_omega =
        (predictor.source.kind == PredictorSource::Kind::ARMOR) && predictor.omega_valid;
    bool fast_target = false;
    if (armor_omega) {
        const double omega_abs = std::fabs(predictor.target_omega);
        if (omega_abs > fast_w_upper_) state_.fast_latch = true;       // 越过上阈值：进入快目标
        else if (omega_abs < fast_w_lower_) state_.fast_latch = false; // 跌破下阈值：退出快目标
        // fast_lower <= |ω| <= fast_upper：保持 state_.fast_latch 不变
        fast_target = state_.fast_latch;
    } else {
        state_.fast_latch = false;
    }

    // 目标选择策略由来源自动选择（Armor → NEAREST，PowerRune → LOWEST_Z）。
    // 目标选择已移入本类：PredictedBallisticSolver::solve 返回全部目标点的解算
    // 结果，实际目标（瞄准点）选择在下方“并行解算 + 顺序粘滞选择”两步中完成
    // （不再设置 solver 的目标选择状态）。
    const TargetStrategy sel = targetStrategyForSource(predictor.source);

    // 快照生成到本次消费之间的延迟：额外预测时间叠加该延迟，补偿 dt 零点（快照帧）
    // 与当前时刻的差值
    const double predictor_age = std::chrono::duration<double>(
        timestamp - predictor.timestamp).count();
    const double extra_predict_time = extra_predict_time_ + predictor_age;
    // ── 同步所有线程的独立 GimbalSolver 树（预测弹道解算依赖当前 muzzle 原点与弹速）──
    for (auto& g : gimbals_) {
        g->setChassisPosition(0.0f, 0.0f, 0.0f);
        // 底盘位姿 + **当前构型**的关节角包（大小 yaw 构型下自动写 θ_big / θ_small / pitch；
        // 包未填充（NaN）时抛异常，不会静默用 NaN 解算）
        applyExtraInputInfoToTree(g->tree(), in.info);
        g->setChassisPosition(0.0f, 0.0f, 0.0f);   // 弹道解算固定以底盘为世界原点
        if (in.has_bullet_velocity) {
            g->setBulletVelocity(in.bullet_velocity);
        }
    }
    const double chassis_yaw = in.chassis_yaw_correction;  // item.yaw 的底盘 yaw 修正项
    // ── yaw 系原点（world 系）：树已同步，同一线程内计算并写入 Result，
    //    随结果沿级联传递给输出模式（弹道线程化后不能直接读 GimbalSolver）──
    const cv::Vec3f yaw_world_origin = gimbals_.front()->yawWorldOrigin();
    // NEAREST 判据所需的当前 muzzle 原点（world 系）：全部线程树已同步且一致，
    // 目标选择在各 worker 内用该值计算（与原来 solve() 内部判据一致）
    const cv::Vec3f muzzle_origin = gimbals_.front()->muzzleWorldOrigin();

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
    const size_t T = gimbals_.size();

    // ── 大小 yaw 构型：逐返回点的大 yaw 关节角 θ_big（世界方位角预测序列 − 当帧底盘 yaw）──
    // 用于逐点求“有效 yaw 旋转中心”（小 yaw 轴偏移随大 yaw 旋转），并作为 MPC 不可用时的退路。
    std::vector<float> yaw_big_seq;
    if (in.big_small) {
        yaw_big_seq = buildYawBigSequence(in, TOTAL);
        last_yaw_big_seq_ = yaw_big_seq;   // 记录本轮解算所用序列，供下一轮退路使用
    }

    // ── 慢目标瞄准点滞回参数（仅本帧判定为慢目标且 ratio > 0 时启用）──
    // 滞回量 = aim_stick_ratio_ × (t=0 全部瞄准点到预测车体中心的平均距离；忽略
    // mask、用全部点，以预测函数直接给出的车体中心为基准——不再由全部瞄准点的
    // 均值位置推算中心，仅作该目标瞄准点分布的几何尺度估计)。
    const bool aim_stick_enabled = slow_target && aim_stick_ratio_ > 0.0;
    double aim_stick_delta = 0.0;
    if (aim_stick_enabled) {
        const PredictedBallisticSolver::PredictorResult now = predictor.function(0.0);
        const std::vector<cv::Point3f>& aims_now = now.second;
        if (!aims_now.empty()) {
            const cv::Vec3f center(now.first.x, now.first.y, now.first.z);
            double dist_sum = 0.0;
            for (const auto& p : aims_now) {
                const cv::Vec3f v(p.x, p.y, p.z);
                dist_sum += cv::norm(v - center);
            }
            aim_stick_delta = aim_stick_ratio_ * (dist_sum / (double)aims_now.size());
        }
    }

    // ── 1. 并行解算：每个实际计算点独立求出全部目标点的解算结果（不做目标选择）──
    std::vector<std::vector<PredictedBallisticSolver::Result>> candidates_all((size_t)U);
    pool_.run_parallel(U, [&](int idx) {
        const int wid = workerGimbalIndex(next_gimbal_);
        const int ret_idx = solve_idx[(size_t)idx];   // 该实际计算点在返回点序列中的索引
        // solve() 返回预测函数列表中全部目标点的解算结果（不再内部选目标）；
        // 实际目标选择在下方顺序循环完成。
        // 大小 yaw 构型：该点用“该时刻大 yaw 实际会到哪”求有效 yaw 旋转中心（θ_big(t)）；
        // 单 yaw 构型：传 NaN ⇒ 走原路径（用树当前 yaw 关节角）
        const float yaw_big = in.big_small ? yaw_big_seq[(size_t)ret_idx]
                                           : std::numeric_limits<float>::quiet_NaN();
        candidates_all[(size_t)idx] = solvers_[(size_t)(wid % T)].solve(
            predictor.function, extra_predict_time + (ret_idx + 1) * dt_control_, yaw_big);
    });

    // ── 2. 顺序目标（瞄准点）选择（按时间顺序逐点传递粘滞）──
    // 粘滞链（仅 aim_stick_enabled 时生效）：本帧第一个实际计算点（= 序列第一个
    // 值）粘上一帧序列第一个值选中的瞄准点索引（state_.last_first_target_index；
    // 无上一帧/来源切换后为 -1 → 直接选最优）；本帧后续实际计算点粘本帧前一个
    // 实际计算点选中的索引（插值点继承左端实际点目标，无需另行处理）。
    // selectTargetResult 内部处理粘滞被屏蔽/不存在时退回纯策略选最优。
    // 需求3（角速度可用的 Armor 目标且本帧非 fast_target）：先为每个候选瞄准点算
    // 中心瞄准夹角（在其预测时刻测量），交给 selectTargetResult“优先排除 |夹角| > π/2”
    // ——该指标优先于其它指标（含慢目标粘滞）。fast_target 帧整条序列会被单点解算
    // 替换、不使用逐点选择结果，故不算夹角（省去预测函数调用）。
    const bool angle_filter_enabled = armor_omega && !fast_target;
    std::vector<PredictedBallisticSolver::Result> solved((size_t)U);
    int sticky_index = aim_stick_enabled ? state_.last_first_target_index : -1;
    for (int u = 0; u < U; ++u) {
        std::vector<double> angles;
        const std::vector<double>* angles_ptr = nullptr;
        if (angle_filter_enabled && !candidates_all[(size_t)u].empty()) {
            angles.resize(candidates_all[(size_t)u].size());
            for (size_t ci = 0; ci < candidates_all[(size_t)u].size(); ++ci) {
                const PredictedBallisticSolver::Result& c = candidates_all[(size_t)u][ci];
                // 与瞄准点同时预测出来的车体中心（同刻预测中心）
                const PredictedBallisticSolver::PredictorResult pr = predictor.function(c.predict_time);
                const cv::Vec3f center(pr.first.x, pr.first.y, pr.first.z);
                angles[ci] = centerAimAngle(center, c.predicted_point, yaw_world_origin);
            }
            angles_ptr = &angles;
        }
        solved[(size_t)u] = selectTargetResult(
            candidates_all[(size_t)u], sel, muzzle_origin, predictor,
            aim_stick_enabled ? sticky_index : -1,
            aim_stick_enabled ? aim_stick_delta : 0.0,
            angles_ptr);
        if (aim_stick_enabled) {
            sticky_index = (solved[(size_t)u].target_index >= 0)
                               ? solved[(size_t)u].target_index
                               : -1;
        }
    }

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

    // ── 4. fast_target（需求3/4）：整条返回序列替换为一次新的单点解算 ──
    // 取「中心瞄准夹角与 ω 异号（正在靠近枪线）且 |夹角| 最小」的 (返回点, 板)；
    // 若没有任何异号者，退回「同号且 |夹角| 最大」的那一个。用该返回点的**总预测
    // 时间**（items[i].predict_time）+ |夹角|/|ω| 作为预测时刻、该板序号为索引做
    // 一次单点解算（solveSingle，不再迭代弹道飞行时间），并把全部 item 都换成这次
    // 解算的结果（云台角/瞄准点相同）——即瞄准那块即将与枪线对齐的板。
    // 同时算出 fast 元数据（参考时刻/参考板/参考夹角/每块板旋转容差角）随 Result
    // 下发，供云台输出模式做火控数组的“有目标在枪线上”判定。
    // 新解算失败时保留原逐点解算序列（不下发 fast 元数据，退回原开火条件）。
    bool   fast_applied = false;
    int    fast_ref_plate = -1;
    double fast_ref_time = 0.0;
    double fast_ref_angle = 0.0;
    double fast_flight_time = 0.0;
    std::vector<float> fast_tolerance;
    if (fast_target && TOTAL > 0) {
        const double omega = predictor.target_omega;
        // 候选：每个返回点 × 每块未被屏蔽的板；中心瞄准夹角在该返回点的预测时刻测量
        int    best_i = -1, best_j = -1;
        double best_angle = 0.0;
        int    fallback_i = -1, fallback_j = -1;
        double fallback_angle = 0.0;
        double fallback_abs = -1.0;
        for (int i = 0; i < TOTAL; ++i) {
            const double t = items[(size_t)i].predict_time;   // 该返回点的总预测时间
            const PredictedBallisticSolver::PredictorResult pr = predictor.function(t);
            const cv::Vec3f center(pr.first.x, pr.first.y, pr.first.z);
            for (int j = 0; j < (int)pr.second.size(); ++j) {
                if (predictor.isIndexMasked(j)) continue;     // 屏蔽板不作为瞄准对象
                const cv::Point3f& p = pr.second[(size_t)j];
                const double a = centerAimAngle(center, cv::Vec3f(p.x, p.y, p.z), yaw_world_origin);
                if (a * omega <= 0.0) {
                    // 夹角与 ω 异号（含恰为 0）：板正在向枪线靠近，取 |夹角| 最小者
                    if (best_i < 0 || std::fabs(a) < std::fabs(best_angle)) {
                        best_i = i;
                        best_j = j;
                        best_angle = a;
                    }
                } else if (std::fabs(a) > fallback_abs) {
                    // 全部同号（板都在远离枪线）：退回取 |夹角| 最大者（最快绕回）
                    fallback_i = i;
                    fallback_j = j;
                    fallback_angle = a;
                    fallback_abs = std::fabs(a);
                }
            }
        }
        if (best_i < 0) {
            best_i = fallback_i;
            best_j = fallback_j;
            best_angle = fallback_angle;
        }

        if (best_i >= 0 && std::fabs(omega) > 1e-9) {
            // 对齐时刻 = 该点总预测时间 + |夹角|/|ω|
            const double t_new = items[(size_t)best_i].predict_time
                                 + std::fabs(best_angle) / std::fabs(omega);
            // 大小 yaw 构型：该时刻的大 yaw 关节角 → 有效 yaw 旋转中心；单 yaw 传 NaN
            const float yaw_big = in.big_small
                ? yawBigAtTime(in, t_new)
                : std::numeric_limits<float>::quiet_NaN();
            // 并行解算已结束，此处于调用线程用第一个解算器做单次解算
            // （solveSingle 不再迭代弹道飞行时间；GimbalSolver 解算只读内部树）
            const PredictedBallisticSolver::Result r =
                solvers_.front().solveSingle(predictor.function, best_j, t_new, yaw_big);
            if (r.success) {
                const Item it = makeActual(r);
                std::fill(items.begin(), items.end(), it);   // 整条序列 = 这次新解算

                // fast 元数据：参考夹角按 t_ref 时刻几何重算（与需求1 同一定义），
                // 每块板旋转容差角 = max(下限, fire_angle_length / 该板 xy 旋转半径)；
                // 旋转半径 = 方向向量（同刻预测中心 − 该板位置）的 xy 投影长度。
                const PredictedBallisticSolver::PredictorResult pr = predictor.function(t_new);
                const cv::Vec3f center(pr.first.x, pr.first.y, pr.first.z);
                if (best_j < (int)pr.second.size()) {
                    const cv::Point3f& p = pr.second[(size_t)best_j];
                    fast_ref_angle = centerAimAngle(center, cv::Vec3f(p.x, p.y, p.z),
                                                    yaw_world_origin);
                }
                fast_tolerance.assign(pr.second.size(), (float)min_rotation_tolerance_angle_);
                for (size_t j = 0; j < pr.second.size(); ++j) {
                    const cv::Point3f& p = pr.second[j];
                    const double radius = std::hypot((double)center[0] - (double)p.x,
                                                     (double)center[1] - (double)p.y);
                    if (radius > 1e-6) {
                        fast_tolerance[j] = (float)std::max(min_rotation_tolerance_angle_,
                                                            fire_angle_length_ / radius);
                    }
                    // 半径过小（退化）时保持下限值，避免容差角发散
                }
                fast_applied     = true;
                fast_ref_plate   = best_j;
                fast_ref_time    = t_new - predictor_age;   // 换算为“相对本次调用时刻”的秒数
                fast_flight_time = r.gimbal.flight_time;
            }
        }
    }

    // ── 5. 结果 ──
    Result res;
    res.items = std::move(items);
    res.valid = res.items.front().success;             // 第一个返回点 = 前导精确点（实际计算点）
    res.first_point = res.items.front().predicted_point;
    res.first_predict_time = res.items.front().predict_time;
    res.yaw_world_origin = yaw_world_origin;
    // 积分补偿开关：仅在预测有效且 MCU 自瞄开关打开时启用
    res.integral_enable = res.valid && in.auto_aim_switch;
    if (fast_applied) {
        res.fast_target            = true;
        res.fast_omega             = predictor.target_omega;
        res.fast_ref_time          = fast_ref_time;
        res.fast_ref_plate         = fast_ref_plate;
        res.fast_ref_center_aim_angle = fast_ref_angle;
        res.fast_flight_time       = fast_flight_time;
        res.fast_plate_tolerance   = std::move(fast_tolerance);
    }

    // ── 慢目标瞄准点滞回：记录本帧序列“第一个值”选中的瞄准点索引，作为下一帧
    // 第一个值的粘滞点（state_ 已在来源切换时整体清零；首点无效 → -1 = 不粘）。
    // 无论本帧是否启用滞回都记录（下一帧可能切换为慢目标需要基准）。
    state_.last_first_target_index =
        (res.valid && !res.items.empty() && res.items.front().target_index >= 0)
            ? res.items.front().target_index
            : -1;

    return res;
}

void SequencePredictor::invalidate()
{
    // 预测器不可用：自身跨帧状态与当前来源记录一并重置；
    // 下次 predict() 将视为新来源并重新初始化状态
    state_ = State{};
    active_source_ = PredictorSource{};
}

// fast_target 帧的火控点“有目标在枪线上”判定（需求5，匀速旋转模型）。
// 云台输出模式对火控数组的每个点调用：与 MPC 轨迹误差条件**同时满足**才开火。
//   - 非 fast_target 帧恒 true（不做门控，保持原行为）；
//   - 火控点 index 的开火时刻 → 命中时刻：
//       t_impact = extra_predict_time + (index+1)·dt_control + fast_flight_time
//     （索引时间与返回点索引同一时间轴：第 i 个返回点索引时刻 = extra+(i+1)·dt；
//      延迟取该点对应的弹道飞行时间 fast_flight_time）；
//   - 参考板在 fast_ref_time 时刻的中心瞄准夹角为 fast_ref_center_aim_angle，之后按
//     匀速旋转以 ω 演化；装甲板按返回列表顺序圆周均布（前哨站 3 块 120°、其它
//     4 块 90°），故第 m 块相对参考板的夹角偏置为 +m·2π/N；
//   - 任一板在命中时刻落在自身旋转容差角内即认为“有目标在枪线上”；
//   - fast 元数据缺失（参考板无效 / 板数为 0 / ω≈0）时返回 false（不开火，安全侧）。
bool SequencePredictor::fastGunLineOk(const Result& seq, int index,
                                      double extra_predict_time, double dt_control) {
    if (!seq.fast_target) return true;   // 非 fast_target：不追加枪线门控
    const int plates = (int)seq.fast_plate_tolerance.size();
    if (plates <= 0 || seq.fast_ref_plate < 0 || seq.fast_ref_plate >= plates) return false;
    const double omega = seq.fast_omega;
    if (std::fabs(omega) < 1e-9) return false;   // 无角速度信息：不满足匀速旋转模型

    // 命中时刻（秒，与 fast_ref_time 同一时间轴：相对 predict() 调用时刻）
    const double t_impact = extra_predict_time
                            + (double)(index + 1) * dt_control + seq.fast_flight_time;
    const double dt = t_impact - seq.fast_ref_time;   // 相对参考时刻的时间差

    const double plate_step = 2.0 * M_PI / (double)plates;
    for (int m = 0; m < plates; ++m) {
        const int plate = (seq.fast_ref_plate + m) % plates;   // 板圆周顺序 = 返回列表顺序
        // 该板在命中时刻的中心瞄准夹角（匀速旋转外推）+ 圆周偏置，解缠绕到 (-π, π]
        const double angle = std::remainder(
            seq.fast_ref_center_aim_angle + omega * dt + (double)m * plate_step, 2.0 * M_PI);
        if (std::fabs(angle) < (double)seq.fast_plate_tolerance[(size_t)plate]) return true;
    }
    return false;
}
