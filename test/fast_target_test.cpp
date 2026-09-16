// fast_target_test.cpp — 验证 SequencePredictor 的“中心瞄准夹角 / 快目标 / 枪线”改造
//
// 覆盖内容（对应需求 1~5）：
//   1) 需求1：中心瞄准夹角 = 方向向量(中心−瞄准点) 与 中心瞄准向量(中心−yaw 轴旋转
//      中心) 的 xy 有向夹角；本测试用**独立实现**的几何公式复算全部 (返回点 × 板)
//      的夹角，并据此校验需求3 选出的瞄准点与对齐时刻；
//   2) 需求2：第二对施密特阈值触发的 fast_target 锁存（越上阈值置位、跌破下阈值
//      复位、介于两者保持），用逐帧改变 ω 的序列验证锁存行为；
//   3) 需求3：fast_target 帧整条返回序列 = 一次新的单点解算（对“夹角与 ω 异号且
//      |夹角| 最小的板”，预测时刻 = 该返回点总预测时间 + |夹角|/|ω|）；并检查该
//      瞄准点在 t_ref 时刻确实位于 yaw 原点→目标中心连线上（|夹角| ≈ 0）；
//   4) 需求4：每块板的旋转容差角 = max(下限, fire_angle_length / 该板 xy 半径)
//      （各板半径/高度可不同：普通四装甲 r1/r2、前哨站 3 面）；
//   5) 需求5：fastGunLineOk 的“有目标在枪线上”判据与独立复算的匀速旋转模型一致，
//      且与“命中时刻几何上某块板的中心瞄准夹角落在该板容差内”一致（目标不平移
//      时两者等价）；非 fast_target 帧恒 true（不做门控）。
//   另外覆盖：普通四装甲（4 块 90°）、前哨站（3 块 120°）、各板半径不同、
//   大小 yaw 构型（当前配置 mode = big_small 时才跑该段，用大 yaw 预测序列取
//   有效 yaw 旋转中心）。
//
// 编译与运行（在项目根目录执行；复用主程序已编译产物，仅把 main.cpp.o 换成本测试）：
//   DEFS=$(sed -n 's/^CXX_DEFINES = //p'  build/CMakeFiles/unified_auto_aim.dir/flags.make)
//   INCS=$(sed -n 's/^CXX_INCLUDES = //p' build/CMakeFiles/unified_auto_aim.dir/flags.make)
//   CXXF=$(sed -n 's/^CXX_FLAGS = //p'    build/CMakeFiles/unified_auto_aim.dir/flags.make)
//   mkdir -p build/ftt
//   /usr/bin/c++ $DEFS $INCS $CXXF -c test/fast_target_test.cpp -o build/ftt/fast_target_test.o
//   cd build
//   LINK=$(cat CMakeFiles/unified_auto_aim.dir/link.txt)
//   LINK=${LINK/CMakeFiles\/unified_auto_aim.dir\/src\/main.cpp.o/ftt\/fast_target_test.o}
//   LINK=${LINK/-o ..\/bin\/unified_auto_aim/-o ftt\/fast_target_test}
//   eval "$LINK" && cd .. && ./build/ftt/fast_target_test
// （需先 make 一次主程序，保证 build/CMakeFiles/unified_auto_aim.dir 下有最新对象文件；
//   默认的 COMMON_SOURCES 列表不含本测试，本文件不参与主程序构建。）
//
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "common/Ballistic/SequencePredictor.h"
#include "common/RobotConfig.h"
#include "tcs/RobotController.h"

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what);
    if (!ok) ++g_fail;
}

// 独立实现的中心瞄准夹角（xy 有向夹角：方向向量相对中心瞄准向量逆时针为正）
double ang(const cv::Vec3f& center, const cv::Vec3f& point, const cv::Vec3f& origin) {
    const double bx = center[0] - point[0], by = center[1] - point[1];
    const double ax = center[0] - origin[0], ay = center[1] - origin[1];
    return std::atan2(ax * by - ay * bx, ax * bx + ay * by);
}

// 合成目标预测器（装甲板圆周均布；p = c + r_j[sin(yaw_i), -cos(yaw_i)]，
// yaw_i = phase + i*2π/N，与 ClassEKF 同一几何与符号约定）——ω 为正时世界系逆时针。
struct Target {
    cv::Vec3f center;
    cv::Vec3f vel{0.0f, 0.0f, 0.0f};
    std::vector<double> radius{0.25, 0.25, 0.25, 0.25};   // 各板 xy 半径（可不同）
    std::vector<double> dz{0.0, 0.0, 0.0, 0.0};           // 各板高度偏置
    double phase0 = 0.0;
    double omega = 0.0;     // rad/s（与 Armor EKF 同一符号约定）
    int    plates() const { return (int)radius.size(); }

    PredictedBallisticSolver::PredictorResult at(double t) const {
        const cv::Vec3f c = center + cv::Vec3f((float)(vel[0] * t), (float)(vel[1] * t),
                                               (float)(vel[2] * t));
        const double phase = phase0 + omega * t;
        const double step = 2.0 * M_PI / (double)plates();
        std::vector<cv::Point3f> pts;
        for (int i = 0; i < plates(); ++i) {
            const double y = phase + (double)i * step;
            pts.emplace_back((float)(c[0] + radius[(size_t)i] * std::sin(y)),
                             (float)(c[1] - radius[(size_t)i] * std::cos(y)),
                             (float)(c[2] + dz[(size_t)i]));
        }
        return {cv::Point3f(c[0], c[1], c[2]), pts};
    }
};

// 一次测试的运行环境（单 yaw / 大小 yaw 两种 predict 入口）
struct Env {
    std::string name;
    std::function<SequencePredictor::Result(const SequencePredictor::Predictor&)> predict;
    double ext = 0.0, dt = 0.0, fire_len = 0.0, min_tol = 0.0;
    std::chrono::steady_clock::time_point ts{};   // 当帧时刻（predict 与快照共用）
};

SequencePredictor::Predictor makePredictor(const Target& tgt, double omega, bool omega_valid,
                                           const std::chrono::steady_clock::time_point& ts,
                                           std::vector<int> masked = {}) {
    SequencePredictor::Predictor p;
    p.function = [tgt](double t) { return tgt.at(t); };
    p.source = SequencePredictor::PredictorSource::armor(0);   // label 0（普通四装甲）
    p.timestamp = ts;   // 必须 == predict() 的调用时刻（predictor_age = 0），否则时间轴整体平移
    p.masked_indices = std::move(masked);
    p.target_omega = omega;
    p.omega_valid = omega_valid;
    return p;
}

// ── 一次快目标帧的完整校验（需求1/3/4/5）──
void checkFastCase(SequencePredictor& sp, const Env& env, Target tgt) {
    const int N = tgt.plates();
    const double omega = tgt.omega;
    std::printf("\n[fast_target] %s：N=%d 板、ω=%.2f rad/s\n", env.name.c_str(), N, omega);
    const SequencePredictor::Result res = env.predict(makePredictor(tgt, omega, true, env.ts));
    check(res.valid, "预测结果有效");
    check(res.fast_target, "fast_target = true");
    check((int)res.fast_plate_tolerance.size() == N, "每块板都有旋转容差角");
    check(res.fast_ref_plate >= 0 && res.fast_ref_plate < N, "参考板索引有效");

    bool all_same = !res.items.empty();
    for (const auto& it : res.items) {
        if (it.predicted_point != res.items.front().predicted_point ||
            it.yaw != res.items.front().yaw || it.pitch != res.items.front().pitch ||
            it.target_index != res.items.front().target_index ||
            it.predict_time != res.items.front().predict_time) {
            all_same = false;
            break;
        }
    }
    check(all_same, "整条返回序列 = 同一次新解算（全部 item 相同）");

    // 独立复算：候选时间取自同一几何、同一 ω 但 omega_valid = false 的参考运行
    // （fast 帧返回序列已被整体替换，逐点时间只在该参考运行里可见）
    const SequencePredictor::Result ref = env.predict(makePredictor(tgt, omega, false, env.ts));
    const cv::Vec3f origin = res.yaw_world_origin;
    int    exp_i = -1, exp_j = -1;
    double exp_a = 0.0;
    int    fb_i = -1, fb_j = -1;
    double fb_a = 0.0, fb_abs = -1.0;
    for (int i = 0; i < (int)ref.items.size(); ++i) {
        const double t = ref.items[(size_t)i].predict_time;
        const PredictedBallisticSolver::PredictorResult pr = tgt.at(t);
        const cv::Vec3f c(pr.first.x, pr.first.y, pr.first.z);
        for (int j = 0; j < N; ++j) {
            const cv::Point3f& p = pr.second[(size_t)j];
            const double a = ang(c, cv::Vec3f(p.x, p.y, p.z), origin);
            if (a * omega <= 0.0) {   // 夹角与 ω 异号（含恰为 0）：正在向枪线靠近
                if (exp_i < 0 || std::fabs(a) < std::fabs(exp_a)) { exp_i = i; exp_j = j; exp_a = a; }
            } else if (std::fabs(a) > fb_abs) {   // 全部同号：退路取 |夹角| 最大者
                fb_i = i; fb_j = j; fb_a = a; fb_abs = std::fabs(a);
            }
        }
    }
    if (exp_i < 0) { exp_i = fb_i; exp_j = fb_j; exp_a = fb_a; }
    check(exp_j == res.fast_ref_plate, "选中的板 = 异号且 |中心瞄准夹角| 最小的板");
    const double t_ref_want = ref.items[(size_t)exp_i].predict_time +
                              std::fabs(exp_a) / std::fabs(omega);
    check(std::fabs(res.fast_ref_time - t_ref_want) < 1e-6,
          "参考时刻 = 该点总预测时间 + |夹角|/|ω|");

    // 新解算瞄准点 = t_ref 时刻参考板位置；其夹角 ≈ 0（板已转到枪线上）
    const PredictedBallisticSolver::PredictorResult pr_ref = tgt.at(res.fast_ref_time);
    const cv::Point3f& p_ref = pr_ref.second[(size_t)res.fast_ref_plate];
    const cv::Vec3f c_ref(pr_ref.first.x, pr_ref.first.y, pr_ref.first.z);
    check(cv::norm(res.items.front().predicted_point - cv::Vec3f(p_ref.x, p_ref.y, p_ref.z)) < 1e-4,
          "瞄准点 = t_ref 时刻参考板位置");
    const double a_ref_calc = ang(c_ref, cv::Vec3f(p_ref.x, p_ref.y, p_ref.z), origin);
    check(std::fabs(a_ref_calc) < 1e-3, "t_ref 时刻参考板位于 yaw 原点→中心连线上（夹角≈0）");
    check(std::fabs(res.fast_ref_center_aim_angle - a_ref_calc) < 1e-3,
          "Result::fast_ref_center_aim_angle 与该时刻几何夹角一致");

    // 需求4：每块板容差角 = max(下限, fire_angle_length / 该板半径)；各板不同半径时容差不同
    bool tol_ok = true;
    double min_seen = 1e9, max_seen = 0.0;
    for (int j = 0; j < N; ++j) {
        const cv::Point3f& p = pr_ref.second[(size_t)j];
        const double r = std::hypot((double)c_ref[0] - (double)p.x, (double)c_ref[1] - (double)p.y);
        const double want = std::max(env.min_tol, env.fire_len / r);
        if (std::fabs((double)res.fast_plate_tolerance[(size_t)j] - want) > 1e-6) tol_ok = false;
        min_seen = std::min(min_seen, (double)res.fast_plate_tolerance[(size_t)j]);
        max_seen = std::max(max_seen, (double)res.fast_plate_tolerance[(size_t)j]);
    }
    check(tol_ok, "各板旋转容差角 = max(下限, fire_angle_length/该板半径)");
    std::printf("      容差角范围 [%.4f, %.4f] rad\n", min_seen, max_seen);

    // 需求5：枪线判定与独立复算一致（模型 + 命中时刻几何）
    bool model_ok = true, geom_ok = true;
    int windows = 0;
    const int K = 60;
    for (int k = 0; k < K; ++k) {
        const bool got = SequencePredictor::fastGunLineOk(res, k, env.ext, env.dt);
        if (got) ++windows;
        const double t_impact = env.ext + (double)(k + 1) * env.dt + res.fast_flight_time;
        const double dt = t_impact - res.fast_ref_time;
        bool want = false;
        for (int m = 0; m < N; ++m) {
            const int plate = (res.fast_ref_plate + m) % N;
            const double off = std::remainder(res.fast_ref_center_aim_angle +
                                              res.fast_omega * dt +
                                              (double)m * 2.0 * M_PI / (double)N, 2.0 * M_PI);
            if (std::fabs(off) < (double)res.fast_plate_tolerance[(size_t)plate]) want = true;
        }
        if (got != want) model_ok = false;
        // 几何复核（目标不平移时与线性模型等价）：命中时刻是否有板落在自身容差内
        const PredictedBallisticSolver::PredictorResult pr = tgt.at(t_impact);
        const cv::Vec3f c(pr.first.x, pr.first.y, pr.first.z);
        bool geom = false;
        for (int j = 0; j < N; ++j) {
            const cv::Point3f& p = pr.second[(size_t)j];
            if (std::fabs(ang(c, cv::Vec3f(p.x, p.y, p.z), origin)) <
                (double)res.fast_plate_tolerance[(size_t)j]) geom = true;
        }
        if (got != geom) geom_ok = false;
    }
    check(model_ok, "与独立复算的匀速旋转模型逐点一致");
    check(geom_ok, "与命中时刻几何判据逐点一致");
    check(windows > 0 && windows < K, "开火窗口是周期性的（既非恒真也非恒假）");
    std::printf("      %d 个火控点中 枪线满足 %d 个（板周期 2π/(%d·|ω|)=%.3f s，dt=%.3f s）\n",
                K, windows, N, 2.0 * M_PI / ((double)N * std::fabs(omega)), env.dt);
}

tcs::RobotController::State makeState(double bullet_velocity) {
    tcs::RobotController::State st;
    st.strict.yaw_pos = 0.0;
    st.strict.pitch_angle = 0.0;
    st.strict.imu_euler_yaw = 0.0;
    st.strict.imu_euler_pitch = 0.0;
    st.strict.imu_euler_roll = 0.0;
    st.strict.chassis_yaw = 0.0;
    st.strict.chassis_pitch = 0.0;
    st.strict.chassis_roll = 0.0;
    st.mcu.valid = true;
    st.mcu.bullet_velocity = bullet_velocity;
    st.mcu.auto_aim_switch = 1;
    return st;
}

} // namespace

int main() {
    const RobotConfig& cfg = RobotConfig::instance();
    const double dt        = cfg.common.dtControl();
    const double ext       = cfg.common.predictedBallistic.extraPredictTime;
    const double fire_len  = cfg.common.predictSequence.fireAngleLength;
    const double min_tol   = cfg.common.predictSequence.minRotationToleranceAngle;
    const double fast_lo   = cfg.armor.targetSelection.fastAngularVelocityLower;
    const double fast_hi   = cfg.armor.targetSelection.fastAngularVelocityUpper;
    const double slow_lo   = cfg.armor.targetSelection.slowAngularVelocityLower;
    const double slow_hi   = cfg.armor.targetSelection.slowAngularVelocityUpper;
    const YawMode mode     = cfg.common.bigSmallYaw.mode;
    std::printf("config: mode=%s dt=%.4f extra=%.4f fire_len=%.3f min_tol=%.4f "
                "slow=[%.2f,%.2f] fast=[%.2f,%.2f]\n",
                mode == YawMode::SINGLE ? "single" : "big_small",
                dt, ext, fire_len, min_tol, slow_lo, slow_hi, fast_lo, fast_hi);

    SequencePredictor sp;
    const auto kNow = std::chrono::steady_clock::now();   // predict 与快照时间戳共用同一时刻

    // ── 调用环境（按当前构型选 predict 入口）──
    Env env;
    tcs::RobotController::State st = makeState(25.0);
    bsy::RobotState bst;
    if (mode == YawMode::SINGLE) {
        env.name = "单 yaw 构型";
        env.predict = [&](const SequencePredictor::Predictor& p) { return sp.predict(st, p, kNow); };
    } else {
        env.name = "大小 yaw 构型";
        bst.valid = true;
        bst.bullet_velocity = 25.0;
        bst.auto_aim_switch = 1;
        bst.yaw_big_joint = 0.0;
        bst.yaw_small_joint = 0.0;
        bst.pitch_joint = 0.0;
        bst.yaw_big_azimuth = 0.0;
        bst.yaw_small_azimuth = 0.0;
        bst.chassis_azimuth = 0.0;
        bst.info_chassis_yaw = 0.0;
        bst.info_chassis_pitch = 0.0;
        bst.info_chassis_roll = 0.0;
        // MPC 预测大 yaw 世界方位角序列（缓慢转动的斜坡），供逐点 θ_big / 有效 yaw 中心
        bst.pred_big_azimuth_seq.assign(80, 0.0);
        for (size_t i = 0; i < bst.pred_big_azimuth_seq.size(); ++i) {
            bst.pred_big_azimuth_seq[i] = 0.001 * (double)i;
        }
        env.predict = [&](const SequencePredictor::Predictor& p) { return sp.predict(bst, p, kNow); };
    }
    env.ext = ext;
    env.ts = kNow;
    env.dt = dt;
    env.fire_len = fire_len;
    env.min_tol = min_tol;

    // ── 1~5：普通四装甲（各板半径相同）──
    Target tgt4;
    tgt4.center = cv::Vec3f(6.0f, 0.8f, 0.3f);
    tgt4.radius = {0.25, 0.25, 0.25, 0.25};
    tgt4.dz     = {0.0, 0.0, 0.0, 0.0};
    tgt4.phase0 = 0.3;
    tgt4.omega  = 6.0;
    checkFastCase(sp, env, tgt4);

    // ── 普通四装甲（各板半径/高度不同：ClassEKF 的 r1/r2 情形）──
    Target tgt4b = tgt4;
    tgt4b.radius = {0.22, 0.28, 0.22, 0.28};
    tgt4b.dz     = {0.0, 0.05, 0.0, 0.05};
    tgt4b.omega  = -5.0;      // 负角速度：验证符号约定
    checkFastCase(sp, env, tgt4b);

    // ── 前哨站：3 块板 120°，各面半径/高度不同 ──
    Target tgt3 = tgt4;
    tgt3.radius = {0.32, 0.36, 0.32};
    tgt3.dz     = {0.0, 0.04, -0.05};
    tgt3.omega  = 4.5;
    checkFastCase(sp, env, tgt3);

    // ── 需求2：施密特锁存（逐帧改变 ω）──
    std::printf("\n[施密特锁存] slow=[%.2f,%.2f] fast=[%.2f,%.2f]\n", slow_lo, slow_hi, fast_lo, fast_hi);
    struct Step { double w; bool want_fast; const char* note; };
    const Step steps[] = {
        {fast_hi + 1.0, true,  "越过 fast 上阈值 → 进入 fast_target"},
        {0.5 * (fast_lo + fast_hi), true, "介于 fast 两阈值之间 → 保持（未跌破下阈值）"},
        {0.5 * fast_lo, false, "跌破 fast 下阈值 → 退出 fast_target"},
        {fast_hi + 2.0, true,  "再次越过上阈值 → 重新进入 fast_target"},
    };
    Target latch = tgt4;
    for (const Step& s : steps) {
        latch.omega = s.w;
        const SequencePredictor::Result r = env.predict(makePredictor(latch, s.w, true, env.ts));
        std::printf("      ω=%.3f → fast_target=%d（期望 %d）%s\n",
                    s.w, (int)r.fast_target, (int)s.want_fast, s.note);
        check(r.fast_target == s.want_fast, s.note);
    }
    {
        latch.omega = 20.0;
        const SequencePredictor::Result r = env.predict(makePredictor(latch, 20.0, false, env.ts));
        check(!r.fast_target, "ω 不可用时 fast_target = false（不做枪线门控）");
        check(SequencePredictor::fastGunLineOk(r, 0, ext, dt), "非 fast_target 帧枪线判定恒 true");
    }

    // ── 需求3（非 fast 分支）：仍为逐点解算，且优先排除 |中心瞄准夹角| > π/2 ──
    std::printf("\n[非 fast 帧] ω=0.2（慢目标）\n");
    {
        Target slow = tgt4;
        slow.omega = 0.2;
        const SequencePredictor::Result r = env.predict(makePredictor(slow, 0.2, true, env.ts));
        check(r.valid && !r.fast_target, "慢目标：valid 且非 fast_target");
        bool any_diff = false;
        for (const auto& it : r.items) {
            if (it.predicted_point != r.items.front().predicted_point) any_diff = true;
        }
        check(any_diff, "慢目标：仍是逐点解算序列（未被单点解算替换）");
        bool violated = false;
        for (const auto& it : r.items) {
            const PredictedBallisticSolver::PredictorResult pr = slow.at(it.predict_time);
            const cv::Vec3f c(pr.first.x, pr.first.y, pr.first.z);
            if (std::fabs(ang(c, it.predicted_point, r.yaw_world_origin)) <= M_PI / 2.0) continue;
            for (int j = 0; j < (int)pr.second.size(); ++j) {
                const cv::Point3f& q = pr.second[(size_t)j];
                if (std::fabs(ang(c, cv::Vec3f(q.x, q.y, q.z), r.yaw_world_origin)) <= M_PI / 2.0) {
                    violated = true;   // 存在 <=π/2 的板却被跳过
                    break;
                }
            }
            if (violated) break;
        }
        check(!violated, "非 fast：优先不考虑 |中心瞄准夹角| > π/2 的瞄准点");
    }

    std::printf("\n%s（失败项 %d）\n", g_fail == 0 ? "全部通过" : "存在失败项", g_fail);
    return g_fail == 0 ? 0 : 1;
}
