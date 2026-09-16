// fast_target_test.cpp — 验证 SequencePredictor 的“中心瞄准夹角 / 快目标 / 枪线”改造
//
// 覆盖内容（对应需求 1~5 与后续修订）：
//   1) 需求1：中心瞄准夹角 = 方向向量(中心−瞄准点) 与 中心瞄准向量(中心−yaw 轴旋转
//      中心) 的 xy 有向夹角；本测试用**独立实现**的几何公式复算夹角；
//   2) 需求2：第二对施密特阈值触发的 fast_target 锁存（越上阈值置位、跌破下阈值
//      复位、介于两者保持），用逐帧改变 ω 的序列验证锁存行为；
//   3) 需求3 + 修订：fast_target 帧**替换精确点的解算目标**——每个精确点用「包装
//      预测器」（瞄准点换成同 z、同 xy 半径、xy 落在『目标中心 → yaw 轴旋转中心』
//      连线线段上的**对齐位置**）调用**原迭代解算器**求解一次，再按各板各自的总预测
//      时间用**原预测器**算实际夹角选板；因此每个精确点的瞄准点必须**恰好落在该点
//      预测时刻的中心连线上**（中心平动已被计入），且中间点全部由插值补完（插值后
//      不再有任何弹道解算）；整段序列 target_index = 合成值 kFastAimTargetIndex(-2)；
//   4) 需求4：每块板的旋转容差角 = max(下限, fire_angle_length / 该板 xy 半径)，
//      半径取 t=0 值（同一索引的板半径不随时间改变）；
//   5) 需求5：fastGunLineOk 的“有目标在枪线上”判据与独立复算的匀速旋转模型一致；
//      非 fast_target 帧恒 true（不做门控）；
//   6) 修订：选板必须把**角度容差**算进去——“刚过枪线”（夹角与 ω 同号但 |夹角| 已在
//      该板容差内）同样可打；测试用掩膜只留两块板并扫描相位构造该局面校验。
//   另外覆盖：普通四装甲（4 块 90°）、前哨站（3 块 120°）、各板半径不同、**目标平动**
//   （验证中心平动被计入对齐位置）、大小 yaw 构型（配置 mode = big_small 时才跑该段）、
//   慢目标帧回归（逐点解算 + |夹角|>π/2 优先排除）。
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
    int M = 0, K = 0, n = 0;                      // 序列参数（精确点集合用）
    std::vector<int> solveIdx() const {
        std::vector<int> v;
        for (int i = 0; i < n; ++i) v.push_back(i);
        for (int j = 0; j < M; ++j) v.push_back(n + j * K);
        return v;
    }
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

// t=0 时各板的 xy 半径（= 实现里用于包装预测器/容差角的半径）
std::vector<double> plateRadii0(const Target& tgt) {
    const PredictedBallisticSolver::PredictorResult pr = tgt.at(0.0);
    const cv::Vec3f c(pr.first.x, pr.first.y, pr.first.z);
    std::vector<double> r(pr.second.size(), 0.0);
    for (size_t j = 0; j < pr.second.size(); ++j) {
        r[j] = std::hypot((double)c[0] - (double)pr.second[j].x,
                          (double)c[1] - (double)pr.second[j].y);
    }
    return r;
}

// 由“瞄准点到中心的 xy 距离 + z”识别该精确点选中的是哪块板（半径/高度各不相同时唯一）
int identifyPlate(const Target& tgt, double t_pred, const cv::Vec3f& aim,
                  const std::vector<double>& radii0, bool enabled) {
    if (!enabled) return -1;
    const PredictedBallisticSolver::PredictorResult pr = tgt.at(t_pred);
    const cv::Vec3f c(pr.first.x, pr.first.y, pr.first.z);
    const double d = std::hypot((double)aim[0] - (double)c[0], (double)aim[1] - (double)c[1]);
    for (int j = 0; j < (int)pr.second.size(); ++j) {
        if (std::fabs(d - radii0[(size_t)j]) > 2e-3) continue;              // 半径匹配
        if (std::fabs((double)aim[2] - (double)pr.second[(size_t)j].z) > 2e-3) continue;  // z 匹配
        return j;
    }
    return -1;
}

// ── 一次快目标帧的完整校验 ──
// identify：各板半径/高度是否足以从瞄准点反推选中的板（全相同时无法区分，跳过选板校验）
// check_geom：是否复核“命中时刻几何判据”（目标平动时匀速旋转模型与几何有偏差，可关）
void checkFastCase(SequencePredictor& sp, const Env& env, Target tgt, bool identify,
                   bool check_geom, const char* tag) {
    const int N = tgt.plates();
    const double omega = tgt.omega;
    const std::vector<double> radii0 = plateRadii0(tgt);
    std::printf("\n[fast_target] %s：%s，N=%d 板、ω=%.2f rad/s、vel=(%.1f,%.1f,%.1f)\n",
                env.name.c_str(), tag, N, omega, tgt.vel[0], tgt.vel[1], tgt.vel[2]);
    const SequencePredictor::Result res = env.predict(makePredictor(tgt, omega, true, env.ts));
    check(res.valid, "预测结果有效");
    check(res.fast_target, "fast_target = true");
    check((int)res.fast_plate_tolerance.size() == N, "每块板都有旋转容差角");
    check(res.fast_ref_plate >= 0 && res.fast_ref_plate < N, "参考板索引有效");

    const int TOTAL = (env.M - 1) * env.K + 1 + env.n;
    check((int)res.items.size() == TOTAL, "总返回点数 = (M-1)*K+1+n（未变）");
    bool idx_ok = !res.items.empty() && SequencePredictor::kFastAimTargetIndex < 0 &&
                  SequencePredictor::kFastAimTargetIndex != -1;
    for (const auto& it : res.items) {
        if (it.target_index != SequencePredictor::kFastAimTargetIndex) idx_ok = false;
    }
    check(idx_ok, "整段序列 target_index = 合成值 kFastAimTargetIndex（非原生板序号、非 -1）");

    // ── 精确点：瞄准点必须落在该点预测时刻的 中心→yaw原点 连线上（中心平动已计入）──
    const std::vector<int> exact = env.solveIdx();
    const cv::Vec3f origin = res.yaw_world_origin;
    int on_line = 0, lead_ok = 0, rule_ok = 0, rule_checked = 0;
    double worst_on_line = 0.0, min_lead = 1e9;
    for (int u : exact) {
        const SequencePredictor::Item& it = res.items[(size_t)u];
        if (!it.success) continue;
        const PredictedBallisticSolver::PredictorResult pr = tgt.at(it.predict_time);
        const cv::Vec3f c(pr.first.x, pr.first.y, pr.first.z);
        const double a_aim = ang(c, it.predicted_point, origin);
        if (std::fabs(a_aim) < 1e-4) ++on_line;
        worst_on_line = std::max(worst_on_line, std::fabs(a_aim));
        // 提前量：预测时刻应晚于该精确点的索引时刻（= extra + (索引+1)·dt）
        const double t_index = env.ext + (double)(u + 1) * env.dt;
        if (it.predict_time > t_index) ++lead_ok;
        min_lead = std::min(min_lead, it.predict_time - t_index);
        // 选板规则（可识别时）：选中板在该点总预测时间上的实际夹角必须“可打”
        const int j = identifyPlate(tgt, it.predict_time, it.predicted_point, radii0, identify);
        if (j >= 0) {
            ++rule_checked;
            const cv::Point3f& p = pr.second[(size_t)j];
            const double a = ang(c, cv::Vec3f(p.x, p.y, p.z), origin);
            const double tol = (double)res.fast_plate_tolerance[(size_t)j];
            if (a * omega <= 0.0 || std::fabs(a) <= tol) ++rule_ok;
        }
    }
    check(on_line == (int)exact.size(),
          "每个精确点瞄准点都落在该点预测时刻的中心连线上（含中心平动）");
    std::printf("      瞄准点偏离中心线最大 %.2e rad；最小提前量 %.4f s\n",
                worst_on_line, min_lead);
    check(lead_ok == (int)exact.size(), "每个精确点的预测时刻都晚于自身索引时刻（有提前量）");
    if (identify) {
        check(rule_checked == (int)exact.size(), "全部精确点都能由瞄准点反推出选中的板");
        check(rule_ok == rule_checked, "选中的板都满足“异号靠近 或 |夹角| 已在容差内”");
    }

    // ── 插值点：全部由左右精确点的线性插值补完（无解算）──
    bool interp_ok = true;
    for (size_t i = 0; i < res.items.size(); ++i) {
        bool is_exact = false;
        for (int u : exact) if ((size_t)u == i) { is_exact = true; break; }
        if (is_exact) continue;
        int lo = -1, hi = -1;
        for (int u : exact) {
            if ((size_t)u < i) lo = std::max(lo, u);
            if (u > (int)i && (hi < 0 || u < hi)) hi = u;
        }
        if (lo < 0 || hi < 0) continue;
        const SequencePredictor::Item& A = res.items[(size_t)lo];
        const SequencePredictor::Item& B = res.items[(size_t)hi];
        const double t = (double)((int)i - lo) / (double)(hi - lo);
        const cv::Vec3f want = A.predicted_point + (float)t * (B.predicted_point - A.predicted_point);
        const double want_t = A.predict_time + t * (B.predict_time - A.predict_time);
        if (cv::norm(res.items[i].predicted_point - want) > 1e-4) interp_ok = false;
        if (std::fabs(res.items[i].predict_time - want_t) > 1e-9) interp_ok = false;
    }
    check(interp_ok, "中间点 = 左右精确点线性插值（未做任何解算）");

    // ── 元数据/需求4：容差角（半径取 t=0 值）；参考夹角 = 该板在 t_ref 的实际夹角 ──
    const double t_ref = res.items.front().predict_time;
    check(std::fabs(res.fast_ref_time - t_ref) < 1e-9,
          "参考时刻 = 序列第一个精确点的预测时刻（predictor_age = 0）");
    check(std::fabs(res.fast_flight_time - res.items.front().flight_time) < 1e-12,
          "参考延迟 = 第一个精确点的弹道飞行时间");
    {
        bool tol_ok = true;
        double tol_min = 1e9, tol_max = 0.0;
        for (int j = 0; j < N; ++j) {
            const double want = std::max(env.min_tol, env.fire_len / radii0[(size_t)j]);
            if (std::fabs((double)res.fast_plate_tolerance[(size_t)j] - want) > 1e-6) tol_ok = false;
            tol_min = std::min(tol_min, (double)res.fast_plate_tolerance[(size_t)j]);
            tol_max = std::max(tol_max, (double)res.fast_plate_tolerance[(size_t)j]);
        }
        check(tol_ok, "各板旋转容差角 = max(下限, fire_angle_length/该板 t=0 半径)");
        std::printf("      容差角范围 [%.4f, %.4f] rad\n", tol_min, tol_max);
        const PredictedBallisticSolver::PredictorResult pr = tgt.at(t_ref);
        const cv::Vec3f c(pr.first.x, pr.first.y, pr.first.z);
        const cv::Point3f& p = pr.second[(size_t)res.fast_ref_plate];
        const double a_ref = ang(c, cv::Vec3f(p.x, p.y, p.z), origin);
        check(std::fabs(res.fast_ref_center_aim_angle - a_ref) < 1e-6,
              "fast_ref_center_aim_angle = 参考板在 t_ref 的实际夹角（需求5 起始相位）");
        const double tol = (double)res.fast_plate_tolerance[(size_t)res.fast_ref_plate];
        check(a_ref * omega <= 0.0 || std::fabs(a_ref) <= tol,
              "参考板满足“异号靠近 或 |夹角| 已在容差内”（含紧过枪线情形）");
        std::printf("      t_ref=%.4f s，参考板 %d 夹角 %.4f rad（容差 %.4f rad，ω=%.2f）\n",
                    t_ref, res.fast_ref_plate, a_ref, tol, omega);
    }

    // ── 需求5：枪线判定与独立复算的匀速旋转模型逐点一致 ──
    bool model_ok = true, geom_ok = true;
    int windows = 0, geom_bad = 0;
    const int KT = 60;
    for (int k = 0; k < KT; ++k) {
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
        if (check_geom) {
            const PredictedBallisticSolver::PredictorResult pr = tgt.at(t_impact);
            const cv::Vec3f c(pr.first.x, pr.first.y, pr.first.z);
            bool geom = false;
            for (int j = 0; j < N; ++j) {
                const cv::Point3f& p = pr.second[(size_t)j];
                if (std::fabs(ang(c, cv::Vec3f(p.x, p.y, p.z), origin)) <
                    (double)res.fast_plate_tolerance[(size_t)j]) geom = true;
            }
            if (got != geom) { geom_ok = false; ++geom_bad; }
        }
    }
    check(model_ok, "与独立复算的匀速旋转模型逐点一致");
    if (check_geom) check(geom_ok, "与命中时刻几何判据逐点一致");
    check(windows > 0 && windows < KT, "开火窗口是周期性的（既非恒真也非恒假）");
    std::printf("      %d 个火控点中 枪线满足 %d 个（板周期 2π/(%d·|ω|)=%.3f s，dt=%.3f s）%s\n",
                KT, windows, N, 2.0 * M_PI / ((double)N * std::fabs(omega)), env.dt,
                geom_bad ? "（几何判据偏差数 %d，见注释）" : "");
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
                "slow=[%.2f,%.2f] fast=[%.2f,%.2f] M=%d K=%d n=%d\n",
                mode == YawMode::SINGLE ? "single" : "big_small",
                dt, ext, fire_len, min_tol, slow_lo, slow_hi, fast_lo, fast_hi,
                cfg.common.predictSequence.predictionPoints,
                cfg.common.predictSequence.interpolationRefine,
                cfg.common.predictSequence.exactLeadPoints);

    SequencePredictor sp;
    const auto kNow = std::chrono::steady_clock::now();   // predict 与快照时间戳共用同一时刻

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
    env.M = cfg.common.predictSequence.predictionPoints;
    env.K = cfg.common.predictSequence.interpolationRefine;
    env.n = cfg.common.predictSequence.exactLeadPoints;

    // ── 普通四装甲（各板半径/高度相同：半径相同无法反推板号，跳过选板校验）──
    Target tgt4;
    tgt4.center = cv::Vec3f(6.0f, 0.8f, 0.3f);
    tgt4.radius = {0.25, 0.25, 0.25, 0.25};
    tgt4.dz     = {0.0, 0.0, 0.0, 0.0};
    tgt4.phase0 = 0.3;
    tgt4.omega  = 6.0;
    checkFastCase(sp, env, tgt4, /*identify=*/false, /*check_geom=*/true, "等半径静止");

    // ── 普通四装甲（半径/高度不同 r1/r2，负角速度：可反推板号 → 校验选板规则）──
    Target tgt4b = tgt4;
    tgt4b.radius = {0.22, 0.28, 0.22, 0.28};
    tgt4b.dz     = {0.0, 0.05, 0.0, 0.05};
    tgt4b.omega  = -5.0;
    checkFastCase(sp, env, tgt4b, true, true, "r1/r2 静止");

    // ── 前哨站：3 块板 120°，各面半径/高度不同 ──
    Target tgt3 = tgt4;
    tgt3.radius = {0.32, 0.36, 0.32};
    tgt3.dz     = {0.0, 0.04, -0.05};
    tgt3.omega  = 4.5;
    checkFastCase(sp, env, tgt3, true, true, "前哨站 3 面");

    // ── 目标平动（本次修订的关键：对齐位置必须落在**当前预测中心**的连线上）──
    Target tgtmv = tgt4b;
    tgtmv.vel   = cv::Vec3f(2.0f, -1.0f, 0.0f);   // 斜向平移 2.24 m/s
    tgtmv.omega = 5.5;
    checkFastCase(sp, env, tgtmv, true, false, "平动 2.24 m/s");

    // ── 修订：选板把角度容差算进去（“刚过枪线仍可打”）──
    // 只留 0/1 两块板（夹角相差 90°）：扫描相位，构造“其中一块刚好过枪线（夹角与 ω 同号
    // 且 |夹角| ≤ 容差）、另一块同号且远”的局面——老规则（只看异号）会落到退路，
    // 新规则应选刚过枪线的那块。
    std::printf("\n[选板含角度容差] 掩膜 {2,3}，扫描相位构造“刚过枪线”\n");
    {
        bool seen_just_passed = false;
        int  just_passed_runs = 0, just_passed_ok = 0, scan_runs = 0, scan_ok = 0;
        const std::vector<int> mask = {2, 3};
        for (int probe = 0; probe < 13; ++probe) {
            Target t = tgt4b;
            t.omega = 5.0;                                    // ω > 0
            t.phase0 = 0.3 + (double)probe * (M_PI / 12.0);    // 每次挪 15°
            const SequencePredictor::Result r =
                env.predict(makePredictor(t, t.omega, true, env.ts, mask));
            if (!r.fast_target || r.fast_ref_plate < 0) continue;
            const PredictedBallisticSolver::PredictorResult pr = t.at(r.fast_ref_time);
            const cv::Vec3f c(pr.first.x, pr.first.y, pr.first.z);
            const cv::Point3f& p = pr.second[(size_t)r.fast_ref_plate];
            const double a = ang(c, cv::Vec3f(p.x, p.y, p.z), r.yaw_world_origin);
            const double tol = (double)r.fast_plate_tolerance[(size_t)r.fast_ref_plate];
            const bool same_sign = (a * t.omega > 0.0);
            ++scan_runs;
            if (a * t.omega <= 0.0 || std::fabs(a) <= tol) ++scan_ok;   // 每帧选中板都可打
            if (same_sign && std::fabs(a) <= tol) {
                // 选中的是“刚过枪线”的板（同号且已在容差内）：老规则（只取异号，退路取
                // 同号 |夹角| 最大）绝不会选它 → 校验另一块板确实会被老规则优先
                ++just_passed_runs;
                const int other = (r.fast_ref_plate + 1) % 2;
                const cv::Point3f& q = pr.second[(size_t)other];
                const double aq = ang(c, cv::Vec3f(q.x, q.y, q.z), r.yaw_world_origin);
                const bool other_opposite = (aq * t.omega < 0.0);              // 老规则的首选
                const bool other_same_sign_farther =
                    (aq * t.omega > 0.0) && std::fabs(aq) > std::fabs(a);      // 老规则的退路
                if (other_opposite || other_same_sign_farther) ++just_passed_ok;
                seen_just_passed = true;
                std::printf("      φ0=%.3f：选中板 %d 夹角 %.4f（同号、容差 %.4f）；"
                            "另一板 %d 夹角 %.4f（异号=%d，同号更远=%d）\n",
                            t.phase0, r.fast_ref_plate, a, tol, other, aq,
                            (int)other_opposite, (int)other_same_sign_farther);
            }
        }
        check(scan_runs > 0 && scan_ok == scan_runs, "扫描各帧选中的板都可打（异号 或 容差内）");
        check(seen_just_passed, "扫描中出现了“选中板刚好过枪线（同号且 |夹角| ≤ 容差）”的帧");
        check(just_passed_runs > 0 && just_passed_ok == just_passed_runs,
              "该帧选中的正是刚过枪线的那块板（老规则会选另一块）");
    }

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
        bool any_diff = false, no_synth = true;
        for (const auto& it : r.items) {
            if (it.predicted_point != r.items.front().predicted_point) any_diff = true;
            if (it.target_index == SequencePredictor::kFastAimTargetIndex) no_synth = false;
        }
        check(any_diff, "慢目标：仍是逐点解算序列（非快目标对齐点）");
        check(no_synth, "慢目标：target_index 仍为原生板序号（未用合成索引）");
        bool violated = false;
        for (const auto& it : r.items) {
            const PredictedBallisticSolver::PredictorResult pr = slow.at(it.predict_time);
            const cv::Vec3f c(pr.first.x, pr.first.y, pr.first.z);
            if (std::fabs(ang(c, it.predicted_point, r.yaw_world_origin)) <= M_PI / 2.0) continue;
            for (int j = 0; j < (int)pr.second.size(); ++j) {
                const cv::Point3f& q = pr.second[(size_t)j];
                if (std::fabs(ang(c, cv::Vec3f(q.x, q.y, q.z), r.yaw_world_origin)) <= M_PI / 2.0) {
                    violated = true;
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
