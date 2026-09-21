#ifndef AIM_SWITCH_LOG_H
#define AIM_SWITCH_LOG_H

// AimSwitchLog.h — 瞄准点（挡板 / 靶点）切换日志（调试用，默认开启，无需配置项）
//
// 用途：每帧记录最终选中的瞄准点索引，在索引发生变化（切换）时打印一行带绝对
//       时间与切换间隔的 SWITCH 记录；全部帧同时写入 CSV，供离线统计切换频率、
//       对齐"切换时刻"与"决策器/预测器重置时刻"，定位频繁切换的原因
//       （粘滞断开 / 预测器无效导致决策器重置 / 旋转计数索引漂移）。
//
// 控制台事件：
//   [SWITCH]     选靶索引发生变化（含距上次切换的间隔、累计切换次数、切换前无效帧数）
//   [PRED_OFF]   本帧无可用目标预测器（main 会调 invalidate()，决策器状态被清空）
//   [PRED_ON]    预测器恢复可用（此后需重新建立观测，可能改选另一块靶）
//   [NO_RESULT]  预测器可用但本帧没选出可用目标（无候选 / 全被屏蔽且决策返回 -1）
//   [RESET_*]    决策器 / 预测器状态重置（由 SequencePredictor 经 event() 发出）
//   [PLAN]       本帧下发的预瞄点计划（逐节拍的目标索引 / fire 许可 / 预瞄点高度 +
//                控制器真正执行到了哪一拍），默认只在切换帧打印，见 plan()
//
// 环境变量（均可省略）：
//   AIM_SWITCH_LOG=0        关闭（默认开启）
//   AIM_SWITCH_LOG_ALL=1    控制台逐帧输出（含逐帧 [PLAN]；默认只在切换与事件时输出）
//   AIM_SWITCH_LOG_DIR=...  输出目录（默认 <项目根>/logs）
//
// 输出文件（同名目录下，同一时间戳）：
//   aim_switch_<时间戳>.csv  每帧一行的选靶/切换/事件记录（见 frame()）
//   aim_plan_<时间戳>.csv    每帧每节拍一行的预瞄点计划（见 plan()），
//                            列含：节拍号、目标索引、预瞄点 xyz、下发的 yaw/pitch、
//                            fire 许可、控制器已执行到的节拍
//
// 说明：纯头文件实现（inline + 函数局部静态单例），不新增源文件与配置项；
//       状态更新与写文件加锁保护，弹道线程与事件调用方共用同一实例。

#include <sys/stat.h>
#include <sys/types.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

class AimSwitchLog {
public:
    /// 一帧的最终选靶结果（弹道线程每帧调用 frame() 一次）
    struct Frame {
        std::chrono::steady_clock::time_point frame_timestamp;  // 该帧流水线时间戳
        bool        predictor_valid = false;   // 本帧是否有可用目标预测器
        bool        result_valid    = false;   // SequencePredictor 返回结果是否有效
        std::string source;                    // "PowerRune" / "Armor" / "None"
        int         selected_index  = -1;      // 本帧选中的瞄准点索引（-1 = 无）
        double      target_z        = 0.0;     // 选中瞄准点的预测世界 z（米）
        double      flight_time     = 0.0;     // 选中瞄准点的弹道飞行时间（秒）
        std::string context;                   // 附加信息（能量机关：亮靶/屏蔽/跳变等）
    };

    static AimSwitchLog& instance() {
        static AimSwitchLog inst;   // 函数局部静态：C++11 起初始化线程安全
        return inst;
    }

    bool enabled() const { return enabled_; }
    bool csvOpen() const { return csv_.is_open(); }
    const std::string& csvPath() const { return csv_path_; }
    const std::string& planPath() const { return plan_path_; }

    // ==================== 预瞄点计划（每帧每节拍一行） ====================
    //
    // 行 = 控制器节拍 tick（与发给控制器的 yaw 序列下标一一对应）：
    //   yaw_sent / pitch_sent / fire_sent 是**该节拍实际下发**的值。注意 pitch 与
    //   fire 序列在发送前被截掉了前 pitch_lead / fire_lead 个（见 GimbalOutput），
    //   而 yaw 序列不截取 —— 所以控制器第 k 拍应用的是「第 k 点的 yaw + 第 k+lead
    //   点的 pitch + 第 k+lead 点的 fire」。本日志把两者都记下来，便于看出这一点。
    struct PlanPoint {
        int    tick         = -1;     // 控制器节拍下标（= yaw 序列下标）
        bool   has_point    = false;  // 该节拍是否有对应预瞄点（= items 范围内）
        bool   sent         = false;  // 该节拍的 yaw / pitch 是否在本次下发序列范围内
        int    target_index = -1;     // 该节拍预瞄点瞄准的目标索引（-1 = 无）
        bool   success      = false;  // 该节拍预瞄点解算是否成功
        double x = 0, y = 0, z = 0;   // 预瞄点（world 系，米）
        double gimbal_yaw = 0, gimbal_pitch = 0;  // 预瞄点的原始关节角（未加偏置）
        double flight_time = 0;       // 预瞄点的弹道飞行时间（秒）
        double yaw_sent = 0, pitch_sent = 0;      // 该节拍实际下发的 yaw / pitch
        int    fire_plan = -1;        // 该节拍预瞄点的 fire 判定（MPC ref vs pred）：1/0/-1
        int    fire_sent = -1;        // 该节拍实际下发的 fire 位：1/0/-1（-1 = 未下发）
    };

    struct Plan {
        std::chrono::steady_clock::time_point frame_timestamp;  // 该帧时间戳
        std::string source;            // "PowerRune" / "Armor" / "None"
        bool sequence_sent = false;    // 本帧是否真的下发了序列（保持模式为 false）
        std::vector<PlanPoint> ticks;  // 逐节拍明细（长度 = 下发序列长度）
        int  yaw_lead = 0, pitch_lead = 0, fire_lead = 0;  // 三段序列的截取偏移
        unsigned long long ticks_since_set = 0;  // MPC 自上次 set() 起已消费的节拍数
        int  exec_tick = -1;           // 真正执行到的节拍（= ticks_since_set - 1，-1 = 未知）
        double actual_yaw = 0, actual_pitch = 0;  // 实测关节角（弧度）
        bool switched = false;         // 与上一帧的首个目标索引不同（跨帧切换）
    };

    /// 每帧一次：记录逐节拍的预瞄点计划；切换帧或 AIM_SWITCH_LOG_ALL=1 时打 [PLAN]
    void plan(const Plan& p);

    /// 记录本帧选靶原因（由决策器 PredictedPointSelector::select 调用，下一次
    /// frame() 取用后清空）：如 sticky / z-lowest/prev-no-solution / none
    void setReason(const std::string& r) {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lk(mtx_);
        pending_reason_ = r;
    }

    /// 每帧一次：索引变化时输出 SWITCH（含切换间隔与累计次数）；
    /// predictor_valid 翻转、预测器可用但无结果等事件也会输出。
    void frame(const Frame& f) {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lk(mtx_);

        ++frame_count_;
        const double steady_s   = steadySecs(std::chrono::steady_clock::now());
        const double frame_ts_s = steadySecs(f.frame_timestamp);
        const double frame_gap_s =
            (last_frame_steady_s_ > 0.0) ? (steady_s - last_frame_steady_s_) : -1.0;
        last_frame_steady_s_ = steady_s;

        // ── 事件分类：只在"两个都有效且不同"时才算真正的切换 ──
        //   与 -1（无目标）之间的变化单列为 LOST / REACQUIRE，避免把预测器掉线
        //   造成的目标丢失混进切换统计。
        enum class Ev { NONE, SWITCH, LOST, REACQUIRE };
        Ev ev = Ev::NONE;
        if (has_prev_index_) {
            if (prev_index_ >= 0 && f.selected_index >= 0) {
                if (f.selected_index != prev_index_) ev = Ev::SWITCH;
            } else if (prev_index_ >= 0 && f.selected_index < 0) {
                ev = Ev::LOST;
            } else if (prev_index_ < 0 && f.selected_index >= 0) {
                ev = Ev::REACQUIRE;
            }
        }
        double switch_gap_s = -1.0;
        // 距上次切换的时间（用于 SWITCH / LOST / REACQUIRE 三类行）
        double since_switch_s = (last_switch_steady_s_ > 0.0)
                                    ? (steady_s - last_switch_steady_s_)
                                    : -1.0;
        if (ev == Ev::SWITCH) {
            ++switch_count_;
            switch_gap_s = since_switch_s;
            last_switch_steady_s_ = steady_s;
            if (first_switch_steady_s_ < 0.0) first_switch_steady_s_ = steady_s;
        } else if (ev == Ev::LOST) {
            ++lost_count_;
        } else if (ev == Ev::REACQUIRE) {
            ++reacquire_count_;
        }
        // 统计"距上次切换以来的预测器无效帧数"：该值 >0 说明切换前预测器掉过线
        if (!f.predictor_valid) ++invalid_since_switch_;

        const int  prev_index      = prev_index_;
        const bool prev_valid      = prev_valid_;
        const bool has_prev_valid  = has_prev_valid_;
        const int  invalid_between = invalid_since_switch_;
        const std::string reason   = pending_reason_;
        pending_reason_.clear();

        const char* tag = (ev == Ev::SWITCH)    ? "SWITCH"
                          : (ev == Ev::LOST)    ? "LOST"
                          : (ev == Ev::REACQUIRE) ? "REACQUIRE"
                                                  : "FRAME";
        const bool switched = (ev == Ev::SWITCH);
        writeRow(tag, steady_s, frame_ts_s, f, prev_index, switched ? 1 : 0,
                 ev == Ev::NONE ? -1.0 : since_switch_s, frame_gap_s, invalid_between, reason);

        // ── 控制台：切换 / 丢失 / 重新捕获 ──
        if (switched) {
            std::string gap = (switch_gap_s < 0.0) ? "首次" : fmtDouble(switch_gap_s) + "s";
            std::printf("[SWITCH] %s  frame#%lld  t=%.3fs  idx %d -> %d  间隔=%s (第%lld次)",
                        wallTime().c_str(), frame_count_, steady_s, prev_index,
                        f.selected_index, gap.c_str(), switch_count_);
            if (frame_gap_s >= 0.0) std::printf("  帧间隔=%.3fs", frame_gap_s);
            std::printf("  |  %s  pred=%d res=%d", src(f).c_str(), f.predictor_valid ? 1 : 0,
                        f.result_valid ? 1 : 0);
            if (invalid_between > 0) std::printf("  切换前无效帧=%d", invalid_between);
            if (!reason.empty()) std::printf("  reason=%s", reason.c_str());
            if (!f.context.empty()) std::printf("  |  %s", f.context.c_str());
            std::printf("\n");
            std::fflush(stdout);
        }
        if (ev == Ev::LOST) {
            std::printf("[LOST] %s  frame#%lld  t=%.3fs  目标丢失 idx %d -> -1  距上次切换=%.3fs",
                        wallTime().c_str(), frame_count_, steady_s, prev_index, since_switch_s);
            if (invalid_between > 0) std::printf("  预测器无效帧=%d", invalid_between);
            if (!f.context.empty()) std::printf("  |  %s", f.context.c_str());
            std::printf("\n");
            std::fflush(stdout);
        } else if (ev == Ev::REACQUIRE) {
            std::printf("[REACQUIRE] %s  frame#%lld  t=%.3fs  重新捕获 idx=%d  距上次切换=%.3fs",
                        wallTime().c_str(), frame_count_, steady_s, f.selected_index,
                        since_switch_s);
            if (invalid_between > 0) std::printf("  预测器无效帧=%d", invalid_between);
            if (!f.context.empty()) std::printf("  |  %s", f.context.c_str());
            std::printf("\n");
            std::fflush(stdout);
        }
        // ── 控制台：预测器可用性翻转 ──
        if (has_prev_valid && prev_valid != f.predictor_valid) {
            const char* tag = f.predictor_valid ? "PRED_ON" : "PRED_OFF";
            std::printf("[%s] %s  frame#%lld  t=%.3fs  predictor_valid %d -> %d",
                        tag, wallTime().c_str(), frame_count_, steady_s,
                        prev_valid ? 1 : 0, f.predictor_valid ? 1 : 0);
            if (!f.context.empty()) std::printf("  |  %s", f.context.c_str());
            std::printf("\n");
            std::fflush(stdout);
            writeEventRow(tag, steady_s, frame_ts_s, f);
        }
        // ── 控制台：预测器可用但本帧无有效结果（无候选 / 全被屏蔽）──
        if (f.predictor_valid && !f.result_valid) {
            std::printf("[NO_RESULT] %s  frame#%lld  t=%.3fs  idx=%d",
                        wallTime().c_str(), frame_count_, steady_s, f.selected_index);
            if (!f.context.empty()) std::printf("  |  %s", f.context.c_str());
            std::printf("\n");
            std::fflush(stdout);
            writeEventRow("NO_RESULT", steady_s, frame_ts_s, f);
        }
        if (console_all_ && ev == Ev::NONE) {
            std::printf("[FRAME] %s  frame#%lld  idx=%d  |  %s\n", wallTime().c_str(),
                        frame_count_, f.selected_index, reason.c_str());
            std::fflush(stdout);
        }

        prev_index_     = f.selected_index;
        has_prev_index_ = true;
        prev_valid_     = f.predictor_valid;
        has_prev_valid_ = true;
        last_source_    = src(f);
        if (csv_.is_open()) csv_.flush();
    }

    /// 事件记录（如决策器 / 预测器重置）：tag 写事件列，detail 写上下文列
    void event(const std::string& tag, const std::string& detail) {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lk(mtx_);
        const auto now = std::chrono::steady_clock::now();
        const double steady_s = steadySecs(now);
        std::printf("[%s] %s  t=%.3fs  %s\n", tag.c_str(), wallTime().c_str(), steady_s,
                    detail.c_str());
        std::fflush(stdout);
        Frame f;
        f.frame_timestamp = now;
        f.context = detail;
        f.selected_index = prev_index_;
        f.predictor_valid = prev_valid_;
        f.source = last_source_;
        writeEventRow(tag, steady_s, steady_s, f);
        if (csv_.is_open()) csv_.flush();
    }

private:
    AimSwitchLog() { init(); }

        ~AimSwitchLog() {
        if (!enabled_) return;
        if (frame_count_ > 0) {
            std::printf("[SWITCH-LOG] 共 %lld 帧: 切换 %lld 次, 目标丢失 %lld 次, 重新捕获 %lld 次",
                        frame_count_, switch_count_, lost_count_, reacquire_count_);
            if (switch_count_ > 1 && first_switch_steady_s_ > 0.0 &&
                last_switch_steady_s_ > first_switch_steady_s_) {
                std::printf("；平均切换间隔 %.3fs",
                            (last_switch_steady_s_ - first_switch_steady_s_) /
                                double(switch_count_ - 1));
            }
            if (csv_.is_open()) std::printf("，日志: %s", csv_path_.c_str());
            if (plan_csv_.is_open()) {
                std::printf("，预瞄点计划(%lld 帧): %s", plan_frame_count_, plan_path_.c_str());
            }
            std::printf("\n");
        }
    }

    AimSwitchLog(const AimSwitchLog&) = delete;
    AimSwitchLog& operator=(const AimSwitchLog&) = delete;

    // ==================== 初始化 ====================
    void init() {
        const char* off = std::getenv("AIM_SWITCH_LOG");
        enabled_ = !(off != nullptr && std::string(off) == "0");
        const char* all = std::getenv("AIM_SWITCH_LOG_ALL");
        console_all_ = (all != nullptr && std::string(all) != "0");
        if (!enabled_) return;

        std::string dir;
        if (const char* d = std::getenv("AIM_SWITCH_LOG_DIR")) {
            dir = d;
        } else {
#ifdef PROJECT_ROOT
            dir = std::string(PROJECT_ROOT) + "/logs";
#else
            dir = "logs";
#endif
        }
        ::mkdir(dir.c_str(), 0755);   // 目录已存在时忽略错误

        csv_path_ = dir + "/aim_switch_" + fileTimestamp() + ".csv";
        csv_.open(csv_path_, std::ios::out | std::ios::trunc);
        plan_path_ = dir + "/aim_plan_" + fileTimestamp() + ".csv";
        plan_csv_.open(plan_path_, std::ios::out | std::ios::trunc);
        std::printf("[SWITCH-LOG] 选靶切换日志已开启");
        if (csv_.is_open()) {
            csv_ << "wall_time,steady_s,frame_ts_s,event,source,predictor_valid,result_valid,"
                    "prev_index,selected_index,switched,switch_gap_s,frame_gap_s,"
                    "invalid_since_switch,target_z,flight_time,reason,context\n";
            csv_ << std::fixed;
            csv_.precision(6);
            std::printf("：%s", csv_path_.c_str());
        } else {
            std::printf("（文件不可写，仅控制台输出）");
        }
        if (plan_csv_.is_open()) {
            // 预瞄点计划：每帧每节拍一行（tick = 下发给控制器的序列下标）
            plan_csv_ << "frame,wall_time,steady_s,frame_ts_s,source,sequence_sent,tick,"
                         "has_point,target_index,success,x,y,z,gimbal_yaw,gimbal_pitch,"
                         "flight_time,yaw_sent,pitch_sent,sent,fire_plan,fire_sent,"
                         "ticks_since_set,exec_tick,exec_now,actual_yaw,actual_pitch,switched,"
                         "yaw_lead,pitch_lead,fire_lead\n";
            plan_csv_ << std::fixed;
            plan_csv_.precision(6);
            std::printf("\n            预瞄点计划：%s", plan_path_.c_str());
        }
        std::printf("\n            关闭：AIM_SWITCH_LOG=0    逐帧输出：AIM_SWITCH_LOG_ALL=1\n");
        std::fflush(stdout);
    }

    // ==================== CSV 写入 ====================
    void writeRow(const char* event, double steady_s, double frame_ts_s, const Frame& f,
                  int prev_index, int switched, double switch_gap_s, double frame_gap_s,
                  int invalid_between, const std::string& reason) {
        if (!csv_.is_open()) return;
        csv_ << wallTime() << ',' << steady_s << ',' << frame_ts_s << ',' << event << ','
             << src(f) << ',' << (f.predictor_valid ? 1 : 0) << ',' << (f.result_valid ? 1 : 0)
             << ',' << prev_index << ',' << f.selected_index << ',' << switched << ','
             << switch_gap_s << ',' << frame_gap_s << ',' << invalid_between << ','
             << f.target_z << ',' << f.flight_time << ',' << csvSafe(reason) << ','
             << csvSafe(f.context) << '\n';
    }

    void writeEventRow(const std::string& tag, double steady_s, double frame_ts_s,
                       const Frame& f) {
        if (!csv_.is_open()) return;
        csv_ << wallTime() << ',' << steady_s << ',' << frame_ts_s << ',' << tag << ','
             << src(f) << ',' << (f.predictor_valid ? 1 : 0) << ',' << (f.result_valid ? 1 : 0)
             << ',' << prev_index_ << ',' << f.selected_index << ',' << 0 << ',' << -1.0 << ','
             << -1.0 << ',' << invalid_since_switch_ << ',' << f.target_z << ','
             << f.flight_time << ',' << ',' << csvSafe(f.context) << '\n';
    }

    // ==================== 工具 ====================
    static std::string src(const Frame& f) { return f.source.empty() ? "None" : f.source; }

    static double steadySecs(const std::chrono::steady_clock::time_point& tp) {
        return std::chrono::duration<double>(tp.time_since_epoch()).count();
    }

    static std::string wallTime() {
        using namespace std::chrono;
        const auto now = system_clock::now();
        const auto t   = system_clock::to_time_t(now);
        const long ms  = long(duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000);
        std::tm tm{};
        localtime_r(&t, &tm);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03ld", tm.tm_hour, tm.tm_min,
                      tm.tm_sec, ms);
        return buf;
    }

    static std::string fileTimestamp() {
        using namespace std::chrono;
        const auto t = system_clock::to_time_t(system_clock::now());
        std::tm tm{};
        localtime_r(&t, &tm);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02d", tm.tm_year + 1900,
                      tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
        return buf;
    }

    static std::string fmtDouble(double v) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << v;
        return oss.str();
    }

    /// CSV 字段安全：上下文含逗号（如 present=[1,4]）时统一替换为分号
    static std::string csvSafe(std::string s) {
        for (char& c : s) {
            if (c == ',') c = ';';
        }
        return s;
    }

    // ==================== 状态 ====================
    bool enabled_     = true;
    bool console_all_ = false;
    std::mutex mtx_;
    std::ofstream csv_;
    std::string csv_path_;
    std::ofstream plan_csv_;
    std::string plan_path_;

    long long frame_count_  = 0;
    long long plan_frame_count_ = 0;
    long long switch_count_ = 0;
    long long lost_count_   = 0;
    long long reacquire_count_ = 0;
    int  prev_index_        = -1;
    bool has_prev_index_    = false;
    bool prev_valid_        = false;
    bool has_prev_valid_    = false;
    int  invalid_since_switch_ = 0;
    std::string pending_reason_;   // 本帧选靶原因（决策器写入，frame() 取用后清空）
    double last_frame_steady_s_  = -1.0;
    double last_switch_steady_s_ = -1.0;
    double first_switch_steady_s_ = -1.0;
    std::string last_source_ = "None";
    int  plan_prev_first_target_ = -1;   // 上一帧计划的首个目标索引（跨帧切换判定）
    bool has_plan_prev_ = false;
};

// ============================================================================
// plan() 定义（预瞄点计划：每帧每节拍一行 CSV + 切换/逐帧模式的 [PLAN] 控制台输出）
// ============================================================================
inline void AimSwitchLog::plan(const Plan& p) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lk(mtx_);
    ++plan_frame_count_;

    // 跨帧切换判定：比较"首个有预瞄点的节拍"对应的目标索引
    int first_target = -1;
    for (const PlanPoint& t : p.ticks) {
        if (t.has_point) {
            first_target = t.target_index;
            break;
        }
    }
    const bool switched = has_plan_prev_ && plan_prev_first_target_ >= 0 &&
                          first_target >= 0 && plan_prev_first_target_ != first_target;
    plan_prev_first_target_ = first_target;
    has_plan_prev_ = true;

    const double steady_s   = steadySecs(std::chrono::steady_clock::now());
    const double frame_ts_s = steadySecs(p.frame_timestamp);
    const std::string source = p.source.empty() ? "None" : p.source;

    // ── CSV：每节拍一行（无对应数值的字段留空，便于离线按 NaN 处理）──
    if (plan_csv_.is_open()) {
        auto num = [](std::ostream& os, bool ok, double v) {
            if (ok) os << v;
        };
        for (const PlanPoint& t : p.ticks) {
            plan_csv_ << plan_frame_count_ << ',' << wallTime() << ',' << steady_s << ','
                      << frame_ts_s << ',' << source << ','
                      << (p.sequence_sent ? 1 : 0) << ',' << t.tick << ','
                      << (t.has_point ? 1 : 0) << ',';
            if (t.has_point) {
                plan_csv_ << t.target_index;
            }
            plan_csv_ << ',' << (t.success ? 1 : 0) << ',';
            num(plan_csv_, t.has_point, t.x);   plan_csv_ << ',';
            num(plan_csv_, t.has_point, t.y);   plan_csv_ << ',';
            num(plan_csv_, t.has_point, t.z);   plan_csv_ << ',';
            num(plan_csv_, t.has_point, t.gimbal_yaw);   plan_csv_ << ',';
            num(plan_csv_, t.has_point, t.gimbal_pitch); plan_csv_ << ',';
            num(plan_csv_, t.has_point, t.flight_time);  plan_csv_ << ',';
            num(plan_csv_, t.sent, t.yaw_sent);   plan_csv_ << ',';
            num(plan_csv_, t.sent, t.pitch_sent); plan_csv_ << ',';
            plan_csv_ << (t.sent ? 1 : 0) << ',';
            if (t.fire_plan >= 0) plan_csv_ << t.fire_plan;
            plan_csv_ << ',';
            if (t.fire_sent >= 0) plan_csv_ << t.fire_sent;
            plan_csv_ << ',' << p.ticks_since_set << ',' << p.exec_tick << ','
                      << (t.tick == p.exec_tick ? 1 : 0) << ','
                      << p.actual_yaw << ',' << p.actual_pitch << ','
                      << (switched ? 1 : 0) << ',' << p.yaw_lead << ',' << p.pitch_lead << ','
                      << p.fire_lead << '\n';
        }
        plan_csv_.flush();
    }

    // ── 控制台：切换帧（或 AIM_SWITCH_LOG_ALL=1）打印逐节拍表 ──
    if (!(switched || console_all_)) return;
    auto fireChar = [](int v) -> const char* {
        return (v > 0) ? "   #" : (v == 0 ? "   ." : "   -");
    };
    std::string l_tick = "   tick    :", l_tgt = "   target  :", l_plan = "   fire判定:",
                l_sent = "   fire下发:", l_x = "   aim x   :", l_z = "   aim z   :",
                l_exec = "   exec    :", l_chg = "   序列内目标变化:";
    int  last_tgt = -2;
    bool has_tgt_change = false;
    char buf[48];
    for (const PlanPoint& t : p.ticks) {
        std::snprintf(buf, sizeof(buf), "%4d", t.tick);
        l_tick += buf;
        if (t.has_point) {
            std::snprintf(buf, sizeof(buf), "%4d", t.target_index);
            l_tgt += buf;
            std::snprintf(buf, sizeof(buf), "%7.2f", t.x);
            l_x += buf;
            std::snprintf(buf, sizeof(buf), "%7.2f", t.z);
            l_z += buf;
            // 序列内部的目标变化（相邻节拍瞄的目标不同）也标出来
            if (last_tgt != -2 && t.target_index != last_tgt) {
                std::snprintf(buf, sizeof(buf), " [tick %d: %d->%d]", t.tick, last_tgt,
                              t.target_index);
                l_chg += buf;
                has_tgt_change = true;
            }
            last_tgt = t.target_index;
        } else {
            l_tgt += "   -";
            l_x += "      -";
            l_z += "      -";
        }
        l_plan += fireChar(t.fire_plan);
        l_sent += fireChar(t.fire_sent);
        l_exec += (t.tick == p.exec_tick) ? "   ^" : "    ";
    }
    std::printf("[PLAN] %s  frame#%lld  t=%.3fs  %s  seq_sent=%d  ticks_since_set=%llu  "
                "exec_tick=%d  actual(yaw=%.3f pitch=%.3f)  switched=%d  "
                "lead(yaw/pitch/fire)=%d/%d/%d\n",
                wallTime().c_str(), plan_frame_count_, steady_s, source.c_str(),
                p.sequence_sent ? 1 : 0, p.ticks_since_set, p.exec_tick, p.actual_yaw,
                p.actual_pitch, switched ? 1 : 0, p.yaw_lead, p.pitch_lead, p.fire_lead);
    std::printf("%s\n%s\n%s\n%s\n%s\n%s\n%s\n", l_tick.c_str(), l_tgt.c_str(), l_plan.c_str(),
                l_sent.c_str(), l_x.c_str(), l_z.c_str(), l_exec.c_str());
    if (has_tgt_change) std::printf("%s\n", l_chg.c_str());
    std::fflush(stdout);
}

#endif  // AIM_SWITCH_LOG_H
