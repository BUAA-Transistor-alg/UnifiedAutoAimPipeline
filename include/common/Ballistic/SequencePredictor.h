// SequencePredictor.h — 预测序列通用类（预测云台控制序列 + 瞄准点序列）
//
// 基于 PredictedBallisticSolver：对目标预测函数生成预测云台控制序列与对应的
// 瞄准点序列，并把最新结果返回给调用方：
//   - GimbalOutput    使用预测云台控制序列（yaw/pitch，已含底盘修正与 yaw/pitch 偏置），
//                     自行计算 fire 序列并截取后发送；
//   - VisualizeOutput 使用瞄准点序列中的第一个值绘制预测瞄准点。
// 结果不再存放在本类的"最新槽"供输出模式跨线程读取：predict() 每帧返回 Result，
// 由 main 弹道线程写入当帧 OutputContext（含瞄准点 / 云台序列 / yaw 系原点），
// 沿"弹道 → 云台 → 可视化"级联逐级转发给输出模式（输出模式不再持有本类引用）。
//
// 目标预测器以 Predictor 结构体传入（而非仅 std::function）：内含预测函数、
// 目标预测器来源标注（PredictorSource，见下）、快照时间戳（predictor_timestamp）
// 与目标屏蔽索引列表（masked_indices，见 Predictor）。
// PredictedBallisticSolver::solve 已不再参与目标（瞄准点）选择：它对预测函数
// 返回列表中的每个目标点独立求解并返回全部结果；实际目标选择由本类 predict()
// 完成——依据来源标注自动选择目标选择策略（Armor → NEAREST、PowerRune →
// LOWEST_Z），并在每个实际计算点的求解结果之间按该策略选出该点使用的目标
// （masked_indices 中索引对应的瞄准点不参与选择）。同时 predict() 维护自身
// 跨帧状态 State（当前暂为空，预留）：target_predictor 来源切换或
// invalidate() 时重置。
//
// 序列生成（config common.predict_sequence）：
//   - 原划分：只精确解算 prediction_points（M）个实际计算点，时间间隔
//     K*dt_control；相邻实际计算点之间按 interpolation_refine（K）细分，
//     线性插值推出 K-1 个插值点；原划分序列点数 = (M-1)*K + 1，时间间隔
//     恰为 dt_control，首末点必为实际计算点；
//   - 相邻实际计算点选取的目标不同时：用左侧点与再上一个点（前一个精确
//     同目标点）之间的线性差值参数外推；若左侧点与再上一个点也目标不同或
//     没有再上一个点，直接复制左侧点的值。
//   - exact_lead_points（n，0 表示关闭）：在序列最前面拼接 n 个逐点精确
//     弹道解算的前导点（不使用插值）。返回序列 = [前导精确点 0..n-1] +
//     [原划分序列]，总返回点数 = (M-1)*K + 1 + n，时间间隔全程均匀为
//     dt_control；精确解算点共 n + M 个（n 个前导点 + M 个原划分实际计算点）。
//     外推参考的例外：原划分首段（紧邻窗口 n+1..n+K-1）需要外推时，参考点
//     改用前导精确区最后一个点 items[n-1]（要求与段左端点目标相同，否则
//     复制左端点）；其余段仍用原规则（上一实际计算点 items[a-K]）。
//
// 任何情况下（无论输出模式）由 main 每帧调用 predict()，保证当帧瞄准点
// 序列可写入 OutputContext。
#ifndef SEQUENCE_PREDICTOR_H
#define SEQUENCE_PREDICTOR_H

#include <atomic>
#include <chrono>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

#include "RobotController.h"
#include "common/TaskPool.h"
#include "common/Ballistic/GimbalSolver.h"
#include "common/Ballistic/PredictedBallisticSolver.h"

class SequencePredictor {
public:
    // 目标预测器来源标注：标识 predict() 所用 target_predictor 的目标来源。
    // 用作 predict() 内部目标选择策略（NEAREST/LOWEST_Z）与自身跨帧状态
    // （State）重置的判断依据：
    //   - Armor 流水线：每种物体（类别 label 0~8，见 ArmorInfer.h 类别映射）
    //     各自算一种来源 —— armor_label 不同即来源不同；
    //   - PowerRune 流水线：整体算一种来源。
    struct PredictorSource {
        enum class Kind { NONE = 0, ARMOR, POWER_RUNE };
        Kind kind = Kind::NONE;
        int  armor_label = -1;   // kind == ARMOR：物体类别（0~8）；否则无效

        static PredictorSource armor(int label) { return {Kind::ARMOR, label}; }
        static PredictorSource powerRune() { return {Kind::POWER_RUNE, -1}; }

        bool operator==(const PredictorSource& o) const {
            return kind == o.kind && armor_label == o.armor_label;
        }
        bool operator!=(const PredictorSource& o) const { return !(*this == o); }
    };

    // 目标预测器（predict() 输入）：原预测函数 + 来源标注 + 快照时间戳 +
    // 目标屏蔽索引列表（两条流水线在输出结果时组装本结构并存于 PipelineResult，
    // main 弹道线程直接传入 predict()）
    struct Predictor {
        PredictedBallisticSolver::Predictor function;   // 原预测函数 std::vector<cv::Point3f>(double)（world 系）
        PredictorSource source;                         // 来源标注
        std::chrono::steady_clock::time_point timestamp;  // predictor_timestamp：
                                                          // 产生该预测器快照的那一帧的时间戳（dt 零点）
        // 目标屏蔽索引列表：位于本列表中的索引（对应当前预测函数返回列表中
        // 瞄准点的下标，即 PredictedBallisticSolver::Result::target_index）对应
        // 的瞄准点不参与目标选择。须保证屏蔽后至少还有一个瞄准点未被屏蔽：
        // 若预测函数返回的全部瞄准点都被屏蔽（全被屏蔽，无点可选），
        // predict() 自动转为调用 invalidate() 并返回无效结果（等同无可用预测器，
        // 输出模式进入保持模式）。
        std::vector<int> masked_indices;

        // 本帧目标是否为“慢目标”（Armor 侧按目标角速度的施密特触发器判定后随
        // 预测器一起传下，见 ArmorPipelineData::Stage5Data.slow_target）。仅当其为
        // true 时 predict() 才启用瞄准点（板）滞回；PowerRune 等无角速度属性的
        // 来源恒为 false（默认不滞回，维持每点独立选最优的原有行为）。
        bool slow_target = false;

        /// 目标索引 index 是否被屏蔽（即位于 masked_indices 中）
        bool isIndexMasked(int index) const {
            for (int m : masked_indices) {
                if (m == index) return true;
            }
            return false;
        }
    };

    // 单个序列返回点（云台控制值 + 瞄准点；实际计算点或插值/外推/复制生成）
    struct Item {
        bool   success = false;
        cv::Vec3f predicted_point;   // 瞄准点（world 系）
        double predict_time = 0.0;   // 预测时间 = 额外预测时间 + 飞行时间（秒）
        float  yaw = 0.0f;           // 云台解算 yaw（已叠加底盘修正与 yaw 偏置）
        float  pitch = 0.0f;         // 云台解算 pitch（已叠加 pitch 偏置）
        float  gimbal_yaw = 0.0f;    // 解算器原始云台 yaw（相对底盘关节角，未叠加底盘修正/偏置；
                                     // 供可视化复现"需要的云台位姿"使用）
        float  gimbal_pitch = 0.0f;  // 解算器原始云台 pitch（未叠加偏置）
        double flight_time = 0.0;    // 弹道飞行时间（秒）
        int    target_index = -1;    // 该点对应的目标索引（实际计算点为选中目标，插值/外推继承左侧）
    };

    // 预测结果：预测云台控制序列 + 瞄准点序列
    struct Result {
        bool valid = false;          // 首个返回点（实际计算点）解算是否有效
        bool integral_enable = false;  // 本帧 yaw 力矩积分补偿是否启用：
                                       // 仅当 valid 且 st.mcu.auto_aim_switch == 1 时为 true
        std::vector<Item> items;     // 总返回点数 = (M-1)*K + 1 + n（前导精确点 + 原划分序列）
        cv::Vec3f first_point;       // 瞄准点序列第一个值（可视化用）
        double first_predict_time = 0.0;
        cv::Vec3f yaw_world_origin = cv::Vec3f(0, 0, 0);   // 本帧预测所用云台 yaw 系原点（world 系，
                                                           // 弹道解算线程化后随 Result 传递给输出模式）
    };

    /// 构造时创建内部 GimbalSolver，序列/弹道/偏置参数从 RobotConfig common 段读取
    SequencePredictor();

    /// 同步内部树（st.strict + MCU 弹速）并生成预测云台控制序列 + 瞄准点序列。
    /// 实际计算点（solve）经内部线程池并行执行；每个工作线程通过 thread_local
    /// 绑定一个独立的 GimbalSolver（内部 pitch 粗搜索保持并行且互不竞争）。
    ///
    /// 目标（瞄准点）选择已从 PredictedBallisticSolver 移入本类：solve() 返回
    /// 预测函数列表中全部目标点的结果，predict() 在每个实际计算点的结果之间
    /// 按 predictor.source 自动选择的策略（Armor → NEAREST，PowerRune →
    /// LOWEST_Z）选出该点实际使用的目标（predictor.masked_indices 中索引对应
    /// 的瞄准点不参与选择），并在来源切换时重置自身跨帧状态 State。
    /// 全被屏蔽（屏蔽后无任何瞄准点可选）时自动转为调用 invalidate() 并返回
    /// 无效结果。
    ///
    /// @param predictor  目标预测器（预测函数 + 来源标注 + 快照时间戳 +
    ///                   目标屏蔽索引列表）
    /// @param timestamp  调用时刻（当前帧时间戳）；额外预测时间自动加上
    ///                   (timestamp - predictor.timestamp)，补偿快照生成到
    ///                   消费之间的延迟
    Result predict(const RobotController::State& st, const Predictor& predictor,
                   const std::chrono::steady_clock::time_point& timestamp);

    /// 预测器不可用：重置自身跨帧状态（State）与当前来源记录；
    /// 下次 predict() 从新来源重新开始维护状态
    void invalidate();

    /// 任一内部云台解算器（仅弹道线程内使用；外部请勿直接访问）
    std::shared_ptr<GimbalSolver> gimbal() const { return gimbals_.front(); }

private:
    // 为每个工作线程准备一个独立的 GimbalSolver（内部 TaskPool 互不竞争）
    std::vector<std::shared_ptr<GimbalSolver>> gimbals_;
    std::vector<PredictedBallisticSolver> solvers_;
    TaskPool pool_;                 // 默认构造（min(硬件核数/2, 4) 线程）
    std::atomic<int> next_gimbal_{0};   // thread_local 绑定：worker 首次执行时领取编号

    // 序列生成参数（构造时从 RobotConfig common 读取）
    double extra_predict_time_;
    double dt_control_;
    double pitch_bias_;
    double yaw_bias_;
    int    prediction_points_;       // M：实际精确解算点数
    int    interpolation_refine_;    // K：插值细化倍数
    int    exact_lead_points_;       // n：序列最前面拼接的精确解算前导点数（0 = 关闭）
    double aim_stick_ratio_;         // 瞄准点滞回幅度系数（无单位，>=0；0 = 关闭）

    // 自身跨帧状态（predict() 内部维护）：当 target_predictor 来源（PredictorSource）
    // 切换或 invalidate() 时整体重置（随“总目标”切换失效）。
    struct State {
        // 慢目标瞄准点滞回：上一帧预测序列“第一个值”选中的瞄准点索引
        // （= 上一帧 items.front().target_index；无有效上一帧时为 -1）。
        // 仅当本帧 Predictor.slow_target == true 时被用于滞回；来源切换/失效时
        // 随 state_ 一并清零（无粘滞点 → 直接选最优）。
        int last_first_target_index = -1;
    };
    State state_;
    PredictorSource active_source_;   // 当前 state_ 对应的来源（无有效预测时为 NONE）

    // 线性插值：lo + t*(hi - lo)（t ∈ [0,1]）
    static Item lerpItem(const Item& lo, const Item& hi, double t);
    // 沿段 (P, A) 方向外推 s 步长：A + s*(A - P)（P 为 A 的前一实际计算点）
    static Item extrapItem(const Item& A, const Item& P, double s);
};

#endif // SEQUENCE_PREDICTOR_H
