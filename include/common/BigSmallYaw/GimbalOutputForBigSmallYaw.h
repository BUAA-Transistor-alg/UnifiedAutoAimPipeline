// GimbalOutputForBigSmallYaw.h — 新构型（大/小双 yaw）云台控制输出模式
//
// 与旧 GimbalOutput 的差异（其余语义保持一致）：
//   - 控制器为 tcbs::RobotController（大/小 yaw 世界方位角 + pitch + fire 序列）；
//   - 流水线解算出的 yaw 序列是**小 yaw 输出的世界方位角** ψ_small*（= item.yaw），
//     大 yaw 参考序列 ψ_big* 由大小 yaw 拆分器（BigSmallYawSplitter）按“大 yaw 平滑
//     轨迹 + 小 yaw 到限位时无限幅快速运动”规划得到（见其文件头注释）；
//   - fire 序列判据改用控制器 MPC 的**小 yaw** 参考/预测序列（与旧版用 yaw 通道一致）；
//   - 预测不可用时进入保持模式：大/小 yaw 均保持当前严格反解世界方位角，自瞄关闭。
//     若开启哨兵扫描控制器（config common.sentry_controller.enabled），无目标超过
//     idle_timeout_sec 后进入扫描模式：大/小 yaw 取**同一个**目标世界方位角
//     （同一条扫描序列，不保持关节角差）、pitch 走锯齿波（见 common/SentryController.h）。
#ifndef BSY_GIMBAL_OUTPUT_FOR_BIG_SMALL_YAW_H
#define BSY_GIMBAL_OUTPUT_FOR_BIG_SMALL_YAW_H

#include <chrono>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common/BigSmallYaw/BigSmallYawSplitter.h"
#include "common/BigSmallYaw/RobotStateForBigSmallYaw.h"
#include "common/Output/IOutputMode.h"
#include "common/SentryController.h"

namespace bsy {

class GimbalOutputForBigSmallYaw : public IOutputMode {
public:
    explicit GimbalOutputForBigSmallYaw(RobotControllerAdapter& ctrl);

    /// @param rc 参数为 nullptr（本模式构造时已持有适配器引用；tcs 控制器不参与新构型）
    void update(const PipelineResult& result, tcs::RobotController* rc,
                OutputContext& ctx) override;

    OutputMode type() const override { return OutputMode::GIMBAL_BIG_SMALL; }
    std::string getName() const override { return "GimbalBigSmall"; }

    /// 本帧下发的序列与拆分器诊断（供状态行/可视化显示）
    struct LastOutput {
        bool auto_aim_enable = false;
        cv::Vec3f predicted_point = cv::Vec3f(0, 0, 0);
        double predict_time = 0.0;
        std::vector<double> big_yaw_seq;     // ψ_big*（世界方位角）
        std::vector<double> small_yaw_seq;   // ψ_small*（世界方位角）
        std::vector<double> pitch_seq;
        std::vector<bool>   fire_seq;
        bool   mpc_available = false;
        double fire_threshold = 0.0;
        // ── 拆分器诊断 ──
        int    jump_count = 0;               // 本帧被修正的点数
        int    unlimited_episodes = 0;       // 本帧无限幅运动段数（连续修正算一段）
        bool   over_limit = false;           // 是否发生越软限位修正
        double theta_small_max_abs = 0.0;    // 本帧 |θ_small| 最大值（rad）
        double soft_min = 0.0, soft_max = 0.0;
    };
    const LastOutput& lastOutput() const { return last_; }

private:
    // 单对 (ref, pred) 的 fire 判定：角度差解缠绕后小于动态阈值
    static bool computeFire(double ref, double pred, double threshold);

    RobotControllerAdapter& ctrl_;
    BigSmallYawSplitter      splitter_;

    // ── 配置（构造时从 RobotConfig common 读取）──
    bool   big_torque_only_;
    bool   small_torque_only_;
    int    pitch_seq_lead_;
    int    fire_seq_lead_;
    double fire_angle_lower_limit_;
    double fire_angle_length_;
    double dt_control_;
    // fast_target 枪线判定（需求5）用的时间轴参数：火控点 index 的开火时刻 =
    // extra_predict_time + (index+1)·dt_control（与返回点索引时间同一时间轴）
    double extra_predict_time_;

    LastOutput last_;

    // 保持模式起点（预测不可用持续超过一个规划时域后复位拆分器跨帧状态）
    bool   holding_ = false;
    std::chrono::steady_clock::time_point hold_start_{};

    // ── 哨兵扫描控制器（可选功能，config common.sentry_controller）──
    // enabled = false 时 sentry_ 所有接口为空操作，下面 else 分支走原有保持逻辑，
    // 行为与未引入本功能时完全一致（见 common/SentryController.h）。
    sentry::SentryController sentry_;
    int    scan_seq_points_;   // 扫描/保持段序列长度 = 正常预测序列总点数
                               // (prediction_points-1)*interpolation_refine+1+exact_lead_points
    // 上一个有效输出序列的首值（进入扫描前的保持段用它填充整条序列；
    // 从未有过有效输出时退化为本帧实测角度）
    bool   have_last_valid_ = false;
    double last_valid_big_   = 0.0;   // ψ_big* 首值
    double last_valid_small_ = 0.0;   // ψ_small* 首值
    double last_valid_pitch_ = 0.0;   // pitch 首值
};

} // namespace bsy

#endif // BSY_GIMBAL_OUTPUT_FOR_BIG_SMALL_YAW_H
