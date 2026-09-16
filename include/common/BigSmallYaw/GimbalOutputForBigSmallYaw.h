// GimbalOutputForBigSmallYaw.h — 新构型（大/小双 yaw）云台控制输出模式
//
// 与旧 GimbalOutput 的差异（其余语义保持一致）：
//   - 控制器为 tcbs::RobotController（大/小 yaw 世界方位角 + pitch + fire 序列）；
//   - 流水线解算出的 yaw 序列是**小 yaw 输出的世界方位角** ψ_small*（= item.yaw），
//     大 yaw 参考序列 ψ_big* 由大小 yaw 拆分器（BigSmallYawSplitter）按“大 yaw 平滑
//     轨迹 + 小 yaw 到限位时无限幅快速运动”规划得到（见其文件头注释）；
//   - fire 序列判据改用控制器 MPC 的**小 yaw** 参考/预测序列（与旧版用 yaw 通道一致）；
//   - 预测不可用时进入保持模式：大/小 yaw 均保持当前严格反解世界方位角，自瞄关闭。
#ifndef BSY_GIMBAL_OUTPUT_FOR_BIG_SMALL_YAW_H
#define BSY_GIMBAL_OUTPUT_FOR_BIG_SMALL_YAW_H

#include <chrono>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common/BigSmallYaw/BigSmallYawSplitter.h"
#include "common/BigSmallYaw/RobotStateForBigSmallYaw.h"
#include "common/Output/IOutputMode.h"

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

    LastOutput last_;

    // 保持模式起点（预测不可用持续超过一个规划时域后复位拆分器跨帧状态）
    bool   holding_ = false;
    std::chrono::steady_clock::time_point hold_start_{};
};

} // namespace bsy

#endif // BSY_GIMBAL_OUTPUT_FOR_BIG_SMALL_YAW_H
