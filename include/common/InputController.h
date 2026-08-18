// InputController.h — 云台输入控制器（序列输入模式，生成 robotController.set 的序列输入）
#ifndef INPUT_CONTROLLER_H
#define INPUT_CONTROLLER_H

#include <memory>
#include <vector>

#include "RobotController.h"
#include "Ballistic/GimbalSolver.h"
#include "Ballistic/PredictedBallisticSolver.h"

// ============================================================================
// 云台输入控制器（序列输入模式）：将 robotController.getState() 的 st 与
// 自身位姿预测函数转换为 robotController.set 的序列输入。
//
// 全部配置参数在构造时直接从 RobotConfig（config/config.yaml 的
// predicted_ballistic / robot_controller / input_controller 配置段）读取。
//
// - 内部持有 PredictedBallisticSolver（预测弹道解算器，完成飞行时间迭代与目标选取）；
// - 预测序列在 InputController 中生成：i = 1..n，以
//   extra_predict_time + i*dt_control 作为额外预测时间调用 solve()，
//   得到每个序列元素的预测目标点与 GimbalSolver 解算结果（yaw/pitch）；
// - yaw 序列：原样返回（长度 n），每个元素叠加底盘 yaw 修正（imu_euler_yaw - yaw_pos）；
// - pitch 序列：截取第 m 个之后的元素返回（长度 n-m，m 必须小于 n），叠加 pitch 偏置；
// - fire 序列：st.mpc.ref_sequence 与 pred_sequence 每一对按当前方法计算
//   （角度差解缠绕后小于动态阈值），截取第 o 个之后的元素返回；
//   动态阈值 = max(fire_angle_lower_limit,
//                  fire_angle_length / 瞄准目标与 yaw 系原点在 world xy 平面投影的距离)；
// - 每个传出序列截取后保证至少有一个元素：原始序列为空时补一个
//   yaw/pitch=0.0、fire=false，截取会清空时至少保留最后一个元素；
// - auto_aim_enable = 首个序列元素解算结果是否有效（无效时 MCU 侧不执行指令，安全）。
// ============================================================================
class InputController {
public:
    using Predictor = PredictedBallisticSolver::Predictor;

    // rc：RobotController 引用；gimbal：云台角度解算器（内部树已同步）。
    // 序列/弹道/控制参数均从 RobotConfig 直接读取，不外部传入。
    InputController(RobotController& rc, std::shared_ptr<GimbalSolver> gimbal);

    // 每帧调用：输入 st（robotController.getState()）与预测函数
    //（ESEKF::capturePosePredictor 返回），内部生成并发送 robotController.set 序列。
    void update(const RobotController::State& st, const Predictor& predictor);

    // 本帧生成的 set 序列输入（供显示/调试/测试）
    struct LastOutput {
        bool   auto_aim_enable = false;
        cv::Vec3f predicted_point;    // 首个序列元素（i=1）的预测目标点（world 系）
        double predict_time = 0.0;    // 首个序列元素的预测时间（秒）
        std::vector<double> yaw_seq;    // yaw 序列（原样，长度 n）
        std::vector<double> pitch_seq;  // pitch 序列（截取第 m 个之后，长度 n-m）
        std::vector<bool>   fire_seq;   // fire 序列（截取第 o 个之后）
        bool   mpc_available = false; // ref/pred 序列是否可用
        double fire_threshold = 0.0;  // 本帧 fire 判定使用的动态阈值（弧度）
    };
    const LastOutput& lastOutput() const { return last_; }

private:
    // 单对 (ref, pred) 的 fire 判定：角度差解缠绕后小于动态阈值
    static bool computeFire(double ref, double pred, double threshold);

    RobotController& rc_;
    std::shared_ptr<GimbalSolver> gimbal_;
    PredictedBallisticSolver pred_ballistic_;   // 预测弹道解算器（成员）

    // ── 配置参数（构造时从 RobotConfig 读取，代码中不写默认值）──
    double extra_predict_time_;   // 额外预测时间（秒）
    double dt_control_;           // 控制周期（秒）
    int    prediction_seq_len_;   // n：预测序列长度
    int    pitch_seq_lead_;       // m：pitch 序列提前数（必须 < n）
    int    fire_seq_lead_;        // o：fire 序列提前数
    double pitch_bias_;           // pitch 轴偏置（弧度），输出时叠加
    double fire_angle_lower_limit_; // fire 判定角度阈值下限（弧度）
    double fire_angle_length_;      // fire 判定弧长（米）：动态阈值 = length / 距离

    LastOutput last_;
};

#endif // INPUT_CONTROLLER_H
