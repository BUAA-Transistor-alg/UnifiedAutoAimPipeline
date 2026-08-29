#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "Armor/EKF/SuperPowerPredictor.h"

// 单类（label 0~5）装甲板的 EKF 生命周期封装：仿照 OutpostESEKF::processFrame
// 的统一帧处理，调用方每帧对每一类只用其对应的数据调用一次 processFrame。
// 内部只通过移植的 SuperPowerPredictor 原接口工作（构造 / update / updatePair /
// missUpdate / clear / hasState），不直接使用 Target / ExtendedKalmanFilter 等
// 内部类型。双板（联合）观测经原接口的 YAML 配置机制（superpower_ekf.joint_update）
// 启用，默认开启，见 Params::jointUpdateEnabled。
//
// 全部可调参数（Params）由调用方从 config armor.super_power_ekf 段提供，经
// buildConfig 组装成原接口的 YAML 配置节点下发（缺省字段原接口用内置默认）。
namespace sp_ekf {

class ClassEKF {
public:
    using TimePoint = std::chrono::steady_clock::time_point;

    // 可调参数（默认值与 SP standard3 普通四装甲一致；ArmorPipeline 从
    // config armor.super_power_ekf 段完整覆盖，不依赖本默认值）
    struct Params {
        double observationLostTimeoutSec;  // 连续无观测多久后自动销毁（秒）
        double initialRadiusM;             // SP 普通四装甲初始半径（米）
        bool   jointUpdateEnabled;         // 启用双板（联合）观测更新（原接口 joint_update.enabled）
        int    minDetectCount;             // 最小识别帧数（DETECTING→TRACKING）
        int    maxTempLostCount;           // 临时失检最大帧数（超过转 LOST 重建）
        double maxDtSec;                   // 最大帧间隔（秒，超过视为时间不连续）
        int    armorNum;                   // 装甲板数量（普通四装甲=4）
        double angularVelocityFitWindowSec;   // 角速度拟合滑窗（秒）
        int    angularVelocityFitMinSamples;  // 角速度拟合最少样本数
        // 双板联合更新门控参数（原接口 joint_update 段）
        double jointMaxNis;                     // 联合 NIS 门控阈值
        double jointMaxSecondaryPositionErrorM; // 副板最大位置误差门控（米）
        double jointMaxSecondaryAngleErrorRad;  // 副板最大角度误差门控（弧度）
        double jointMeasurementVarianceScale;   // 联合更新位置方差放大系数
        double jointAngleVarianceScale;         // 联合更新角度方差放大系数

        Params()
            : observationLostTimeoutSec(2.0),
              initialRadiusM(0.2),
              jointUpdateEnabled(true),
              minDetectCount(5),
              maxTempLostCount(15),
              maxDtSec(0.1),
              armorNum(4),
              angularVelocityFitWindowSec(0.20),
              angularVelocityFitMinSamples(4),
              jointMaxNis(20.09),
              jointMaxSecondaryPositionErrorM(0.45),
              jointMaxSecondaryAngleErrorRad(0.80),
              jointMeasurementVarianceScale(1.5),
              jointAngleVarianceScale(4.0) {}
    };

    explicit ClassEKF(const Params& params = Params{});

    /**
     * @brief 统一帧处理：用本类本帧的观测数据自动完成 初始化 / 更新 / 超时自动销毁。
     *
     * world_pos 与 world_euler 一一对应（每物体的世界坐标（米）/ 世界欧拉角（弧度）），
     * 可为空（本帧无该类观测）。内部逻辑（全部经 SuperPowerPredictor 原接口）：
     *  - 有有效观测（world_pos 非 (0,0,0)；PnP 失败填充的零观测自动跳过）：
     *      * 未初始化：以首条有效观测构造 SuperPowerPredictor（内部建立 Target，dt=0）；
     *      * 已初始化且本帧至少两块有效观测（同一时间戳）：updatePair(primary,
     *        secondary) 双板联合更新——门控失败由原接口自动回退为单板更新；
     *      * 已初始化且仅一块观测：update(primary) 单板更新；
     *      预测 dt 由原接口内部按“距上一次更新”计算（失检期间仅预测不重复计入）。
     *  - 无观测（失检）：连续超过 observationLostTimeoutSec → clear() 销毁内部目标
     *    （下次观测自动重建）；未超时 → missUpdate(t) 仅做时间预测推进运动模型。
     * 观测单位在边界换算：世界坐标米→毫米（原接口约定 mm），世界欧拉 yaw（弧度）
     * 作为项目 yaw 传入（原接口内部再转 SP 外法线角）。
     * @param world_pos   本类本帧各物体世界坐标（米）
     * @param world_euler 本类本帧各物体世界欧拉角（弧度）
     * @param t           本帧时间戳
     * @return 处理结束后本类 state 是否可用（predictor_->ready()）；可用时已用
     *         predictor_->state() 刷新内部 state_ 快照，供 capturePosePredictor() 使用
     */
    bool processFrame(const std::vector<cv::Vec3f>& world_pos,
                      const std::vector<cv::Vec3f>& world_euler,
                      const TimePoint& t);

    /**
     * @brief 仿照 OutpostESEKF::capturePosePredictor：state 可用时完全捕捉当前
     * 后验状态，构造一个独立的装甲位置预测器快照。
     *
     * 返回的函数签名为 std::vector<cv::Point3f>(double dt)：传入预测时间 dt（秒，
     * 可为任意实数，0 表示当前时刻），返回 4 块装甲中心位置（世界坐标，米）。
     * 与 OutpostESEKF 的差异：目标数是 4（旋转装甲），运动模型为匀速平移 +
     * 匀角速旋转外推。快照复制调用时刻的 state_（值拷贝），不随后续
     * processFrame / update / missUpdate 改变。
     *
     * @return state 可用（最近一帧 predictor_->ready()）时返回预测器；否则返回 nullptr
     */
    std::unique_ptr<std::function<std::vector<cv::Point3f>(double)>>
    capturePosePredictor() const;

    /// 经原接口 clear() 清空内部预测器状态与计时（下次观测自动重新初始化）。
    void reset();

    bool hasTarget() const { return predictor_ && predictor_->hasState(); }
    /// 当前内部预测器（原接口，未初始化时为 nullptr）：供后续输出阶段经
    /// state() / predict() / debugState() 等公开接口取结果，勿触碰其内部类型。
    const SuperPowerPredictor* predictor() const { return predictor_.get(); }

    /// state 是否有效（最近一帧 predictor_->ready()，见 processFrame 返回值）。
    bool stateAvailable() const { return state_available_; }

    /// 车体中心世界坐标（米）。仅 state 有效时有效（先查 stateAvailable()）。
    const cv::Vec3d& getPosition() const { return position_; }

    /// 车体当前角度（yaw）对应的旋转矩阵（CV_64F，world 系）。
    /// 仅 state 有效时有效（先查 stateAvailable()）；与 OutpostESEKF::getRotationMatrix
    /// 形式一致，只是仅含绕世界系 z 轴的旋转（车体 yaw）。
    cv::Mat getRotationMatrix() const { return R_; }

private:
    // 按 Params 构造原接口（SuperPowerPredictor）的 YAML 配置节点：段内字段与
    // config armor.super_power_ekf 一一对应，全部下发到原接口（普通四装甲）。
    static std::shared_ptr<YAML::Node> buildConfig(const Params& params);

    Params params_;
    std::shared_ptr<YAML::Node> config_;       // 原接口配置节点（构造 SuperPowerPredictor 时传入）
    std::unique_ptr<SuperPowerPredictor> predictor_;  // 原接口预测器（内部含 Target/EKF）
    TimePoint last_obs_ts_;                    // 最近一次观测帧时间戳（超时自动销毁判定用）
    EKFTargetState state_;                     // 最近一帧 ready() 时保存的滤波后验（原接口 state()）
    bool state_available_ = false;             // state_ 是否有效（最近一帧 predictor_->ready()）
    cv::Vec3d position_ = cv::Vec3d(0, 0, 0);  // state 有效时的车体中心（world，米，state_ mm→m）
    cv::Mat   R_;                              // state 有效时的车体旋转矩阵（CV_64F，由 yaw 构造）
};

}  // namespace sp_ekf
