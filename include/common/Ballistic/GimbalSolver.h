// GimbalSolver.h — 云台角度解算器
#ifndef GIMBAL_SOLVER_H
#define GIMBAL_SOLVER_H

#include <memory>

#include <opencv2/opencv.hpp>

#include "Ballistic/BallisticSolver.h"
#include "TaskPool.h"
#include "TransformTree/RobotTfTree.h"

// 云台角度解算器：将目标点（world 系 3D 坐标）解算为云台 yaw / pitch 角度。
//
// - 内部维护一个独立的 RobotTfTree（各节点偏移参数来自 RobotConfig），
//   供坐标解算使用，与外部实时变换树互不影响；
// - yaw 解算（由 RobotTfTree::computeYawToAimTarget 迁移而来，语义不变）：
//   在 pitch 关节为 0 时，使 muzzle 系 +y 射线在 world xy 平面上的投影经过目标点 xy 投影；
// - pitch 解算：给定 yaw，把「world 系下目标点 3D 坐标相对 muzzle 系原点的相对位置」
//   连同该 yaw 一起传入 BallisticSolver，取使弹道最近点 distance 最短的 pitch；
//   最短 distance 超过阈值（RobotConfig::gimbal.distanceThreshold）时解算失效；
// - solveAim：打包解算 yaw + pitch，任意一步失效即返回 success = false。
class GimbalSolver {
public:
    // 打包解算结果
    struct AimResult {
        bool   success = false;   // 是否解算成功
        float  yaw     = 0.0f;    // 解算出的 yaw（弧度，归一化到 (-pi, pi]）
        float  pitch   = 0.0f;    // 解算出的 pitch（弧度）
        double distance = 0.0;    // 弹道最近点与目标的距离（米）
        double flight_time = 0.0; // 解算成功时选取的云台角度参数所用的弹道飞行时间（秒）
    };

    GimbalSolver();

    // 内部坐标系树（调用方可直接同步底盘位姿 / 云台角度 / 状态）
    RobotTfTree& tree() { return *tree_; }
    const RobotTfTree& tree() const { return *tree_; }

    // 便捷同步接口（作用于内部树，调用前需确保内部树处于解锁状态）
    void setChassisPosition(float x, float y, float z);
    void setChassisEuler(float yaw, float pitch, float roll);
    void setYaw(float yaw);
    void setPitch(float pitch);

    // 默认弹丸初速（m/s），可在运行时覆盖
    void setBulletVelocity(double v) { bullet_velocity_ = v; }
    double bulletVelocity() const { return bullet_velocity_; }

    // 运行时调整 pitch 搜索范围 / 步长（弧度），默认来自 RobotConfig::gimbal
    void setPitchRange(float min, float max) {
        pitch_min_ = min;
        pitch_max_ = max;
    }
    void setPitchSearchStep(float step) { pitch_step_ = step; }

    // yaw 解算：在给定 pitch 下，使 muzzle 系 +y 方向（Rz(yaw)*Rx(pitch)*[0,1,0]）在 world xy
    // 平面上的投影经过目标点 xy 投影（muzzle 原点位置随 pitch 一并计入）。pitch = 0 时退化为
    // 原有"水平瞄准"语义。只读取内部树节点数据，不依赖变换缓存，不受上锁限制。
    // 成功返回 true 并写入 yawOut（归一化到 (-pi, pi]）；
    // 失败（退化 / 无解 / 射线背向目标等）返回 false，yawOut 置为内部树当前 yaw。
    bool computeYawToAimTarget(const cv::Vec3f& targetWorld, float pitch, float& yawOut) const;

    // pitch 解算：给定 yaw（需附加），求使 BallisticSolver 最近点 distance 最短的 pitch。
    // 每个候选 pitch 都会重新计算 muzzle 原点（随 pitch 移动）并求相对位置后传入 BallisticSolver。
    // 当 pitch 允许范围较大时，距离曲线可能存在多个局部极小值（弹道较水平 / 较高，非线性阻力下
    // 可能更多）：收集全部局部极小值点，取其中 pitch 最小的一个进行黄金分割细化。
    // 粗搜索的各候选 pitch 评估相互独立，交由内部线程池（TaskPool）并行执行以加速求解；
    // 评估函数只读树数据、不修改内部树（线程安全）。
    // 成功（细化后最短 distance <= 阈值）返回 true 并写入 pitchOut / minDistanceOut /
    // minDistancePlaneOut（弹道平面内距离分量）；否则返回 false。
    bool computePitchToAimTarget(const cv::Vec3f& targetWorld, float yaw, double bulletVelocity,
                                 float& pitchOut, double& minDistanceOut,
                                 double& minDistancePlaneOut) const;
    // 详细版：额外输出选中 pitch 对应的弹道飞行时间 flightTimeOut（秒）
    bool computePitchToAimTarget(const cv::Vec3f& targetWorld, float yaw, double bulletVelocity,
                                 float& pitchOut, double& minDistanceOut,
                                 double& minDistancePlaneOut, double& flightTimeOut) const;

    // 当前 muzzle 系原点在 world 系下的坐标（使用内部树当前的 yaw/pitch 关节角；
    // 只读节点数据，不依赖变换缓存/上锁）。muzzleWorldOrigin 调用前需先同步内部树关节角。
    cv::Vec3f muzzleWorldOrigin() const;

    // yaw 系原点（yaw 关节旋转中心）在 world 系下的坐标（只读节点数据，不依赖缓存/上锁）。
    // 供 fire 判定的动态角度阈值计算（瞄准目标与 yaw 原点在 world xy 平面的投影距离）。
    cv::Vec3f yawWorldOrigin() const;

    // 打包解算：先以 pitch = 0 解 yaw，再以该 yaw 解 pitch；
    // 若首轮最近距离超过 distance_iterate_threshold（迭代触发阈值），则用新 pitch 重算 yaw
    // 并再次解 pitch（两轮保留更优者）；最终最近距离不超过 distance_threshold（原阈值）才算成功。
    // 任意一步失效返回 success = false。
    AimResult solveAim(const cv::Vec3f& targetWorld, double bulletVelocity) const;
    // 使用默认弹丸初速（RobotConfig::gimbal.bulletVelocity，可经 setBulletVelocity 覆盖）
    AimResult solveAim(const cv::Vec3f& targetWorld) const;

private:
    // pitch 搜索期间不变的树数据（构造一次后供并行评估读取，避免各线程竞争内部树）
    struct EvalContext {
        cv::Vec3f chassisPos;        // chassis 位置（world）
        cv::Mat   Rc;                // chassis 旋转（chassis -> world）
        cv::Vec3f yawPos;            // yaw 旋转中心相对 chassis 的位置
        cv::Vec3f pitchPos;          // pitch 节点相对 yaw 的位置
        cv::Vec3f u;                 // headPos + muzzlePos（muzzle 原点在 pitch 系中的位置）
        double    stopZ;             // 弹道计算截止高度（world 系，米，由 buildContext 从配置赋值）
    };
    EvalContext buildContext() const;

    // 单个 (yaw, pitch) 下：解析求 muzzle 原点/指向（不修改内部树，线程安全）→ 相对位置
    // → BallisticSolver 最近点结果
    BallisticSolver::NearestResult evaluateDistance(const EvalContext& ctx,
                                                    const cv::Vec3f& targetWorld,
                                                    float yaw, float pitch,
                                                    double bulletVelocity) const;
    // 以粗搜索最优点为中心做黄金分割细化
    float refinePitch(const EvalContext& ctx, const cv::Vec3f& targetWorld, float yaw,
                      double bulletVelocity, float coarseBest) const;

    std::shared_ptr<RobotTfTree>     tree_;      // 内部专用坐标系树（偏移来自 RobotConfig）
    std::shared_ptr<BallisticSolver> ballistic_; // 弹道解算器（弹丸参数来自 RobotConfig）
    mutable TaskPool pool_;        // 线程池：并行加速 pitch 粗搜索评估（TaskPool，复制自 Power_Rune_Auto_Aim）
    // 以下参数均在构造函数中从 RobotConfig（config/config.yaml）读取，代码中不写默认值
    double bullet_velocity_;               // 默认弹丸初速（m/s）
    double distance_threshold_;            // 成功阈值（米）：最终最近距离不超过该值才算解算成功
    double distance_iterate_threshold_;    // 迭代触发阈值（米）：首轮距离大于该值则触发迭代优化
    double stop_z_;                        // 弹道计算截止高度（world 系，米），低于该高度停止积分
    float  pitch_min_;                     // pitch 搜索下界（弧度）
    float  pitch_max_;                     // pitch 搜索上界（弧度）
    float  pitch_step_;                    // pitch 粗搜索步长（弧度）
};

#endif // GIMBAL_SOLVER_H
