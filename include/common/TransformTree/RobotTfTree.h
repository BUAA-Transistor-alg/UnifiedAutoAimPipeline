// RobotTfTree.h
#ifndef ROBOT_TF_TREE_H
#define ROBOT_TF_TREE_H

#include <string>

#include <opencv2/opencv.hpp>

#include "common/RobotConfig.h"          // YawMode
#include "common/TransformTree/TransformTreeManager.h"

// 基于 TransformTreeManager 实现的机器人坐标系变换树。
//
// yaw 构型在**构造时**由 RobotConfig::common.bigSmallYaw.mode 定型，运行中不可切换：
//   - YawMode::SINGLE（单 yaw，旧）：
//        root -> world -> chassis -> yaw -> pitch -> head -> (imu, camera, muzzle)
//        yaw 节点（setYaw）即云台 yaw 关节（相对底盘单关节）；
//   - YawMode::BIG_SMALL（大/小双 yaw，新）：
//        root -> world -> chassis -> yaw_big -> yaw_small -> pitch -> head -> (imu, camera, muzzle)
//        yaw_big（setYawBig）：大 yaw 关节（相对底盘）；
//        yaw_small（setYawSmall）：小 yaw 关节（相对大 yaw），其节点位置 = 配置里
//        小 yaw 轴相对大 yaw 轴的 xyz 偏移（**大 yaw 系**，随大 yaw 一起旋转）；
//        「云台瞄准方向」由两级之和决定（θ_big + θ_small），见 GimbalSolver。
//
// ⚠ 给后续修改者：**越界调用立刻抛 std::logic_error**（例如 BIG_SMALL 构型下调
//   setYaw / 用 RobotTfTree::YAW 取节点），绝不静默按另一种构型的语义执行。
class RobotTfTree {
public:
    RobotTfTree();

    // 当前构型（构造时定型）
    YawMode yawMode() const { return yaw_mode_; }
    bool isBigSmallYaw() const { return yaw_mode_ == YawMode::BIG_SMALL; }

    // chassis 位置与欧拉角设置接口（两构型通用）
    void setChassisPosition(float x, float y, float z);
    void setChassisEuler(float yaw, float pitch, float roll);

    // ── yaw 关节设置（按构型区分，越界调用抛 std::logic_error）──
    // 单 yaw 构型（SINGLE）：yaw 关节角（绕 z 轴）
    void setYaw(float yaw);
    // 大小 yaw 构型（BIG_SMALL）：大 yaw 关节角（相对底盘，绕 z 轴）
    void setYawBig(float yaw_big);
    // 大小 yaw 构型（BIG_SMALL）：小 yaw 关节角（相对大 yaw，绕 z 轴）
    void setYawSmall(float yaw_small);

    // ── yaw 关节读取（按构型区分，越界调用抛 std::logic_error）──
    float yawSingle() const;   // SINGLE：yaw 关节角
    float yawBig() const;      // BIG_SMALL：大 yaw 关节角
    float yawSmall() const;    // BIG_SMALL：小 yaw 关节角

    // pitch 关节（绕 x 轴），两构型通用
    void setPitch(float pitch);

    // 小 yaw 轴相对大 yaw 轴的偏移（**大 yaw 系**，米；来自配置
    // common.big_small_yaw.big_small.tf.small_yaw_offset_*）。
    // SINGLE 构型下返回零向量（该构型不存在小 yaw 轴）。
    cv::Vec3f smallYawAxisOffset() const { return small_yaw_axis_offset_; }

    void unlock();
    void lockAndComputeCache();
    bool isLocked() const;

    cv::Vec3f transformPoint(const std::string& from, const std::string& to, const cv::Vec3f& point) const;
    cv::Vec3f transformEuler(const std::string& from, const std::string& to, const cv::Vec3f& euler) const;
    cv::Vec3f transformVector(const std::string& from, const std::string& to, const cv::Vec3f& vector) const;

    // 全部可调状态（chassis 位姿、yaw/pitch 关节角、锁定状态）。
    // yaw / yawBig+yawSmall 两套字段**按构型只填一套**，另一套为 NaN（防止跨构型误用）。
    struct State {
        cv::Vec3f chassisPosition = cv::Vec3f(0.0f, 0.0f, 0.0f);
        cv::Vec3f chassisEuler    = cv::Vec3f(0.0f, 0.0f, 0.0f);
        float yaw      = std::numeric_limits<float>::quiet_NaN();  // SINGLE 专用
        float yawBig   = std::numeric_limits<float>::quiet_NaN();  // BIG_SMALL 专用
        float yawSmall = std::numeric_limits<float>::quiet_NaN();  // BIG_SMALL 专用
        float pitch    = 0.0f;
        bool  locked   = false;
        YawMode mode   = YawMode::SINGLE;   // 保存时的构型（restoreState 会校验）
    };
    // 保存全部状态到结构体（只读节点数据，不检查锁）
    State saveState() const;
    // 从结构体恢复全部状态（内部自动 unlock 并设置各关节，若原状态为上锁则重新上锁）；
    // 状态构型与当前树构型不符时抛 std::logic_error
    void restoreState(const State& state);

    TransformTreeManager& manager();
    const TransformTreeManager& manager() const;

    // 节点命名常量
    static constexpr const char* ROOT    = "root";
    static constexpr const char* WORLD   = "world";
    static constexpr const char* CHASSIS = "chassis";
    static constexpr const char* YAW     = "yaw";        // SINGLE 构型的 yaw 关节
    static constexpr const char* YAW_BIG   = "yaw_big";  // BIG_SMALL 构型的大 yaw 关节
    static constexpr const char* YAW_SMALL = "yaw_small";// BIG_SMALL 构型的小 yaw 关节
    static constexpr const char* PITCH   = "pitch";
    static constexpr const char* HEAD    = "head";
    static constexpr const char* IMU     = "imu";
    static constexpr const char* CAMERA  = "camera";
    static constexpr const char* MUZZLE  = "muzzle";

private:
    // 构型校验：不符则抛 std::logic_error（what = 调用者名字）
    void requireMode(YawMode expected, const char* what) const;

    TransformTreeManager manager_;
    YawMode     yaw_mode_ = YawMode::SINGLE;
    cv::Vec3f   small_yaw_axis_offset_ = cv::Vec3f(0.0f, 0.0f, 0.0f);
};

#endif // ROBOT_TF_TREE_H
