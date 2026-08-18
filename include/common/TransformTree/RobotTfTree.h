// RobotTfTree.h
#ifndef ROBOT_TF_TREE_H
#define ROBOT_TF_TREE_H

#include <string>

#include <opencv2/opencv.hpp>

#include "TransformTreeManager.h"

// 基于 TransformTreeManager 实现的机器人坐标系变换树。
// 变换链：root -> world -> chassis -> yaw -> pitch -> head -> (imu, camera, muzzle)
class RobotTfTree {
public:
    RobotTfTree();

    // chassis 位置与欧拉角设置接口
    void setChassisPosition(float x, float y, float z);
    void setChassisEuler(float yaw, float pitch, float roll);

    // yaw 关节（绕 z 轴）
    void setYaw(float yaw);

    // pitch 关节（绕 x 轴）
    void setPitch(float pitch);

    void unlock();
    void lockAndComputeCache();
    bool isLocked() const;

    cv::Vec3f transformPoint(const std::string& from, const std::string& to, const cv::Vec3f& point) const;
    cv::Vec3f transformEuler(const std::string& from, const std::string& to, const cv::Vec3f& euler) const;

    // 全部可调状态（chassis 位姿、yaw/pitch 关节角、锁定状态）
    struct State {
        cv::Vec3f chassisPosition = cv::Vec3f(0.0f, 0.0f, 0.0f);
        cv::Vec3f chassisEuler    = cv::Vec3f(0.0f, 0.0f, 0.0f);
        float yaw   = 0.0f;
        float pitch = 0.0f;
        bool  locked = false;
    };
    // 保存全部状态到结构体（只读节点数据，不检查锁）
    State saveState() const;
    // 从结构体恢复全部状态（内部自动 unlock 并设置各关节，若原状态为上锁则重新上锁）
    void restoreState(const State& state);

    TransformTreeManager& manager();
    const TransformTreeManager& manager() const;

    // 节点命名常量
    static constexpr const char* ROOT    = "root";
    static constexpr const char* WORLD   = "world";
    static constexpr const char* CHASSIS = "chassis";
    static constexpr const char* YAW     = "yaw";
    static constexpr const char* PITCH   = "pitch";
    static constexpr const char* HEAD    = "head";
    static constexpr const char* IMU     = "imu";
    static constexpr const char* CAMERA  = "camera";
    static constexpr const char* MUZZLE  = "muzzle";

private:
    TransformTreeManager manager_;
};

#endif // ROBOT_TF_TREE_H
