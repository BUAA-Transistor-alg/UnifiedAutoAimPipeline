// RobotTfTree.cpp
#include "common/TransformTree/RobotTfTree.h"

#include "common/RobotConfig.h"

RobotTfTree::RobotTfTree() {
    // 构造时自动读取项目根目录 config/config.yaml（相对路径经 PathResolver 解析），
    // 各节点偏移不再硬编码，修改配置文件即可调整。
    const RobotConfig& cfg = RobotConfig::instance();
    const RobotConfig::TfOffsets& o = cfg.common.tf;

    const float yawJointZOffset   = o.yawJointZOffset;   // yaw 关节沿 z 轴偏移（相对 chassis）（实际上为pitch轴关节相对底盘最低点高度）
    const float pitchJointYOffset = o.pitchJointYOffset; // pitch 关节沿 y 轴偏移（相对 yaw 旋转中心）
    const float imuOffsetX        = o.imuOffsetX;        // imu 相对 head 的 x 偏移 // head 节点实际上为 picth轴 和 一个垂直于picth轴且经过yaw轴的面 的交点位置
    const float imuOffsetY        = o.imuOffsetY;        // imu 相对 head 的 y 偏移
    const float imuOffsetZ        = o.imuOffsetZ;        // imu 相对 head 的 z 偏移
    const float cameraOffsetX     = o.cameraOffsetX;     // camera 相对 head 的 x 偏移
    const float cameraOffsetY     = o.cameraOffsetY;     // camera 相对 head 的 y 偏移
    const float cameraOffsetZ     = o.cameraOffsetZ;     // camera 相对 head 的 z 偏移
    const float muzzleOffsetX     = o.muzzleOffsetX;     // muzzle 相对 head 的 x 偏移
    const float muzzleOffsetY     = o.muzzleOffsetY;     // muzzle 相对 head 的 y 偏移
    const float muzzleOffsetZ     = o.muzzleOffsetZ;     // muzzle 相对 head 的 z 偏移

    // 构建变换链 root -> world -> chassis -> yaw -> pitch -> head -> (imu, camera, muzzle)
    manager_.addNode(WORLD, ROOT);
    manager_.addNode(CHASSIS, WORLD);
    manager_.addNode(YAW, CHASSIS);
    manager_.addNode(PITCH, YAW);
    manager_.addNode(HEAD, PITCH);
    manager_.addNode(IMU, HEAD);
    manager_.addNode(CAMERA, HEAD);
    manager_.addNode(MUZZLE, HEAD);

    // world 固定为原点、零欧拉角
    manager_.setPosition(WORLD, 0.0f, 0.0f, 0.0f);
    manager_.setEuler(WORLD, 0.0f, 0.0f, 0.0f);

    // yaw 节点固定位置，欧拉角由 setYaw 设置（初始为 0）
    manager_.setPosition(YAW, 0.0f, 0.0f, yawJointZOffset);
    manager_.setEuler(YAW, 0.0f, 0.0f, 0.0f);

    // pitch 节点固定位置，欧拉角由 setPitch 设置（初始为 0）
    manager_.setPosition(PITCH, 0.0f, pitchJointYOffset, 0.0f);
    manager_.setEuler(PITCH, 0.0f, 0.0f, 0.0f);

    // head 固定为原点、零欧拉角
    manager_.setPosition(HEAD, 0.0f, 0.0f, 0.0f);
    manager_.setEuler(HEAD, 0.0f, 0.0f, 0.0f);

    // imu / camera 固定位置、零欧拉角
    manager_.setPosition(IMU, imuOffsetX, imuOffsetY, imuOffsetZ);
    manager_.setEuler(IMU, 0.0f, 0.0f, 0.0f);

    manager_.setPosition(CAMERA, cameraOffsetX, cameraOffsetY, cameraOffsetZ);
    manager_.setEuler(CAMERA, 0.0f, 0.0f, 0.0f);

    manager_.setPosition(MUZZLE, muzzleOffsetX, muzzleOffsetY, muzzleOffsetZ);
    manager_.setEuler(MUZZLE, 0.0f, 0.0f, 0.0f);

    // chassis 默认位于 world 原点、零欧拉角
    manager_.setPosition(CHASSIS, 0.0f, 0.0f, 0.0f);
    manager_.setEuler(CHASSIS, 0.0f, 0.0f, 0.0f);
}

void RobotTfTree::setChassisPosition(float x, float y, float z) {
    manager_.setPosition(CHASSIS, x, y, z);
}

void RobotTfTree::setChassisEuler(float yaw, float pitch, float roll) {
    manager_.setEuler(CHASSIS, yaw, pitch, roll);
}

void RobotTfTree::setYaw(float yaw) {
    manager_.setEuler(YAW, yaw, 0.0f, 0.0f);
}

void RobotTfTree::setPitch(float pitch) {
    manager_.setEuler(PITCH, 0.0f, pitch, 0.0f);
}

void RobotTfTree::unlock() {
    manager_.unlock();
}

void RobotTfTree::lockAndComputeCache() {
    manager_.lockAndComputeCache();
}

bool RobotTfTree::isLocked() const {
    return manager_.isLocked();
}

cv::Vec3f RobotTfTree::transformPoint(const std::string& from, const std::string& to, const cv::Vec3f& point) const {
    return manager_.transformPoint(from, to, point);
}

cv::Vec3f RobotTfTree::transformEuler(const std::string& from, const std::string& to, const cv::Vec3f& euler) const {
    return manager_.transformEuler(from, to, euler);
}

RobotTfTree::State RobotTfTree::saveState() const {
    State s;
    if (auto chassis = manager_.getNode(CHASSIS)) {
        s.chassisPosition = chassis->getPosition();
        s.chassisEuler    = chassis->getEuler();
    }
    if (auto yaw = manager_.getNode(YAW)) {
        s.yaw = yaw->getEuler()[0];  // yaw 节点欧拉角为 (yaw, 0, 0)
    }
    if (auto pitch = manager_.getNode(PITCH)) {
        s.pitch = pitch->getEuler()[1];  // pitch 节点欧拉角为 (0, pitch, 0)
    }
    s.locked = manager_.isLocked();
    return s;
}

void RobotTfTree::restoreState(const State& s) {
    unlock();
    setChassisPosition(s.chassisPosition[0], s.chassisPosition[1], s.chassisPosition[2]);
    setChassisEuler(s.chassisEuler[0], s.chassisEuler[1], s.chassisEuler[2]);
    setYaw(s.yaw);
    setPitch(s.pitch);
    if (s.locked) {
        lockAndComputeCache();
    }
}

TransformTreeManager& RobotTfTree::manager() {
    return manager_;
}

const TransformTreeManager& RobotTfTree::manager() const {
    return manager_;
}
