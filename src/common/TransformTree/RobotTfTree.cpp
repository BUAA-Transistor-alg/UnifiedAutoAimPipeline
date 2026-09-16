// RobotTfTree.cpp
#include "common/TransformTree/RobotTfTree.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "common/RobotConfig.h"

namespace {
constexpr float kNaNf = std::numeric_limits<float>::quiet_NaN();

void checkFiniteOrThrow(float v, const char* what) {
    if (!std::isfinite(v)) {
        throw std::logic_error(std::string("RobotTfTree: ") + what +
                               " 收到非有限关节角（NaN/Inf）——多半是当前构型的信息包未填充"
                               "（见 ExtraInputInfo 的单 yaw / 大小 yaw 两包约定）");
    }
}
} // namespace

RobotTfTree::RobotTfTree() {
    // 构造时自动读取机器配置文件（config/robots/<active_config>.yaml，相对路径
    // 经 PathResolver 解析），各节点偏移不再硬编码，修改配置文件即可调整；
    // yaw 构型同样在此时定型（common.big_small_yaw.mode），运行中不可切换。
    const RobotConfig& cfg = RobotConfig::instance();
    // tf 偏移按构型取（single / big_small 两支字段不同，见 RobotConfig::BigSmallYawParams）
    const RobotConfig::TfOffsets& o = cfg.common.tf();
    yaw_mode_ = cfg.common.bigSmallYaw.mode;

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

    // 小 yaw 轴相对大 yaw 轴的偏移（大 yaw 系，米）
    small_yaw_axis_offset_ = cv::Vec3f(o.smallYawOffsetX, o.smallYawOffsetY,
                                       o.smallYawOffsetZ);

    if (yaw_mode_ == YawMode::SINGLE) {
        // ── 单 yaw 构型：root -> world -> chassis -> yaw -> pitch -> head -> (imu/camera/muzzle) ──
        manager_.addNode(WORLD, ROOT);
        manager_.addNode(CHASSIS, WORLD);
        manager_.addNode(YAW, CHASSIS);
        manager_.addNode(PITCH, YAW);
    } else {
        // ── 大小 yaw 构型：原 yaw 节点拆成串联的两个节点 ──
        // root -> world -> chassis -> yaw_big -> yaw_small -> pitch -> head -> (imu/camera/muzzle)
        // yaw_small 节点位置 = 小 yaw 轴相对大 yaw 轴的 xyz 偏移（大 yaw 系，随大 yaw 旋转）
        manager_.addNode(WORLD, ROOT);
        manager_.addNode(CHASSIS, WORLD);
        manager_.addNode(YAW_BIG, CHASSIS);
        manager_.addNode(YAW_SMALL, YAW_BIG);
        manager_.addNode(PITCH, YAW_SMALL);
    }
    manager_.addNode(HEAD, PITCH);
    manager_.addNode(IMU, HEAD);
    manager_.addNode(CAMERA, HEAD);
    manager_.addNode(MUZZLE, HEAD);

    // world 固定为原点、零欧拉角
    manager_.setPosition(WORLD, 0.0f, 0.0f, 0.0f);
    manager_.setEuler(WORLD, 0.0f, 0.0f, 0.0f);

    if (yaw_mode_ == YawMode::SINGLE) {
        // yaw 节点固定位置，欧拉角由 setYaw 设置（初始为 0）
        manager_.setPosition(YAW, 0.0f, 0.0f, yawJointZOffset);
        manager_.setEuler(YAW, 0.0f, 0.0f, 0.0f);
    } else {
        // 大 yaw 节点：位置同原 yaw 节点（相对 chassis），欧拉角由 setYawBig 设置
        manager_.setPosition(YAW_BIG, 0.0f, 0.0f, yawJointZOffset);
        manager_.setEuler(YAW_BIG, 0.0f, 0.0f, 0.0f);
        // 小 yaw 节点：位置 = 配置的小 yaw 轴偏移（大 yaw 系），欧拉角由 setYawSmall 设置
        manager_.setPosition(YAW_SMALL, small_yaw_axis_offset_);
        manager_.setEuler(YAW_SMALL, 0.0f, 0.0f, 0.0f);
    }

    // pitch 节点固定位置（相对其父节点 = yaw / yaw_small 的旋转中心），欧拉角由 setPitch 设置
    manager_.setPosition(PITCH, 0.0f, pitchJointYOffset, 0.0f);
    manager_.setEuler(PITCH, 0.0f, 0.0f, 0.0f);

    // head 固定为原点、零欧拉角
    manager_.setPosition(HEAD, 0.0f, 0.0f, 0.0f);
    manager_.setEuler(HEAD, 0.0f, 0.0f, 0.0f);

    // imu / camera / muzzle 固定位置、零欧拉角
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

void RobotTfTree::requireMode(YawMode expected, const char* what) const {
    if (yaw_mode_ != expected) {
        throw std::logic_error(
            std::string("RobotTfTree: ") + what + " 只适用于 " +
            (expected == YawMode::SINGLE ? "单 yaw 构型（YawMode::SINGLE）"
                                         : "大小 yaw 构型（YawMode::BIG_SMALL）") +
            "，当前构型为 " +
            (yaw_mode_ == YawMode::SINGLE ? "单 yaw（SINGLE）" : "大小 yaw（BIG_SMALL）") +
            "（由 config common.big_small_yaw.mode 决定，构造时定型、运行中不可切换）");
    }
}

void RobotTfTree::setChassisPosition(float x, float y, float z) {
    manager_.setPosition(CHASSIS, x, y, z);
}

void RobotTfTree::setChassisEuler(float yaw, float pitch, float roll) {
    manager_.setEuler(CHASSIS, yaw, pitch, roll);
}

void RobotTfTree::setYaw(float yaw) {
    requireMode(YawMode::SINGLE, "setYaw");
    checkFiniteOrThrow(yaw, "setYaw");
    manager_.setEuler(YAW, yaw, 0.0f, 0.0f);
}

void RobotTfTree::setYawBig(float yaw_big) {
    requireMode(YawMode::BIG_SMALL, "setYawBig");
    checkFiniteOrThrow(yaw_big, "setYawBig");
    manager_.setEuler(YAW_BIG, yaw_big, 0.0f, 0.0f);
}

void RobotTfTree::setYawSmall(float yaw_small) {
    requireMode(YawMode::BIG_SMALL, "setYawSmall");
    checkFiniteOrThrow(yaw_small, "setYawSmall");
    manager_.setEuler(YAW_SMALL, yaw_small, 0.0f, 0.0f);
}

float RobotTfTree::yawSingle() const {
    requireMode(YawMode::SINGLE, "yawSingle");
    auto n = manager_.getNode(YAW);
    return n ? n->getEuler()[0] : 0.0f;
}

float RobotTfTree::yawBig() const {
    requireMode(YawMode::BIG_SMALL, "yawBig");
    auto n = manager_.getNode(YAW_BIG);
    return n ? n->getEuler()[0] : 0.0f;
}

float RobotTfTree::yawSmall() const {
    requireMode(YawMode::BIG_SMALL, "yawSmall");
    auto n = manager_.getNode(YAW_SMALL);
    return n ? n->getEuler()[0] : 0.0f;
}

void RobotTfTree::setPitch(float pitch) {
    checkFiniteOrThrow(pitch, "setPitch");
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

cv::Vec3f RobotTfTree::transformVector(const std::string& from, const std::string& to, const cv::Vec3f& vector) const {
    return manager_.transformVector(from, to, vector);
}

RobotTfTree::State RobotTfTree::saveState() const {
    State s;
    s.mode = yaw_mode_;
    if (auto chassis = manager_.getNode(CHASSIS)) {
        s.chassisPosition = chassis->getPosition();
        s.chassisEuler    = chassis->getEuler();
    }
    if (yaw_mode_ == YawMode::SINGLE) {
        if (auto yaw = manager_.getNode(YAW)) {
            s.yaw = yaw->getEuler()[0];  // yaw 节点欧拉角为 (yaw, 0, 0)
        }
    } else {
        if (auto yawBig = manager_.getNode(YAW_BIG)) {
            s.yawBig = yawBig->getEuler()[0];
        }
        if (auto yawSmall = manager_.getNode(YAW_SMALL)) {
            s.yawSmall = yawSmall->getEuler()[0];
        }
    }
    if (auto pitch = manager_.getNode(PITCH)) {
        s.pitch = pitch->getEuler()[1];  // pitch 节点欧拉角为 (0, pitch, 0)
    }
    s.locked = manager_.isLocked();
    return s;
}

void RobotTfTree::restoreState(const State& s) {
    if (s.mode != yaw_mode_) {
        throw std::logic_error("RobotTfTree: restoreState 的状态来自另一种 yaw 构型"
                               "（保存时构型与当前构型不一致）——不允许跨构型恢复");
    }
    unlock();
    setChassisPosition(s.chassisPosition[0], s.chassisPosition[1], s.chassisPosition[2]);
    setChassisEuler(s.chassisEuler[0], s.chassisEuler[1], s.chassisEuler[2]);
    if (yaw_mode_ == YawMode::SINGLE) {
        setYaw(s.yaw);
    } else {
        setYawBig(s.yawBig);
        setYawSmall(s.yawSmall);
    }
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
