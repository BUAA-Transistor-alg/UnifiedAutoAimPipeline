// CameraInputMode.h — 海康相机输入模式（IInputMode 实现）
#ifndef CAMERA_INPUT_MODE_H
#define CAMERA_INPUT_MODE_H

#include "IInputMode.h"
#include "camera/Camera.h"
#include "RobotController.h"

/**
 * @brief 实时相机输入模式：包装海康 Camera，取帧同时捕获
 *        RobotController 的严格反解数据包（StrictPose）并打包进 ExtraInputInfo。
 *
 * - 时间戳由 Camera::getLatestFrame 在成功取帧时刻打上（steady_clock）；
 * - strict 数据在同一时刻（取帧后立即）调用 getState() 填充，保证
 *   "帧 + 时间戳 + 同时刻状态" 三者一一对应；底盘 xyz 填 0；
 * - 帧间无人工延迟（getFrameDelay = 0），由相机自身帧率驱动。
 */
class CameraInputMode : public IInputMode {
public:
    CameraInputMode(Camera& camera, RobotController& rc);

    bool getNextFrame(cv::Mat& frame,
                      std::chrono::steady_clock::time_point& timestamp,
                      ExtraInputInfo& extra_info) override;

    std::string getName() const override { return "Camera"; }

    float getFrameDelay() const override { return 0.0f; }

private:
    Camera& camera_;
    RobotController& rc_;
};

#endif // CAMERA_INPUT_MODE_H
