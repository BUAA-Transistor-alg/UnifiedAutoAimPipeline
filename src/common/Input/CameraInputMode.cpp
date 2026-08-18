// CameraInputMode.cpp — 海康相机输入模式实现
#include "Input/CameraInputMode.h"

#include <iostream>

CameraInputMode::CameraInputMode(Camera& camera, RobotController& rc)
    : camera_(camera), rc_(rc) {}

bool CameraInputMode::getNextFrame(cv::Mat& frame,
                                   std::chrono::steady_clock::time_point& timestamp,
                                   ExtraInputInfo& extra_info) {
    // 相机取流失败（无新帧）时返回空帧，调用方应 sleep 后继续轮询
    if (!camera_.getLatestFrame(frame, timestamp)) {
        frame.release();  // 标记无新帧
        return true;
    }

    // 取帧同一时刻捕获 RobotController 严格反解数据包（StrictPose）→ ExtraInputInfo
    // 底盘 xyz 固定为 0（相机模式下底盘视为世界原点）。
    extra_info = ExtraInputInfo{};
    const RobotController::State st = rc_.getState();
    extra_info.imu_euler_yaw   = st.strict.imu_euler_yaw;
    extra_info.imu_euler_pitch = st.strict.imu_euler_pitch;
    extra_info.imu_euler_roll  = st.strict.imu_euler_roll;
    extra_info.yaw_pos         = st.strict.yaw_pos;
    extra_info.pitch_angle     = st.strict.pitch_angle;
    extra_info.chassis_yaw     = st.strict.chassis_yaw;
    extra_info.chassis_pitch   = st.strict.chassis_pitch;
    extra_info.chassis_roll    = st.strict.chassis_roll;
    // chassis_x / chassis_y / chassis_z 保持 0
    return true;
}
