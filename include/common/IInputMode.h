#ifndef IINPUTMODE_H
#define IINPUTMODE_H

#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>

/**
 * @brief 与一帧图像绑定的额外信息（统一为 StrictPose 数据包 + 底盘世界坐标）。
 *
 * 字段语义与 RobotController::StrictPose 一致（角度 wrap 到 (-π, π]，
 * 缺失数据以 0 参与），保证 R_imu = R_chassis·Rz(yaw_pos)·Rx(pitch_angle)
 * 恒成立；chassis_x/y/z 为底盘在世界坐标系下的位置。
 *
 * 各输入模式的填充规则：
 *  - 相机模式（CameraInputMode）：取帧同时刻 RobotController::getState().strict
 *    填充，底盘 xyz 填 0；
 *  - 视频模式（VideoInputMode）：txt 解析的相机欧拉角 → chassis 欧拉角 且
 *    imu_euler 同值，相机坐标 → 底盘 xyz，其余字段为 0；
 *  - 交互模式（InteractiveInputMode）：全 0。
 */
struct ExtraInputInfo {
    // ── StrictPose 数据包 ──
    double imu_euler_yaw   = 0.0;
    double imu_euler_pitch = 0.0;
    double imu_euler_roll  = 0.0;
    double yaw_pos         = 0.0;   // 反解所用 yaw_pos（wrap 后）
    double pitch_angle     = 0.0;   // 反解所用 pitch_angle（wrap 后）
    double chassis_yaw     = 0.0;
    double chassis_pitch   = 0.0;
    double chassis_roll    = 0.0;
    // ── 底盘世界坐标（米）──
    double chassis_x = 0.0;
    double chassis_y = 0.0;
    double chassis_z = 0.0;
};

/**
 * @brief Abstract interface for input sources.
 *
 * Each concrete implementation provides frames one-by-one
 * via getNextFrame(). Returns false when the source is exhausted
 * or the user requests to quit.
 */
class IInputMode {
public:
    virtual ~IInputMode() = default;

    /**
     * @brief Retrieve the next frame from the input source.
     * @param frame       [out] The captured frame (clone-safe).
     * @param timestamp   [out] The timestamp associated with this frame.
     * @param extra_info [out] Extra input info (StrictPose + chassis xyz).
     * @return true       Frame was retrieved successfully.
     * @return false      No more frames available (source ended or user quit).
     */
    virtual bool getNextFrame(cv::Mat& frame,
                              std::chrono::steady_clock::time_point& timestamp,
                              ExtraInputInfo& extra_info) = 0;

    /**
     * @brief Human-readable name of this input mode (e.g. "Interactive", "Video").
     */
    virtual std::string getName() const = 0;

    /**
     * @brief Returns the recommended inter-frame delay in seconds
     *        (0 = no artificial delay).
     */
    virtual float getFrameDelay() const = 0;
};

#endif // IINPUTMODE_H
