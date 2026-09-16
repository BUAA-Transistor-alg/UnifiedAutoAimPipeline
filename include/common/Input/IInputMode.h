#ifndef IINPUTMODE_H
#define IINPUTMODE_H

#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>

#include "common/RobotConfig.h"   // YawMode

/**
 * @brief 与一帧图像绑定的额外信息（**按 yaw 构型分两包** + 共用底盘位姿）。
 *
 * ⚠ 给后续修改者（构型分离约定，避免静默错误）：
 *   - 单 yaw 构型（YawMode::SINGLE，config common.big_small_yaw.mode = single）
 *     与大小 yaw 构型（YawMode::BIG_SMALL）**关节角的物理含义完全不同**：
 *       SINGLE    ：仅一个 yaw 关节角（相对底盘），imu_euler 为“IMU 在头上”语义；
 *       BIG_SMALL ：大 yaw 关节角 θ_big（相对底盘）+ 小 yaw 关节角 θ_small（相对大 yaw），
 *                   imu_euler 为“IMU 在大 yaw 转子上”语义（见子模组 StrictPose）。
 *     因此**两套信息分别打包**（single / big_small），**当前构型用哪一包就只填哪一包**，
 *     另一包**整体置 NaN**（默认构造即为 NaN）：下游任何按错包取值的代码都会立刻
 *     因为 NaN 暴露出来，而不是静默用错语义的角度算出一个看似合理的姿态。
 *   - 底盘位姿（chassis_*）两构型语义相同（底盘在世界系下的位姿），共用，不打包。
 *   - 判断“本帧当前构型包是否可用”用 currentPackValid(mode)；取包用 currentPack(mode)。
 *
 * 各输入模式的填充规则：
 *   - 相机模式（CameraInputMode）：取帧同时刻控制器状态（旧构型 tcs::RobotController::
 *     State::strict / 新构型 tcbs 状态适配器）填充当前构型包，底盘 xyz 填 0；
 *   - 视频模式（VideoInputMode）：录制文件 txt 解析；v2 记录只有单 yaw 包，
 *     v3 记录两包都有（未使用的一包记录为 NaN）；
 *   - 交互模式（InteractiveInputMode）：当前构型包全 0（无云台数据）。
 */
struct ExtraInputInfo {
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    // ── 单 yaw 构型包（旧：chassis -> yaw -> pitch -> head）──
    // 语义与 tcs::RobotController::StrictPose 一致（角度 wrap 到 (-π, π]，
    // 缺失数据以 0 参与），保证 R_imu = R_chassis·Rz(yaw_pos)·Rx(pitch_angle) 恒成立。
    struct SingleYawPose {
        double yaw_pos         = kNaN;   // yaw 关节角（相对底盘，wrap 后）
        double pitch_angle     = kNaN;   // pitch 关节角
        double imu_euler_yaw   = kNaN;   // IMU 原始欧拉角（IMU 装在头上）
        double imu_euler_pitch = kNaN;
        double imu_euler_roll  = kNaN;

        bool valid() const {
            return std::isfinite(yaw_pos) && std::isfinite(pitch_angle) &&
                   std::isfinite(imu_euler_yaw) && std::isfinite(imu_euler_pitch) &&
                   std::isfinite(imu_euler_roll);
        }
    };

    // ── 大小 yaw 构型包（新：chassis -> yaw_big -> yaw_small -> pitch -> head）──
    // 语义与子模组 tcbs::dual_yaw::StrictPose 一致（IMU 固定在大 yaw 转子上时：
    // R_world_imu = R_chassis·Rz(θ_big)·R_A_IMU）。
    struct BigSmallYawPose {
        double yaw_big_pos     = kNaN;   // 大 yaw 关节角 θ_big（相对底盘，wrap 后）
        double yaw_small_pos   = kNaN;   // 小 yaw 关节角 θ_small（相对大 yaw，wrap 后）
        double pitch_angle     = kNaN;   // pitch 关节角
        double imu_euler_yaw   = kNaN;   // IMU 原始欧拉角（IMU 装在大 yaw 转子上）
        double imu_euler_pitch = kNaN;
        double imu_euler_roll  = kNaN;

        bool valid() const {
            return std::isfinite(yaw_big_pos) && std::isfinite(yaw_small_pos) &&
                   std::isfinite(pitch_angle) && std::isfinite(imu_euler_yaw) &&
                   std::isfinite(imu_euler_pitch) && std::isfinite(imu_euler_roll);
        }
    };

    // ── 共用：底盘位姿（两构型语义相同）──
    double chassis_yaw   = 0.0;
    double chassis_pitch = 0.0;
    double chassis_roll  = 0.0;
    // 底盘世界坐标（米）
    double chassis_x = 0.0;
    double chassis_y = 0.0;
    double chassis_z = 0.0;

    // ── 构型相关两包（**未使用的包整体为 NaN**）──
    SingleYawPose   single;
    BigSmallYawPose big_small;

    /// 当前构型包是否可用（无云台数据的输入模式填 0 也算可用；NaN 表示“未填/不适用”）
    bool currentPackValid(YawMode mode) const {
        return (mode == YawMode::SINGLE) ? single.valid() : big_small.valid();
    }

    /// 无云台数据时，把**当前构型**的包显式填 0（等价于“关节角全 0”），另一包保持 NaN
    void fillCurrentPackZeros(YawMode mode) {
        if (mode == YawMode::SINGLE) {
            single = SingleYawPose{0.0, 0.0, 0.0, 0.0, 0.0};
        } else {
            big_small = BigSmallYawPose{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }
    }
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
     *                    返回空 Mat（frame.empty()）表示输入源当前无新帧：
     *                    调用方应短暂 sleep 后继续轮询（区别于返回 false 表示源已耗尽）。
     * @param timestamp   [out] The timestamp associated with this frame.
     *                    即使无新帧也会被更新（见下方约定）。
     * @param extra_info [out] Extra input info (按构型分包的关节角 + 共用底盘位姿).
     *                    即使无新帧也会被更新（见下方约定）。
     * @return true       Frame was retrieved successfully (frame 可能为空 = 无新帧)。
     * @return false      No more frames available (source ended or user quit).
     *
     * 无新帧（返回空 frame）时的时间戳约定：
     *  - CameraInputMode：时间戳取调用 getNextFrame 的时刻（now），extra_info 取
     *    该时刻对应的延迟状态（有输入源新帧时行为不变）；
     *  - 其余输入方式：若时间戳来自 now（如 InteractiveInputMode），则重新取 now；
     *    否则（时间戳来自累计时间，如 VideoInputMode）沿用上一次成功返回的时间。
     */
    virtual bool getNextFrame(cv::Mat& frame,
                              std::chrono::steady_clock::time_point& timestamp,
                              ExtraInputInfo& extra_info) = 0;

    /**
     * @brief Human-readable name of the input mode (e.g. "Interactive", "Video").
     */
    virtual std::string getName() const = 0;

    /**
     * @brief Returns the recommended inter-frame delay in seconds
     *        (0 = no artificial delay).
     */
    virtual float getFrameDelay() const = 0;
};

#endif // IINPUTMODE_H
