// CameraInputMode.h — 海康相机输入模式（IInputMode 实现）
#ifndef CAMERA_INPUT_MODE_H
#define CAMERA_INPUT_MODE_H

#include "common/Input/IInputMode.h"
#include "common/camera/Camera.h"
#include "RobotController.h"

#include <chrono>
#include <deque>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

/**
 * @brief 实时相机输入模式：包装海康 Camera，取帧同时返回
 *        RobotController 严格反解数据包（StrictPose）打包成的 ExtraInputInfo。
 *
 * - 时间戳由 Camera::getLatestFrame 在成功取帧时刻打上（steady_clock）；
 * - extra_info 来自「延迟状态队列」：后台线程持续调用 rc.getState()（~1kHz），
 *   把采样时刻与对应 ExtraInputInfo 压入队列，并保持队头数据 = 相对当前时刻
 *   extra_info_delay 前的数据（extra_info_delay 来自 config.yaml
 *   common.input_mode.camera_mode.extra_info_delay，必填；0.0 = 最新状态），且至少保留一个数据；
 *   成功取帧时返回队头数据（该帧 + 时间戳 + 延迟对齐的 tf 状态三者对应）；
 *   底盘 xyz 填 0；
 * - 相机取流失败（无新帧）时返回空帧（调用方 sleep 后继续轮询），但时间戳与
 *   extra_info 仍会被更新：时间戳取调用 getNextFrame 的时刻（在相机检查之前
 *   采样，配合 Camera::getLatestFrame 的持锁协议保证返回时间戳单调不减），
 *   extra_info 取该时刻对应的延迟状态（队头），与成功取帧分支使用同一套语义；
 * - 帧间无人工延迟（getFrameDelay = 0），由相机自身帧率驱动。
 */
class CameraInputMode : public IInputMode {
public:
    CameraInputMode(Camera& camera, RobotController& rc);
    ~CameraInputMode() override;

    bool getNextFrame(cv::Mat& frame,
                      std::chrono::steady_clock::time_point& timestamp,
                      ExtraInputInfo& extra_info) override;

    std::string getName() const override { return "Camera"; }

    float getFrameDelay() const override { return 0.0f; }

private:
    // ── extra_info 延迟状态队列 ──
    struct DelayedState {
        std::chrono::steady_clock::time_point ts;   // 采样时刻
        ExtraInputInfo info;                        // 采样时刻的 RobotController 状态
    };

    Camera& camera_;
    RobotController& rc_;

    double extra_info_delay_ = 0.0;          // 秒，来自 config.yaml common.input_mode.camera_mode.extra_info_delay
    std::thread state_thread_;               // 后台采样线程
    std::atomic<bool> state_thread_exit_{false};
    std::deque<DelayedState> state_queue_;   // 时间升序；队头为相对当前时刻 delay 前的数据
    std::mutex state_mtx_;
    std::condition_variable state_cv_;       // getNextFrame 等待首个采样数据

    void stateSamplerLoop();                                  // 后台线程主体
    void trimStateQueueLocked(const std::chrono::steady_clock::time_point& now);  // 需持 state_mtx_
    static ExtraInputInfo stateToExtraInfo(const RobotController::State& st);     // strict → ExtraInputInfo
};

#endif // CAMERA_INPUT_MODE_H
