// CameraInputMode.cpp — 海康相机输入模式实现
#include "common/Input/CameraInputMode.h"
#include "common/RobotConfig.h"

#include <iostream>

CameraInputMode::CameraInputMode(Camera& camera, RobotController& rc)
    : camera_(camera), rc_(rc) {
    // ── extra_info 延迟状态队列：启动后台采样线程 ──
    // 延迟时间必须由配置文件提供（无默认值）
    extra_info_delay_ = RobotConfig::instance().common.inputMode.cameraMode.extraInfoDelay;
    std::cout << "[CameraInputMode] Extra info delayed state queue: ON"
              << " (extra_info_delay = " << extra_info_delay_ << "s)" << std::endl;
    state_thread_ = std::thread(&CameraInputMode::stateSamplerLoop, this);
}

CameraInputMode::~CameraInputMode() {
    state_thread_exit_.store(true, std::memory_order_release);
    state_cv_.notify_all();
    if (state_thread_.joinable()) state_thread_.join();
}

bool CameraInputMode::getNextFrame(cv::Mat& frame,
                                   std::chrono::steady_clock::time_point& timestamp,
                                   ExtraInputInfo& extra_info) {
    // 无新帧时的时间戳取"调用 getNextFrame 的时刻"：必须在相机检查之前采样。
    // 若在检查（持锁）之后采样，写入线程可能在检查与采样之间发布一帧更早打戳的
    // 新帧，导致下一次成功取帧返回的时间戳回退；检查前采样则保证写线程随后
    // 打戳的新帧时间戳必然晚于本调用时刻（配合 Camera::getLatestFrame 的持锁协议）。
    const auto now = std::chrono::steady_clock::now();

    // 相机取流失败（无新帧）时返回空帧，调用方应 sleep 后继续轮询。
    // 无新帧时也必须更新时间戳与 extra_info：时间戳取调用时刻（now），
    // extra_info 取该时刻对应的延迟状态（队头，见下方成功取帧分支的说明）。
    if (!camera_.getLatestFrame(frame, timestamp)) {
        frame.release();  // 标记无新帧
        timestamp = now;
        std::unique_lock<std::mutex> lock(state_mtx_);
        state_cv_.wait(lock, [this]() { return !state_queue_.empty(); });
        trimStateQueueLocked(timestamp);
        extra_info = state_queue_.front().info;
        return true;
    }

    // 成功取帧（有输入源新帧）：行为不变——extra_info 取延迟状态队列队头
    // （= 相对当前时刻 extra_info_delay 前的数据，至少有一个数据；
    // 底盘 xyz 固定为 0，相机模式下底盘视为世界原点）。
    std::unique_lock<std::mutex> lock(state_mtx_);
    state_cv_.wait(lock, [this]() { return !state_queue_.empty(); });
    trimStateQueueLocked(timestamp);
    extra_info = state_queue_.front().info;
    return true;
}

// ==================== extra_info 延迟状态队列 ====================

ExtraInputInfo CameraInputMode::stateToExtraInfo(const RobotController::State& st) {
    // strict 数据包 → ExtraInputInfo；底盘 xyz 保持 0
    ExtraInputInfo info;
    info.imu_euler_yaw   = st.strict.imu_euler_yaw;
    info.imu_euler_pitch = st.strict.imu_euler_pitch;
    info.imu_euler_roll  = st.strict.imu_euler_roll;
    info.yaw_pos         = st.strict.yaw_pos;
    info.pitch_angle     = st.strict.pitch_angle;
    info.chassis_yaw     = st.strict.chassis_yaw;
    info.chassis_pitch   = st.strict.chassis_pitch;
    info.chassis_roll    = st.strict.chassis_roll;
    // chassis_x / chassis_y / chassis_z 保持 0
    return info;
}

void CameraInputMode::trimStateQueueLocked(
    const std::chrono::steady_clock::time_point& now) {
    // 保持队头为相对当前时刻 extra_info_delay 前的数据：
    // 若下一元素（更新）的时间戳 + delay 已不晚于 now，则弹出当前队头；
    // 队列始终至少保留一个数据（size > 1 守卫）。
    // 注：extra_info_delay == 0 时退化为「队头 = 最新采样」。
    while (state_queue_.size() > 1 &&
           state_queue_[1].ts +
                   std::chrono::duration<double>(extra_info_delay_) <= now) {
        state_queue_.pop_front();
    }
}

void CameraInputMode::stateSamplerLoop() {
    while (!state_thread_exit_.load(std::memory_order_acquire)) {
        const RobotController::State st = rc_.getState();
        ExtraInputInfo info = stateToExtraInfo(st);
        const auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(state_mtx_);
            state_queue_.push_back(DelayedState{now, std::move(info)});
            trimStateQueueLocked(now);
        }
        state_cv_.notify_one();
        // 采样频率远高于相机帧率即可（~1kHz），保证延迟队列时间分辨率足够
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
