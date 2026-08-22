#ifndef VIDEOINPUTMODE_H
#define VIDEOINPUTMODE_H

#include "common/Input/IInputMode.h"
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @brief Video-file input mode.
 *
 * Reads frames sequentially from a video file.
 * When the video ends, it automatically loops back to the first frame.
 *
 * Optionally reads an extra-info text file that supplies per-frame
 * camera position (x, y, z) and Euler angles (yaw, pitch, roll)
 * plus inter-frame dt values.
 *
 * 支持两种 extra-info 文件格式（自动探测，表头/注释行以 '#' 开头被忽略）：
 *  - v1（原格式，8 列）：frame_index dt x y z yaw pitch roll
 *    （相机位姿 → chassis 欧拉角 + 底盘 xyz，其余字段为 0）
 *  - v2（录制器 FrameRecorder 输出，15 列）：
 *    frame_index dt timestamp_s accepted
 *    imu_euler_yaw imu_euler_pitch imu_euler_roll yaw_pos pitch_angle
 *    chassis_yaw chassis_pitch chassis_roll chassis_x chassis_y chassis_z
 *    （完整 ExtraInputInfo 按原样还原；accepted 表示该帧当时是否成功加入流水线）
 *
 * 当 skip_unaccepted_frames 为 true 时，accepted==0 的帧（录制时未成功加入
 * 流水线的帧）在读取时被跳过（不返回给调用方），时间轴仍按全部帧累计。
 */
class VideoInputMode : public IInputMode {
public:
    /**
     * @param video_path        Path to the video file.
     * @param extra_info_path  Optional path to extra info txt file.
     * @param skip_unaccepted_frames 为 true 时跳过 accepted==0 的帧（仅 v2 格式有效）
     * @param test_max_fps    为 true 时 getFrameDelay() 返回 0（不做按视频帧率的
     *                        节流），用于测量视频输入 + 流水线的最大帧数/FPS。
     * @throws std::runtime_error if the video cannot be opened.
     */
    explicit VideoInputMode(const std::string& video_path,
                            const std::string& extra_info_path = "",
                            bool skip_unaccepted_frames = false,
                            bool test_max_fps = false);

    bool getNextFrame(cv::Mat& frame,
                      std::chrono::steady_clock::time_point& timestamp,
                      ExtraInputInfo& extra_info) override;
    std::string getName() const override { return "Video"; }
    float getFrameDelay() const override;

private:
    struct ExtraFrameEntry {
        float dt = 0.0f;
        // ── v1 格式（8 列）：相机位姿 ──
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float yaw = 0.0f;
        float pitch = 0.0f;
        float roll = 0.0f;
        // ── v2 格式（15 列）：完整 ExtraInputInfo + accepted + 绝对时间戳 ──
        bool has_full_info = false;   // true 表示该条目为 v2 格式
        bool accepted = true;         // 该帧当时是否成功加入流水线（addFrame 返回值）
        double timestamp = 0.0;       // 绝对时间戳（steady_clock 秒）
        ExtraInputInfo full_info;     // 完整额外信息（按原样还原）
    };

    cv::VideoCapture cap_;
    double fps_;
    std::string video_path_;
    bool first_frame_ = true;
    std::chrono::steady_clock::time_point start_time_;
    int64_t loop_frame_index_ = 0;
    int64_t total_frame_index_ = 0;

    // Extra info file support
    std::string extra_info_path_;
    std::unordered_map<int, ExtraFrameEntry> extra_info_map_;
    int extra_info_max_frame_ = -1;  // max frame_index_ in file
    ExtraFrameEntry last_valid_entry_;      // recent valid info frame
    bool last_valid_entry_initialized_ = false;
    float next_dt_ = 0.0f;                   // pre-extracted dt for the next frame
    double accumulated_time_ = 0.0;          // accumulated dt sum for timestamp
    bool warned_overflow_ = false;           // only warn once per loop about overflow

    bool skip_unaccepted_frames_ = false;    // 跳过 accepted==0 的帧（v2）
    bool format_v2_ = false;                 // extra-info 文件是否为 v2 格式（首个数据行探测）
    bool test_max_fps_ = false;              // 测试最大帧率：开启时 getFrameDelay() 返回 0

    void reOpenVideoFile();
    void parseExtraInputInfoFile();
};

#endif // VIDEOINPUTMODE_H