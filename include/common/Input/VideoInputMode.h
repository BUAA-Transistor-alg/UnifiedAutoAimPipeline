#ifndef VIDEOINPUTMODE_H
#define VIDEOINPUTMODE_H

#include "IInputMode.h"
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
 * Optionally reads a Extra info text file that supplies per-frame
 * camera position (x, y, z) and Euler angles (yaw, pitch, roll)
 * plus inter-frame dt values.
 */
class VideoInputMode : public IInputMode {
public:
    /**
     * @param video_path        Path to the video file.
     * @param extra_info_path  Optional path to extra info txt file.
     * @throws std::runtime_error if the video cannot be opened.
     */
    explicit VideoInputMode(const std::string& video_path,
                            const std::string& extra_info_path = "");

    bool getNextFrame(cv::Mat& frame,
                      std::chrono::steady_clock::time_point& timestamp,
                      ExtraInputInfo& extra_info) override;
    std::string getName() const override { return "Video"; }
    float getFrameDelay() const override;

private:
    struct ExtraFrameEntry {
        float dt     = 0.0f;
        float x      = 0.0f;
        float y      = 0.0f;
        float z      = 0.0f;
        float yaw    = 0.0f;
        float pitch  = 0.0f;
        float roll   = 0.0f;
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

    void reOpenVideoFile();
    void parseExtraInputInfoFile();
};

#endif // VIDEOINPUTMODE_H