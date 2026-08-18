#include "Input/VideoInputMode.h"
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cctype>

using namespace std;

VideoInputMode::VideoInputMode(const string& video_path,
                               const string& extra_info_path,
                               bool skip_unaccepted_frames)
    : video_path_(video_path), extra_info_path_(extra_info_path),
      skip_unaccepted_frames_(skip_unaccepted_frames) {
    cap_.open(video_path);
    if (!cap_.isOpened()) {
        throw runtime_error("Failed to open video file: " + video_path);
    }
    fps_ = cap_.get(cv::CAP_PROP_FPS);
    if (fps_ <= 0.0) fps_ = 30.0;
    cout << "[VideoInputMode] Opened: " << video_path
         << " | FPS: " << fps_ << endl;

    // Initialize default dt from fps
    next_dt_ = 1.0f / static_cast<float>(fps_);

    if (!extra_info_path_.empty()) {
        parseExtraInputInfoFile();
    }
}

void VideoInputMode::parseExtraInputInfoFile() {
    ifstream file(extra_info_path_);
    if (!file.is_open()) {
        cerr << "[VideoInputMode] WARNING: Cannot open Extra info file: "
             << extra_info_path_ << endl;
        return;
    }

    int line_no = 0;
    string line;
    bool format_detected = false;   // 首个非注释数据行决定文件格式
    format_v2_ = false;
    while (getline(file, line)) {
        line_no++;

        // Trim leading/trailing whitespace
        size_t start = 0;
        while (start < line.size() && isspace(static_cast<unsigned char>(line[start])))
            start++;
        size_t end = line.size();
        while (end > start && isspace(static_cast<unsigned char>(line[end - 1])))
            end--;
        if (start >= end) continue;  // empty line

        string trimmed = line.substr(start, end - start);
        if (trimmed[0] == '#') continue;  // 注释 / 表头

        // Split by whitespace (space or tab)
        istringstream iss(trimmed);
        vector<string> tokens;
        string token;
        while (iss >> token) {
            tokens.push_back(token);
        }

        // 格式探测：v2 = 15 列（frame_index dt timestamp accepted + 11 个 ExtraInputInfo 字段）；
        // v1 = 原 8 列（frame_index dt x y z yaw pitch roll）
        if (!format_detected) {
            format_v2_ = (tokens.size() >= 15);
            format_detected = true;
        }

        if (format_v2_) {
            if (tokens.size() < 15) {
                cerr << "[VideoInputMode] WARNING: Extra info line " << line_no
                     << " (v2) has insufficient fields (" << tokens.size()
                     << " < 15), skipping: " << trimmed << endl;
                continue;
            }
            ExtraFrameEntry entry;
            try {
                int frame_idx = stoi(tokens[0]);
                entry.dt       = stof(tokens[1]);
                entry.timestamp = stod(tokens[2]);
                entry.accepted = (stoi(tokens[3]) != 0);
                entry.has_full_info = true;
                entry.full_info.imu_euler_yaw   = stod(tokens[4]);
                entry.full_info.imu_euler_pitch = stod(tokens[5]);
                entry.full_info.imu_euler_roll  = stod(tokens[6]);
                entry.full_info.yaw_pos         = stod(tokens[7]);
                entry.full_info.pitch_angle     = stod(tokens[8]);
                entry.full_info.chassis_yaw     = stod(tokens[9]);
                entry.full_info.chassis_pitch   = stod(tokens[10]);
                entry.full_info.chassis_roll    = stod(tokens[11]);
                entry.full_info.chassis_x       = stod(tokens[12]);
                entry.full_info.chassis_y       = stod(tokens[13]);
                entry.full_info.chassis_z       = stod(tokens[14]);

                extra_info_map_[frame_idx] = entry;
                if (frame_idx > extra_info_max_frame_) {
                    extra_info_max_frame_ = frame_idx;
                }
            } catch (const std::exception& e) {
                cerr << "[VideoInputMode] WARNING: Extra info line " << line_no
                     << " parse error: " << e.what() << ", skipping: " << trimmed << endl;
            }
        } else {
            // v1（原格式）
            if (tokens.size() < 8) {
                cerr << "[VideoInputMode] WARNING: Extra info line " << line_no
                     << " has insufficient fields (" << tokens.size()
                     << " < 8), skipping: " << trimmed << endl;
                continue;
            }
            ExtraFrameEntry entry;
            try {
                int frame_idx = stoi(tokens[0]);
                entry.dt    = stof(tokens[1]);
                entry.x     = stof(tokens[2]);
                entry.y     = stof(tokens[3]);
                entry.z     = stof(tokens[4]);
                entry.yaw   = stof(tokens[5]);
                entry.pitch = stof(tokens[6]);
                entry.roll  = stof(tokens[7]);

                extra_info_map_[frame_idx] = entry;
                if (frame_idx > extra_info_max_frame_) {
                    extra_info_max_frame_ = frame_idx;
                }
            } catch (const std::exception& e) {
                cerr << "[VideoInputMode] WARNING: Extra info line " << line_no
                     << " parse error: " << e.what() << ", skipping: " << trimmed << endl;
            }
        }
    }

    cout << "[VideoInputMode] Loaded " << extra_info_map_.size()
         << " Extra info entries (format v" << (format_v2_ ? "2" : "1")
         << ", max frame index: " << extra_info_max_frame_ << ")";
    if (skip_unaccepted_frames_) {
        cout << " [skip unaccepted frames: ON]";
    }
    cout << endl;
}

float VideoInputMode::getFrameDelay() const {
    return next_dt_;
}

bool VideoInputMode::getNextFrame(cv::Mat& frame,
                                  std::chrono::steady_clock::time_point& timestamp,
                                  ExtraInputInfo& extra_info) {
    while (true) {
        // ── 读一帧视频（含到达末尾时的循环重绕）──
        if (!cap_.read(frame) || frame.empty()) {
            // Reached end of video — loop back to start
            cout << "[VideoInputMode] End of video reached, looping to start." << endl;
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            loop_frame_index_ = 0;
            warned_overflow_ = false;

            // Check if loop_frame_index_ (0) exceeds extra_info_max_frame_
            if (!extra_info_map_.empty() && loop_frame_index_ > extra_info_max_frame_) {
                cerr << "[VideoInputMode] WARNING: loop_frame_index_ " << loop_frame_index_
                     << " exceeds Extra info max frame " << extra_info_max_frame_
                     << ", Extra info will fall back to last valid." << endl;
            }

            // Read the first frame after rewind
            if (!cap_.read(frame) || frame.empty()) {
                cerr << "[VideoInputMode] Failed to read frame after rewind, try reopen." << endl;
                reOpenVideoFile();

                if (!cap_.read(frame) || frame.empty()) {
                    cerr << "[VideoInputMode] Failed" << endl;
                    return false;
                }
            }
        }

        if (first_frame_) {
            start_time_ = std::chrono::steady_clock::now();
            first_frame_ = false;
        }

        // --- Handle loop reset of Extra info (frame 0 invalid -> reset defaults) ---
        // Must be done BEFORE extra_info lookup so frame 0 gets the reset values.
        if (loop_frame_index_ == 0 && !extra_info_map_.empty()) {
            auto it0 = extra_info_map_.find(0);
            if (it0 == extra_info_map_.end()) {
                float default_dt = 1.0f / static_cast<float>(fps_);
                last_valid_entry_ = ExtraFrameEntry{};
                last_valid_entry_.dt = default_dt;
                last_valid_entry_initialized_ = true;
            }
        }

        // --- Populate extra_info from map ---
        const ExtraFrameEntry* current_entry = nullptr;   // 当前帧索引在 map 中的条目（若有）
        if (!extra_info_map_.empty()) {
            auto it = extra_info_map_.find(static_cast<int>(loop_frame_index_));
            if (it != extra_info_map_.end()) {
                last_valid_entry_ = it->second;
                last_valid_entry_initialized_ = true;
                current_entry = &it->second;
            } else {
                if (last_valid_entry_initialized_) {
                    cerr << "[VideoInputMode] WARNING: No Extra info for frame "
                         << loop_frame_index_ << ", using last valid entry." << endl;
                } else {
                    float default_dt = 1.0f / static_cast<float>(fps_);
                    last_valid_entry_ = ExtraFrameEntry{};
                    last_valid_entry_.dt = default_dt;
                    last_valid_entry_initialized_ = true;
                    cerr << "[VideoInputMode] WARNING: No Extra info for frame "
                         << loop_frame_index_ << ", no prior valid entry, using defaults."
                         << endl;
                }
            }

            if (last_valid_entry_.has_full_info) {
                // v2：完整 ExtraInputInfo 按原样还原
                extra_info = last_valid_entry_.full_info;
            } else {
                // v1：解析出的相机欧拉角 → chassis 欧拉角 且 imu_euler 同值，
                // 相机坐标 → 底盘 xyz，其余字段（yaw_pos/pitch_angle 等）为 0。
                extra_info = ExtraInputInfo{};
                extra_info.chassis_x     = last_valid_entry_.x;
                extra_info.chassis_y     = last_valid_entry_.y;
                extra_info.chassis_z     = last_valid_entry_.z;
                extra_info.chassis_yaw   = last_valid_entry_.yaw;
                extra_info.chassis_pitch = last_valid_entry_.pitch;
                extra_info.chassis_roll  = last_valid_entry_.roll;
                extra_info.imu_euler_yaw   = last_valid_entry_.yaw;
                extra_info.imu_euler_pitch = last_valid_entry_.pitch;
                extra_info.imu_euler_roll  = last_valid_entry_.roll;
            }
        } else {
            extra_info = ExtraInputInfo{};
        }

        // --- Check overflow warning ---
        if (!extra_info_map_.empty() && !warned_overflow_) {
            if (loop_frame_index_ > extra_info_max_frame_) {
                cerr << "[VideoInputMode] WARNING: loop_frame_index_ " << loop_frame_index_
                     << " exceeds Extra info max frame " << extra_info_max_frame_ << endl;
                warned_overflow_ = true;
            }
        }

        // --- dt 累计（无论该帧是否被跳过都累计，保证全局时间轴一致）---
        float current_dt;
        if (!extra_info_map_.empty()) {
            current_dt = last_valid_entry_.dt;
        } else {
            current_dt = 1.0f / static_cast<float>(fps_);
        }
        accumulated_time_ += current_dt;

        // --- 记录当前帧索引并递增 ---
        const int cur_index = static_cast<int>(loop_frame_index_);
        loop_frame_index_++;
        total_frame_index_++;

        // --- Pre-extract dt for next frame ---
        if (!extra_info_map_.empty()) {
            auto next_it = extra_info_map_.find(static_cast<int>(loop_frame_index_));
            if (next_it != extra_info_map_.end()) {
                next_dt_ = next_it->second.dt;
            }

            // If looping and frame 1 info is invalid, reset next_dt_
            if (loop_frame_index_ == 1) {
                auto it1 = extra_info_map_.find(1);
                if (it1 == extra_info_map_.end()) {
                    next_dt_ = 1.0f / static_cast<float>(fps_);
                }
            }
        }

        // --- 可选：跳过录制时未成功加入流水线的帧（v2 且 accepted==0）---
        if (skip_unaccepted_frames_) {
            bool skip = false;
            if (current_entry != nullptr && current_entry->has_full_info &&
                !current_entry->accepted) {
                skip = true;
            }
            if (skip) {
                continue;   // 读取下一视频帧（时间已累计，时间戳保持全局一致）
            }
        }

        // --- 计算时间戳：start_time_ + 累计 dt ---
        auto delta_ns = std::chrono::nanoseconds(
            static_cast<int64_t>(accumulated_time_ * 1e9));
        timestamp = start_time_ + delta_ns;

        return true;
    }
}

void VideoInputMode::reOpenVideoFile() {
    cap_.release();

    cap_.open(video_path_);
    if (!cap_.isOpened()) {
        throw runtime_error("Failed to open video file: " + video_path_);
    }
    fps_ = cap_.get(cv::CAP_PROP_FPS);
    if (fps_ <= 0.0) fps_ = 30.0;
    cout << "[VideoInputMode] Opened: " << video_path_
         << " | FPS: " << fps_ << endl;
}
