// FrameRecorder.cpp — 帧录制器实现
#include "Record/FrameRecorder.h"
#include "PathResolver.h"

#include <sys/statvfs.h>

#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>

namespace fs = std::filesystem;

FrameRecorder::FrameRecorder(Config cfg) : cfg_(std::move(cfg)) {}

FrameRecorder::~FrameRecorder() {
    close();
}

bool FrameRecorder::start() {
    // 幂等：若已处于录制状态先关闭
    close();

    // ── 解析输出目录（相对路径以项目根目录为基准）──
    std::string out_dir = cfg_.outputDir.empty() ? "recordings" : cfg_.outputDir;
    if (!out_dir.empty() && out_dir[0] != '/') {
        out_dir = PathResolver::resolvePath(out_dir);
    }

    // ── 确保输出根目录存在（空间检查需要 statvfs 一个存在的路径）──
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) {
        std::cerr << "[FrameRecorder] 无法创建输出目录 " << out_dir
                  << ": " << ec.message() << std::endl;
        return false;
    }
    output_root_ = out_dir;

    // ── 剩余空间检查：不足则拒绝开始录制（此时尚未创建会话目录，不留下空目录）──
    if (!checkSpace()) {
        std::cerr << "[FrameRecorder] 剩余空间不足（阈值 " << cfg_.minFreeSpaceBytes
                  << " 字节），拒绝开始录制。" << std::endl;
        output_root_.clear();
        return false;
    }

    // ── 创建本次会话子目录：<output_dir>/record_YYYYmmdd_HHMMSS ──
    const std::string ts = timestampString();
    session_dir_ = (fs::path(out_dir) / ("record_" + ts)).string();
    ec.clear();
    fs::create_directories(session_dir_, ec);
    if (ec) {
        std::cerr << "[FrameRecorder] 无法创建输出目录 " << session_dir_
                  << ": " << ec.message() << std::endl;
        session_dir_.clear();
        output_root_.clear();
        return false;
    }

    video_path_ = (fs::path(session_dir_) / ("video_" + ts + ".mkv")).string();
    info_path_  = (fs::path(session_dir_) / ("frame_info_" + ts + ".txt")).string();

    // ── 信息文件（含表头，标注 v2 格式列含义）──
    info_file_.open(info_path_, std::ios::out | std::ios::trunc);
    if (!info_file_.is_open()) {
        std::cerr << "[FrameRecorder] 无法打开信息文件 " << info_path_ << std::endl;
        session_dir_.clear();
        return false;
    }
    info_file_ << "# UnifiedAutoAim frame_info v2\n";
    info_file_ << "# columns: frame_index dt_s timestamp_s accepted"
                  " imu_euler_yaw imu_euler_pitch imu_euler_roll yaw_pos pitch_angle"
                  " chassis_yaw chassis_pitch chassis_roll chassis_x chassis_y chassis_z\n";

    started_ = true;
    stopped_no_space_ = false;
    frame_index_ = 0;
    first_written_ = true;

    std::cout << "[FrameRecorder] 录制开始，输出目录: " << session_dir_ << std::endl;
    std::cout << "  video : " << video_path_ << std::endl;
    std::cout << "  info  : " << info_path_ << std::endl;
    std::cout << "  剩余空间阈值: "
              << (cfg_.minFreeSpaceBytes > 0 ? std::to_string(cfg_.minFreeSpaceBytes) : std::string("不检查"))
              << " 字节" << std::endl;
    return true;
}

void FrameRecorder::close() {
    if (writer_) {
        writer_->close();
        writer_.reset();
    }
    if (info_file_.is_open()) {
        info_file_.close();
    }
    if (started_) {
        std::cout << "[FrameRecorder] 录制结束，共写入 " << frame_index_ << " 帧，"
                  << "输出目录: " << session_dir_ << std::endl;
    }
    started_ = false;
}

bool FrameRecorder::active() const {
    return started_ && !stopped_no_space_;
}

bool FrameRecorder::spaceExhausted() const {
    return stopped_no_space_;
}

size_t FrameRecorder::recordedFrames() const {
    return static_cast<size_t>(frame_index_);
}

bool FrameRecorder::recordFrame(const cv::Mat& frame,
                                const std::chrono::steady_clock::time_point& timestamp,
                                const ExtraInputInfo& extra_info,
                                bool accepted) {
    if (!started_ || stopped_no_space_) return false;

    // ── 剩余空间检查：低于阈值 → 停止写入并关闭本次录制 ──
    if (cfg_.minFreeSpaceBytes > 0 && !checkSpace()) {
        stopped_no_space_ = true;
        close();
        std::cerr << "[FrameRecorder] 剩余空间低于阈值（" << cfg_.minFreeSpaceBytes
                  << " 字节），停止录制。输出目录: " << session_dir_ << std::endl;
        return false;
    }

    // ── 首帧：打开视频写入器（此时才知道帧尺寸）──
    if (!writer_) {
        writer_ = std::make_unique<MkvAllIntraWriter>(cfg_.queueSize);
        if (!writer_->open(video_path_, frame.cols, frame.rows, cfg_.fps, cfg_.bitrate)) {
            std::cerr << "[FrameRecorder] 打开视频写入器失败: " << video_path_
                      << "，录制已禁用。" << std::endl;
            started_ = false;
            close();
            return false;
        }
    }

    // ── 统一为 3 通道 BGR（swscale 需要 BGR24）──
    cv::Mat bgr = frame;
    if (frame.channels() == 1) {
        cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, bgr, cv::COLOR_RGBA2BGR);
    }

    // ── dt = 与上一个「已写入视频」帧的时间间隔（秒）──
    double dt = 0.0;
    if (!first_written_) {
        dt = std::chrono::duration<double>(timestamp - last_written_timestamp_).count();
    }

    // ── 写视频帧（异步队列，满则丢弃）──
    bool write_ok = writer_->writeFrame(bgr, true);
    if (!write_ok) {
        // 视频队列满被丢弃：不写信息行（保证 txt 与视频帧一一对应）
        return false;
    }
    last_written_timestamp_ = timestamp;
    first_written_ = false;

    // ── 写信息行：frame_index dt timestamp accepted + 完整 ExtraInputInfo ──
    const double ts_s =
        std::chrono::duration<double>(timestamp.time_since_epoch()).count();
    info_file_ << std::fixed << std::setprecision(9)
               << frame_index_ << " "
               << dt << " "
               << ts_s << " "
               << (accepted ? 1 : 0) << " "
               << std::setprecision(12)
               << extra_info.imu_euler_yaw << " "
               << extra_info.imu_euler_pitch << " "
               << extra_info.imu_euler_roll << " "
               << extra_info.yaw_pos << " "
               << extra_info.pitch_angle << " "
               << extra_info.chassis_yaw << " "
               << extra_info.chassis_pitch << " "
               << extra_info.chassis_roll << " "
               << extra_info.chassis_x << " "
               << extra_info.chassis_y << " "
               << extra_info.chassis_z << "\n";
    ++frame_index_;
    return true;
}

bool FrameRecorder::checkSpace() {
    if (cfg_.minFreeSpaceBytes <= 0) return true;

    // 优先检查输出根目录（会话目录与它在同一文件系统，统计结果一致）
    const std::string& dir = output_root_.empty() ? session_dir_ : output_root_;
    if (dir.empty()) return true;

    struct statvfs st;
    if (statvfs(dir.c_str(), &st) != 0) {
        std::cerr << "[FrameRecorder] statvfs 失败: " << dir << std::endl;
        return true;  // 无法获取剩余空间 → 不阻止录制（避免误停）
    }
    const uint64_t free_bytes =
        static_cast<uint64_t>(st.f_bavail) * static_cast<uint64_t>(st.f_frsize);
    return free_bytes >= static_cast<uint64_t>(cfg_.minFreeSpaceBytes);
}

std::string FrameRecorder::timestampString() {
    std::time_t now = std::time(nullptr);
    std::tm* lt = std::localtime(&now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", lt);
    return buf;
}
