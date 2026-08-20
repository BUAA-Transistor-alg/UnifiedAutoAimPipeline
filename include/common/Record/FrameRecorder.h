// FrameRecorder.h — 帧录制器：将输入帧 + 时间戳 + ExtraInputInfo 完整保存
//
// 用途（main.cpp input_thread 中使用）：
//   每取到一帧（无论是否成功加入流水线）都调用 recordFrame()，录制器把该帧写入
//   MKV 视频（异步队列），并把对应的时间戳与完整的 ExtraInputInfo（按原样保存）
//   写入同目录的 frame_info txt 文件；同时保存 addFrame 是否成功（accepted）。
//   解析时（VideoInputMode v2 格式）可选择跳过 accepted==0 的帧。
//
// 磁盘保护：录制期间每次写入前检查输出目录所在文件系统的剩余空间，当剩余空间
//   低于配置阈值（min_free_space_bytes）时停止写入并关闭本次录制（不再产生数据）。
//
// 输出目录结构（demo.cpp 风格）：
//   <output_dir>/record_YYYYmmdd_HHMMSS/video_YYYYmmdd_HHMMSS.mkv
//   <output_dir>/record_YYYYmmdd_HHMMSS/frame_info_YYYYmmdd_HHMMSS.txt
#ifndef FRAME_RECORDER_H
#define FRAME_RECORDER_H

#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>

#include "common/Input/IInputMode.h"   // ExtraInputInfo
#include "common/Record/MkvWriter.h"

class FrameRecorder {
public:
    struct Config {
        std::string outputDir;               // 输出目录（相对项目根目录或以 / 开头为绝对路径）
        int64_t minFreeSpaceBytes = 0;       // 剩余空间低于该值（字节）停止写入；<=0 不检查
        double  fps = 30.0;                  // 容器名义帧率（播放器用；回放节奏由 txt 的 dt 决定）
        int64_t bitrate = 8000000;           // 视频码率（bps），默认 8 Mbps（与 demo.cpp 一致）
        size_t  queueSize = 60;              // MkvWriter 异步写入队列长度
    };

    explicit FrameRecorder(Config cfg);
    ~FrameRecorder();

    // 禁止拷贝
    FrameRecorder(const FrameRecorder&) = delete;
    FrameRecorder& operator=(const FrameRecorder&) = delete;

    /// 创建本次录制会话（目录 + 信息文件；视频写入器在首帧到达时才打开，因为
    /// 那时才知道帧尺寸）。失败返回 false（录制被禁用）。
    bool start();

    /// 关闭写入器与信息文件（写视频尾；幂等，可多次调用）
    void close();

    /// 是否处于录制状态（start 成功且未因空间不足 / 打开失败而停止）
    bool active() const;

    /// 是否因剩余空间不足而停止写入
    bool spaceExhausted() const;

    /// 记录一帧（无论 addFrame 是否成功；accepted 表示该帧是否成功加入流水线）。
    /// 视频队列满时该帧被丢弃且不写信息行（保证 txt 与视频帧一一对应），返回 false。
    bool recordFrame(const cv::Mat& frame,
                     const std::chrono::steady_clock::time_point& timestamp,
                     const ExtraInputInfo& extra_info,
                     bool accepted);

    /// 已成功写入视频的信息行数（= 视频帧数）
    size_t recordedFrames() const;

private:
    Config cfg_;

    std::unique_ptr<MkvAllIntraWriter> writer_;
    std::ofstream info_file_;

    std::string output_root_;   // 输出根目录（绝对路径，已创建；空间检查用）
    std::string session_dir_;   // 本次录制输出目录（绝对路径）
    std::string video_path_;
    std::string info_path_;

    bool started_ = false;
    bool stopped_no_space_ = false;

    int64_t frame_index_ = 0;                          // 信息行索引（与视频帧一一对应）
    bool first_written_ = true;                        // 首个写入视频的帧
    std::chrono::steady_clock::time_point last_written_timestamp_;  // 上一次写入视频的时间戳

    /// 检查输出目录所在文件系统剩余空间是否 >= 阈值
    bool checkSpace();
    static std::string timestampString();
};

#endif // FRAME_RECORDER_H
