// MkvAllIntraWriter.h (修改后)
#ifndef MKV_ALL_INTRA_WRITER_H
#define MKV_ALL_INTRA_WRITER_H

#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/error.h>
#include <libswscale/swscale.h>
}

#include <opencv2/opencv.hpp>

class MkvAllIntraWriter {
public:
    MkvAllIntraWriter(size_t maxQueueSize = 30);
    ~MkvAllIntraWriter();

    /**
     * 打开输出文件并初始化编码器
     * @param filename 输出文件名（建议 .mkv）
     * @param width    视频宽度
     * @param height   视频高度
     * @param fps      帧率
     * @param bitrate  编码码率（bps），例如 8000000 表示 8 Mbps
     * @return true 成功，false 失败
     */
    bool open(const std::string& filename, int width, int height, double fps, int64_t bitrate);

    /**
     * 异步写入一帧 cv::Mat（BGR 格式）
     * @param bgrMat       OpenCV Mat，尺寸需与构造时一致
     * @param dropWhenFull 当队列已满时是否丢弃该帧
     * @param cloneFrame   true: 入队前深拷贝（调用方之后仍会使用该 Mat）；
     *                     false: 移动入队（调用方不再使用，省一次全帧拷贝）
     * @return true 成功推入队列，false 失败（如错误状态或关闭）或丢弃（如果允许）
     */
    bool writeFrame(cv::Mat bgrMat, bool dropWhenFull = true, bool cloneFrame = true);

    /**
     * 关闭写入器，停止后台线程并写入文件尾
     */
    void close();

    /**
     * 获取当前写入队列长度
     * @return 队列中等待写入的帧数
     */
    size_t getQueueSize() const;

private:
    // FFmpeg 相关（仅在后台线程使用）
    AVFormatContext* fmtCtx_;
    AVCodecContext*  codecCtx_;
    SwsContext*      swsCtx_;
    AVStream*        videoStream_;
    AVFrame*         frame_;

    // 异步队列与线程
    size_t maxQueueSize_;
    std::queue<cv::Mat> frameQueue_;
    std::thread worker_;
    mutable std::mutex mtx_;
    std::condition_variable cvNotFull_;
    std::condition_variable cvNotEmpty_;
    std::atomic<bool> stop_;
    std::atomic<bool> error_;
    std::atomic<int64_t> frameCount_;   // 仅用于 PTS，需原子操作或加锁

    // 内部函数：后台线程主循环
    void encodingLoop();

    // 内部函数：实际编码并写入一帧（从 AVFrame 中读取 YUV 数据）
    bool encodeAndWriteFrame(AVFrame* yuvFrame);
};

#endif