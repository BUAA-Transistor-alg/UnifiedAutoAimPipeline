// IPipeline.h — 流水线统一接口（OutpostPipeline / PowerRunePipeline 共同实现）
//
// 两个流水线实例在启动时全部构造（模型编译一次），之后不重建；
// 任意时刻只有一个是"激活"状态（接收 addFrame），切换时调用 clear()
// 清空其队列与滤波状态，再开始填充另一条。
#ifndef IPIPELINE_H
#define IPIPELINE_H

#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>

#include "PipelineResult.h"

enum class PipelineMode { OUTPOST, POWER_RUNE };

class IPipeline {
public:
    virtual ~IPipeline() = default;

    virtual PipelineMode mode() const = 0;

    /// 输入帧：构造 PipelineData 加入输入缓冲队列（队列满时阻塞）
    virtual void addFrame(cv::Mat frame,
                          const std::chrono::steady_clock::time_point& frame_timestamp,
                          const ExtraInputInfo& extra_info) = 0;

    /// 提取到时帧：若输出队列队头时间戳与传入时间戳相差 >= max_delay 则返回
    virtual PipelineResult tryPopFrame(
        const std::chrono::steady_clock::time_point& timestamp) = 0;

    /// 清空所有缓冲队列并重置阶段内滤波/预测状态（切换流水线模式时调用）
    virtual void clear() = 0;

    virtual std::string name() const = 0;
};

#endif // IPIPELINE_H
