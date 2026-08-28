#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

// 单类（label 7~8，基地/基地大装甲）首物体位姿保持：只保存最近一次
// processFrame 传入物体列表中第一个有效物体的位姿，内部设有最小识别帧数
// 与重置时间（无观测超时自动重置）。capturePosePredictor 返回的预测器
// 始终返回该物体最新位置（仅 1 个物体），与 ClassEKF 一起参与最近目标选择。
namespace sp_ekf {

class FirstObjectTracker {
public:
    using TimePoint = std::chrono::steady_clock::time_point;

    // 可调参数
    struct Params {
        int    minDetectCount = 5;    // 最小识别帧数（累计达到后才视为有效）
        double resetTimeoutSec = 2.0; // 连续无观测多久后重置（秒）

        Params()
            : minDetectCount(5),
              resetTimeoutSec(2.0) {}
    };

    explicit FirstObjectTracker(const Params& params = Params{});

    /**
     * @brief 统一帧处理：只取本帧物体列表中第一个有效物体保存其位姿。
     *
     * world_pos 与 world_euler 一一对应（每物体的世界坐标（米）/ 世界欧拉角（弧度）），
     * 可为空（本帧无该类观测）。内部逻辑：
     *  - 有有效观测（world_pos 非 (0,0,0)；PnP 失败填充的零观测自动跳过）：
     *      取第一个有效物体，保存其世界坐标与旋转矩阵（由欧拉角构造），累计识别帧数；
     *  - 无观测：连续超过 resetTimeoutSec → 重置（帧数与有效标志清零，下次观测重新累计）；
     *      未超时保持当前位姿与有效状态。
     * @return 处理结束后本类是否有效（累计识别帧数 ≥ minDetectCount 且未超时重置）
     */
    bool processFrame(const std::vector<cv::Vec3f>& world_pos,
                      const std::vector<cv::Vec3f>& world_euler,
                      const TimePoint& t);

    /// 清空帧数与位姿状态（下次观测重新累计）。
    void reset();

    /// 是否有效（识别帧数达标且未超时重置）。
    bool valid() const { return valid_; }

    /// 保存的物体世界坐标（米）。仅 valid() 时有效。
    const cv::Vec3d& getPosition() const { return position_; }

    /// 保存的物体旋转矩阵（CV_64F，world 系，由欧拉角构造）。仅 valid() 时有效。
    cv::Mat getRotationMatrix() const { return R_; }

    /**
     * @brief 快照预测器：valid() 时返回仅含该物体最新位置的预测器。
     *
     * 返回的函数签名为 std::vector<cv::Point3f>(double dt)：始终返回保存的
     * 最新位置（world，米），仅 1 个物体；无运动模型，dt 参数不参与外推。
     * 快照复制调用时刻的 position_，不随后续 processFrame 改变。
     *
     * @return valid() 时返回预测器；否则返回 nullptr
     */
    std::unique_ptr<std::function<std::vector<cv::Point3f>(double)>>
    capturePosePredictor() const;

private:
    Params params_;
    int detect_count_ = 0;                     // 累计识别帧数
    TimePoint last_obs_ts_;                    // 最近一次观测帧时间戳（重置判定用）
    bool valid_ = false;                       // 识别帧数达标且未超时重置
    cv::Vec3d position_ = cv::Vec3d(0, 0, 0);  // 保存的物体世界坐标（米）
    cv::Mat   R_;                              // 保存的物体旋转矩阵（CV_64F）
};

}  // namespace sp_ekf
