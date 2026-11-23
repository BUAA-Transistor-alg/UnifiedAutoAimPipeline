// ArmorRoiPlanner.h — 装甲板自适应 ROI 规划器（config: armor.roi）
//
// 目的：远距离目标的装甲板在整幅画面里只有十几个像素，整幅缩放送进网络后关键点
// 精度不足。规划器按目标距离裁剪出原画面中的一块矩形（ROI），按网络输入尺寸比例
// 缩放到网络输入后，关键点精度不因整幅缩放而损失。
//
// 三种识别视图（View）：
//   全画面（full_frame = true）  —— 当前帧与历史行为一致；
//   ROI（full_frame = false）    —— 原画面内按网络输入尺寸比例截取的一块矩形，
//                                   缩放到网络输入后被识别；检测结果由流水线映射
//                                   回原图坐标。
//
// 五种工作模式（Mode）：
//   OFF        功能关闭：只用全画面；
//   NO_TARGET  无目标：每帧全画面；额外每 scanInterval 帧识别一个分块 ROI
//              （全画面按最小 ROI 无缝覆盖、重叠均匀分布，循环遍历全部分块）；
//   TARGET_ROI 已由 stage5 锁定目标：按目标相机距离截取目标周围固定物理边长的
//              ROI（最小 = 网络输入尺寸），每 monitorInterval 帧同时识别全画面；
//   BBOX_ROI   有物体但尚未锁定目标：截取包围全部物体的矩形放大 k 倍的 ROI，
//              直到下游锁定目标，每 monitorInterval 帧同时识别全画面；
//   FULL_HOLD  ROI 外/边界上识别到物体：至少 fullFrameHoldSec 秒内只用全画面，
//              交由下游判断是否切换目标，之后重新按触发条件决定模式。
//
// 通用约束：所有 ROI 按网络输入尺寸比例截取（宽高比 = inputWidth:inputHeight）、
// 最小尺寸 = 网络输入尺寸（再缩小只是把像素放大，不增加信息且损失视野），并限制
// 在原画面内；任一边长超过原画面时退回全画面。
#ifndef ARMOR_ROI_PLANNER_H
#define ARMOR_ROI_PLANNER_H

#include <chrono>
#include <mutex>
#include <vector>

#include <opencv2/core.hpp>

namespace ArmorRoi {

// 规划器工作模式（含义见文件头注释）
enum class Mode {
    OFF = 0,
    NO_TARGET = 1,
    TARGET_ROI = 2,
    BBOX_ROI = 3,
    FULL_HOLD = 4,
};

// 模式名（调试打印 / 可视化文字用）
const char* modeName(Mode mode);

// 本帧的一个识别视图
struct View {
    bool full_frame = true;   // true = 整幅原画面
    cv::Rect rect;            // full_frame = false 时有效：原图坐标系下的截取矩形
};

// 本帧的识别视图规划
struct Plan {
    Mode mode = Mode::OFF;
    // 至少一个视图；供识别的顺序固定为：ROI 模式下 [ROI, 全画面]，
    // 无目标模式下 [全画面, 分块 ROI]，即“主视图在前、附加视图在后”。
    std::vector<View> views;
    // 是否存在非全画面视图
    bool hasRoi() const;
};

// 规划器只需要物体的几何与类别：流水线把后处理结果转换成该结构后传入
struct Detection {
    cv::Rect2f rect;   // 原图坐标（流水线已把 ROI 内检测结果平移回原图）
    int label = -1;    // 类别 label 0~8
};

// 自适应 ROI 规划器（线程安全：取帧线程调 plan，阶段3/阶段5 线程调另外两个接口）
class Planner {
public:
    // 全部参数由 config armor.roi 与相机内参提供；非法取值在构造时抛出异常
    struct Params {
        bool   enabled = false;
        double targetSpanM = 1.5;         // 目标 ROI 覆盖的物理边长（米，目标距离处）
        int    monitorInterval = 5;       // 使用 ROI 时每隔多少帧同时识别全画面（>= 1）
        double fullFrameHoldSec = 0.5;    // ROI 外/边界识别到物体后全画面保持时长（秒，>= 0）
        int    scanInterval = 3;          // 无目标时每隔多少帧额外识别一个分块 ROI（>= 1）
        int    tileOverlapPx = 64;        // 分块 ROI 之间至少重叠的像素（>= 0，且小于输入短边）
        double bboxExpandFactor = 2.0;    // 包围全部物体的矩形放大倍数（>= 1）
        int    boundaryMarginPx = 2;      // 边界判定内缩像素（>= 0）：检测框完全落在
                                          // ROI 内缩该像素后的区域内才算“内部”
        int    inputWidth = 640;          // 网络输入宽（像素，ROI 最小尺寸与之相同）
        int    inputHeight = 640;         // 网络输入高（像素）
        double fx = 0.0;                  // 相机内参 fx / fy（目标 ROI 像素尺寸换算）
        double fy = 0.0;
    };

    explicit Planner(const Params& params);

    // 取帧时调用：返回本帧的识别视图规划（同时推进帧计数与模式状态）
    Plan plan(const cv::Size& frame_size);

    // 阶段3 调用：本帧各视图的检测结果（原图坐标，顺序与 plan.views 一一对应）。
    // 返回与 view_detections 同形状的保留掩码：1 = 该检测进入下游，0 = 丢弃。
    // 丢弃规则：
    //   · 非全画面视图里没有完全落在视图矩形内的检测（被 ROI 边界截断）→ 丢弃；
    //     若当前为 TARGET_ROI / BBOX_ROI，同时触发 FULL_HOLD；
    //   · 全画面视图中与保留的非全画面检测重复（同 label 且 IoU 高或中心互相包含）
    //     的框 → 丢弃（保留分辨率更高的 ROI 结果）；
    //   · 全画面视图中落在 ROI 外或边界上（未完全落在 ROI 内缩 boundaryMarginPx 的
    //     区域内）的框 → 保留（供下游判断是否切换目标），并触发 FULL_HOLD。
    // 同时更新内部物体集合（供 BBOX_ROI 计算包围矩形）与全画面保持计时。
    std::vector<std::vector<char>> filterDetections(
        const Plan& plan,
        const std::vector<std::vector<Detection>>& view_detections);

    // 阶段5 调用：stage5 本帧所选目标的相机距离（米）与投影图像点（原图像素坐标）。
    // target_valid = false 或 point_valid = false 时清除目标状态（下一帧退回
    // 包围 ROI / 无目标模式）。
    void observeTarget(bool target_valid, double camera_distance_m,
                       const cv::Point2f& image_point, bool point_valid);

    // 清空全部跨帧状态（流水线 clear() 时调用）
    void reset();

    // 当前模式（调试/可视化用；不加锁读取，仅作提示）
    Mode mode() const { return mode_; }

private:
    // 目标 ROI 矩形：按相机距离换算（失败/需要退回全画面时返回 false）
    bool targetRoiRect(const cv::Size& frame_size, cv::Rect& rect) const;
    // 包围全部物体的 ROI 矩形（失败/需要退回全画面时返回 false）
    bool bboxRoiRect(const cv::Size& frame_size, cv::Rect& rect) const;
    // 按当前画面尺寸重建分块 ROI 列表（最小 ROI 无缝覆盖 + 重叠均匀分布）
    void rebuildTiles(const cv::Size& frame_size);
    // 模式切换：重置帧计数
    void enterMode(Mode mode);
    // 计算一个按网络输入比例、限制在画面内的 ROI 矩形；失败返回 false（退回全画面）
    bool makeRoi(const cv::Size& frame_size, const cv::Point2f& center,
                 double needed_w, double needed_h, cv::Rect& rect) const;

    mutable std::mutex mtx_;
    Params params_;

    Mode mode_ = Mode::OFF;
    std::chrono::steady_clock::time_point hold_until_{};

    // ── 阶段5 目标观测（相机距离 + 投影点）──
    bool target_valid_ = false;
    bool target_point_valid_ = false;
    double target_distance_m_ = 0.0;
    cv::Point2f target_point_{0.0f, 0.0f};

    // ── 最近一帧保留下来的检测（原图坐标，供 BBOX_ROI）──
    std::vector<Detection> detections_;

    // ── 帧计数（按模式重置）──
    int roi_frame_index_ = 0;   // TARGET_ROI / BBOX_ROI：全画面监视帧计数
    int scan_frame_index_ = 0;  // NO_TARGET：分块扫描帧计数

    // ── 分块 ROI 缓存（画面尺寸变化时重建）──
    cv::Size tile_frame_size_{0, 0};
    std::vector<cv::Rect> tiles_;
    size_t tile_index_ = 0;
};

} // namespace ArmorRoi

#endif // ARMOR_ROI_PLANNER_H
