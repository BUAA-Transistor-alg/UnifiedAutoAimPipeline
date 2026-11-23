// ArmorRoiPlanner.cpp — 装甲板自适应 ROI 规划器实现（接口契约见 ArmorRoiPlanner.h）
#include "Armor/ArmorRoiPlanner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ArmorRoi {
namespace {

using Clock = std::chrono::steady_clock;

// 检测框完全落在视图矩形内（闭区间比较，浮点坐标）
bool containedIn(const cv::Rect2f& inner, const cv::Rect& outer) {
    const float outer_right = static_cast<float>(outer.x + outer.width);
    const float outer_bottom = static_cast<float>(outer.y + outer.height);
    return inner.x >= static_cast<float>(outer.x) &&
           inner.y >= static_cast<float>(outer.y) &&
           inner.x + inner.width <= outer_right &&
           inner.y + inner.height <= outer_bottom;
}

// 点落在矩形内（闭区间）
bool pointInside(const cv::Rect2f& rect, const cv::Point2f& point) {
    return point.x >= rect.x && point.x <= rect.x + rect.width &&
           point.y >= rect.y && point.y <= rect.y + rect.height;
}

// 两个检测框的几何重合判定（label 是否相同由调用处判断）：
// IoU >= 0.3，或任一矩形的中心落在对方矩形内（闭区间）；
// 并集面积为 0（两框都退化）时只走中心包含判定。
bool duplicateGeometry(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float inter = (x2 > x1 && y2 > y1) ? (x2 - x1) * (y2 - y1) : 0.0f;
    const float uni = a.area() + b.area() - inter;
    if (uni > 0.0f && inter / uni >= 0.3f) {
        return true;
    }
    const cv::Point2f center_a(a.x + a.width * 0.5f, a.y + a.height * 0.5f);
    const cv::Point2f center_b(b.x + b.width * 0.5f, b.y + b.height * 0.5f);
    return pointInside(b, center_a) || pointInside(a, center_b);
}

// UAP_DEBUG_ROI=1 时逐帧打印规划结果
bool debugEnabled() {
    const char* flag = std::getenv("UAP_DEBUG_ROI");
    return flag != nullptr && std::string(flag) == "1";
}

void printPlanDebug(const Plan& plan) {
    std::cout << "[ROI] 模式: " << modeName(plan.mode) << " 视图:";
    for (const View& view : plan.views) {
        if (view.full_frame) {
            std::cout << " 全画面";
        } else {
            std::cout << " ROI(" << view.rect.x << "," << view.rect.y << ","
                      << view.rect.width << "," << view.rect.height << ")";
        }
    }
    std::cout << std::endl;
}

} // namespace

const char* modeName(Mode mode) {
    switch (mode) {
        case Mode::OFF:       return "OFF";
        case Mode::NO_TARGET: return "NO_TARGET";
        case Mode::TARGET_ROI: return "TARGET_ROI";
        case Mode::BBOX_ROI:  return "BBOX_ROI";
        case Mode::FULL_HOLD: return "FULL_HOLD";
    }
    return "UNKNOWN";
}

bool Plan::hasRoi() const {
    for (const View& view : views) {
        if (!view.full_frame) {
            return true;
        }
    }
    return false;
}

Planner::Planner(const Params& params) : params_(params) {
    // 非法参数就地抛出异常，不做默认值兜底
    if (params_.inputWidth < 1) {
        throw std::invalid_argument("ArmorRoi::Planner: inputWidth 必须 >= 1");
    }
    if (params_.inputHeight < 1) {
        throw std::invalid_argument("ArmorRoi::Planner: inputHeight 必须 >= 1");
    }
    if (!(params_.fx > 0.0)) {
        throw std::invalid_argument("ArmorRoi::Planner: fx 必须 > 0");
    }
    if (!(params_.fy > 0.0)) {
        throw std::invalid_argument("ArmorRoi::Planner: fy 必须 > 0");
    }
    if (params_.monitorInterval < 1) {
        throw std::invalid_argument("ArmorRoi::Planner: monitorInterval 必须 >= 1");
    }
    if (params_.scanInterval < 1) {
        throw std::invalid_argument("ArmorRoi::Planner: scanInterval 必须 >= 1");
    }
    if (params_.tileOverlapPx < 0) {
        throw std::invalid_argument("ArmorRoi::Planner: tileOverlapPx 必须 >= 0");
    }
    if (!(params_.bboxExpandFactor >= 1.0)) {
        throw std::invalid_argument("ArmorRoi::Planner: bboxExpandFactor 必须 >= 1.0");
    }
    if (!(params_.fullFrameHoldSec >= 0.0)) {
        throw std::invalid_argument("ArmorRoi::Planner: fullFrameHoldSec 必须 >= 0");
    }
    if (!(params_.targetSpanM > 0.0)) {
        throw std::invalid_argument("ArmorRoi::Planner: targetSpanM 必须 > 0");
    }
    if (params_.boundaryMarginPx < 0) {
        throw std::invalid_argument("ArmorRoi::Planner: boundaryMarginPx 必须 >= 0");
    }
}

Plan Planner::plan(const cv::Size& frame_size) {
    std::lock_guard<std::mutex> lock(mtx_);
    const Clock::time_point now = Clock::now();

    // 当前帧应当处于的模式（优先级：总开关 > 全画面保持 > 目标 ROI > 包围 ROI > 无目标）
    Mode desired = Mode::NO_TARGET;
    if (!params_.enabled) {
        desired = Mode::OFF;
    } else if (now < hold_until_) {
        desired = Mode::FULL_HOLD;
    } else if (target_valid_ && target_point_valid_) {
        desired = Mode::TARGET_ROI;
    } else if (!detections_.empty()) {
        desired = Mode::BBOX_ROI;
    }
    if (desired != mode_) {
        enterMode(desired);
    }

    Plan result;
    result.mode = mode_;
    switch (mode_) {
        case Mode::OFF:
        case Mode::FULL_HOLD:
            result.views.push_back(View{true, cv::Rect()});
            break;
        case Mode::TARGET_ROI: {
            cv::Rect roi;
            if (targetRoiRect(frame_size, roi)) {
                result.views.push_back(View{false, roi});
                // 使用 ROI 时每隔 monitorInterval 帧同时识别一次全画面
                if (roi_frame_index_ % params_.monitorInterval == 0) {
                    result.views.push_back(View{true, cv::Rect()});
                }
                ++roi_frame_index_;
            } else {
                // 目标距离过近导致 ROI 超出画面：退回全画面，不推进监视帧计数
                result.views.push_back(View{true, cv::Rect()});
            }
            break;
        }
        case Mode::BBOX_ROI: {
            cv::Rect roi;
            if (bboxRoiRect(frame_size, roi)) {
                result.views.push_back(View{false, roi});
                if (roi_frame_index_ % params_.monitorInterval == 0) {
                    result.views.push_back(View{true, cv::Rect()});
                }
                ++roi_frame_index_;
            } else {
                result.views.push_back(View{true, cv::Rect()});
            }
            break;
        }
        case Mode::NO_TARGET:
            // 画面尺寸变化时重建分块缓存（进入无目标模式后的首帧也由此完成首次构建）
            if (tile_frame_size_ != frame_size) {
                rebuildTiles(frame_size);
            }
            result.views.push_back(View{true, cv::Rect()});
            // 每 scanInterval 帧额外识别一个分块 ROI，循环遍历全部分块
            if (!tiles_.empty() && scan_frame_index_ % params_.scanInterval == 0) {
                const cv::Rect& tile = tiles_[tile_index_];
                // 分块恰好覆盖整幅画面时与全画面视图完全相同，跳过该分块（计数语义不变）
                const bool full_frame_tile = tile.x == 0 && tile.y == 0 &&
                                             tile.width == frame_size.width &&
                                             tile.height == frame_size.height;
                if (!full_frame_tile) {
                    result.views.push_back(View{false, tile});
                }
                tile_index_ = (tile_index_ + 1) % tiles_.size();
            }
            ++scan_frame_index_;
            break;
    }

    if (debugEnabled()) {
        printPlanDebug(result);
    }
    return result;
}

std::vector<std::vector<char>> Planner::filterDetections(
    const Plan& plan,
    const std::vector<std::vector<Detection>>& view_detections) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (view_detections.size() != plan.views.size()) {
        throw std::invalid_argument(
            "ArmorRoi::Planner::filterDetections: 视图检测结果数量(" +
            std::to_string(view_detections.size()) + ")与计划视图数量(" +
            std::to_string(plan.views.size()) + ")不一致");
    }

    const Clock::time_point now = Clock::now();
    const Clock::duration hold_duration = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(params_.fullFrameHoldSec));
    auto trigger_full_hold = [&]() {
        hold_until_ = std::max(hold_until_, now + hold_duration);
    };

    std::vector<std::vector<char>> keep(plan.views.size());
    for (std::size_t v = 0; v < plan.views.size(); ++v) {
        keep[v].assign(view_detections[v].size(), 1);
    }

    // 计划保证至多一个非全画面视图；全画面监视视图的边界判定以它为准
    const cv::Rect* roi_view = nullptr;
    for (const View& view : plan.views) {
        if (!view.full_frame) {
            roi_view = &view.rect;
            break;
        }
    }

    // 第 1 遍：非全画面视图（ROI / 分块）的检测必须完全落在视图矩形内，
    // 被边界截断的检测直接丢弃（不进入下游）
    std::vector<Detection> priority_kept;
    for (std::size_t v = 0; v < plan.views.size(); ++v) {
        if (plan.views[v].full_frame) {
            continue;
        }
        for (std::size_t i = 0; i < view_detections[v].size(); ++i) {
            const Detection& detection = view_detections[v][i];
            if (!containedIn(detection.rect, plan.views[v].rect)) {
                keep[v][i] = 0;
                if (plan.mode == Mode::TARGET_ROI || plan.mode == Mode::BBOX_ROI) {
                    trigger_full_hold();
                }
            } else {
                priority_kept.push_back(detection);
            }
        }
    }

    // 第 2 遍：全画面视图去重（保留分辨率更高的 ROI 结果），并判定监视视图外/边界上的物体
    const bool monitor_view =
        (plan.mode == Mode::TARGET_ROI || plan.mode == Mode::BBOX_ROI) && roi_view != nullptr;
    for (std::size_t v = 0; v < plan.views.size(); ++v) {
        if (!plan.views[v].full_frame) {
            continue;
        }
        for (std::size_t i = 0; i < view_detections[v].size(); ++i) {
            const Detection& detection = view_detections[v][i];
            bool duplicate = false;
            for (const Detection& kept : priority_kept) {
                if (kept.label == detection.label &&
                    duplicateGeometry(detection.rect, kept.rect)) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                keep[v][i] = 0;
                continue;
            }
            if (monitor_view) {
                const int margin = params_.boundaryMarginPx;
                const int deflated_width = roi_view->width - 2 * margin;
                const int deflated_height = roi_view->height - 2 * margin;
                bool inside = false;
                if (deflated_width >= 1 && deflated_height >= 1) {
                    const cv::Rect deflated(roi_view->x + margin, roi_view->y + margin,
                                            deflated_width, deflated_height);
                    inside = containedIn(detection.rect, deflated);
                }
                if (!inside) {
                    // ROI 外/边界上的物体保留给下游判断是否切换目标，并退回全画面至少 s 秒
                    trigger_full_hold();
                }
            }
        }
    }

    // 覆盖上一帧的物体集合：按视图顺序展平所有保留下来的检测（原图坐标）
    detections_.clear();
    for (std::size_t v = 0; v < plan.views.size(); ++v) {
        for (std::size_t i = 0; i < view_detections[v].size(); ++i) {
            if (keep[v][i]) {
                detections_.push_back(view_detections[v][i]);
            }
        }
    }
    return keep;
}

void Planner::observeTarget(bool target_valid, double camera_distance_m,
                            const cv::Point2f& image_point, bool point_valid) {
    std::lock_guard<std::mutex> lock(mtx_);
    const bool valid = target_valid && point_valid && camera_distance_m > 0.0 &&
                       std::isfinite(image_point.x) && std::isfinite(image_point.y);
    if (valid) {
        target_valid_ = true;
        target_point_valid_ = true;
        target_distance_m_ = camera_distance_m;
        target_point_ = image_point;
    } else {
        // 距离/投影点无效：清除目标状态，下一帧退回包围 ROI / 无目标模式
        target_valid_ = false;
        target_point_valid_ = false;
    }
}

void Planner::reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    mode_ = Mode::OFF;
    hold_until_ = Clock::time_point{};
    target_valid_ = false;
    target_point_valid_ = false;
    target_distance_m_ = 0.0;
    target_point_ = cv::Point2f(0.0f, 0.0f);
    detections_.clear();
    roi_frame_index_ = 0;
    scan_frame_index_ = 0;
    tile_frame_size_ = cv::Size(0, 0);
    tiles_.clear();
    tile_index_ = 0;
}

bool Planner::targetRoiRect(const cv::Size& frame_size, cv::Rect& rect) const {
    const double distance = std::max(target_distance_m_, 1e-6);
    const double needed_width = params_.fx * params_.targetSpanM / distance;
    const double needed_height = params_.fy * params_.targetSpanM / distance;
    return makeRoi(frame_size, target_point_, needed_width, needed_height, rect);
}

bool Planner::bboxRoiRect(const cv::Size& frame_size, cv::Rect& rect) const {
    if (detections_.empty()) {
        return false;
    }
    float min_x = detections_.front().rect.x;
    float min_y = detections_.front().rect.y;
    float max_x = detections_.front().rect.x + detections_.front().rect.width;
    float max_y = detections_.front().rect.y + detections_.front().rect.height;
    for (const Detection& detection : detections_) {
        min_x = std::min(min_x, detection.rect.x);
        min_y = std::min(min_y, detection.rect.y);
        max_x = std::max(max_x, detection.rect.x + detection.rect.width);
        max_y = std::max(max_y, detection.rect.y + detection.rect.height);
    }
    const double needed_width = static_cast<double>(max_x - min_x) * params_.bboxExpandFactor;
    const double needed_height = static_cast<double>(max_y - min_y) * params_.bboxExpandFactor;
    const cv::Point2f center((min_x + max_x) * 0.5f, (min_y + max_y) * 0.5f);
    return makeRoi(frame_size, center, needed_width, needed_height, rect);
}

bool Planner::makeRoi(const cv::Size& frame_size, const cv::Point2f& center,
                      double needed_width, double needed_height, cv::Rect& rect) const {
    // q >= 1：ROI 不小于网络输入尺寸（再缩小只是把像素放大，不增加信息且损失视野）
    const double scale = std::max(
        1.0, std::max(needed_width / params_.inputWidth, needed_height / params_.inputHeight));
    // 先按 double 取整再判断越界，避免超大需求值转 int 时溢出
    const double rounded_width = static_cast<double>(std::lround(params_.inputWidth * scale));
    const double rounded_height = static_cast<double>(std::lround(params_.inputHeight * scale));
    if (!(rounded_width >= 1.0) || !(rounded_height >= 1.0) ||
        rounded_width > frame_size.width || rounded_height > frame_size.height) {
        // 任一边长超过画面：退回全画面
        return false;
    }
    const int width = static_cast<int>(rounded_width);
    const int height = static_cast<int>(rounded_height);
    // ROI 恰好覆盖整幅画面时与全画面视图的像素完全相同（没有任何放大收益），
    // 直接退回全画面，避免监视帧重复识别同一张图像
    if (width == frame_size.width && height == frame_size.height) {
        return false;
    }
    int x = static_cast<int>(std::lround(center.x - width / 2.0));
    int y = static_cast<int>(std::lround(center.y - height / 2.0));
    x = std::min(std::max(x, 0), frame_size.width - width);
    y = std::min(std::max(y, 0), frame_size.height - height);
    rect = cv::Rect(x, y, width, height);
    return true;
}

void Planner::rebuildTiles(const cv::Size& frame_size) {
    tile_frame_size_ = frame_size;
    tile_index_ = 0;
    tiles_.clear();

    const int roi_width = params_.inputWidth;
    const int roi_height = params_.inputHeight;
    const int frame_width = frame_size.width;
    const int frame_height = frame_size.height;
    if (roi_width > frame_width || roi_height > frame_height) {
        // 最小 ROI 已经大于画面：无法分块
        return;
    }

    // 重叠量上限为分块边长 - 1：避免步长 <= 0 除零（config 要求 tileOverlapPx 小于输入短边）
    const int overlap_x = std::min(params_.tileOverlapPx, roi_width - 1);
    const int overlap_y = std::min(params_.tileOverlapPx, roi_height - 1);
    const int step_x = roi_width - overlap_x;
    const int step_y = roi_height - overlap_y;
    const int columns = (frame_width <= roi_width)
                            ? 1
                            : (frame_width - roi_width + step_x - 1) / step_x + 1;
    const int rows = (frame_height <= roi_height)
                         ? 1
                         : (frame_height - roi_height + step_y - 1) / step_y + 1;

    // 一维分块起点：首尾对齐画面两端，其余等距分布（相邻重叠量尽量均匀）
    auto tile_positions = [](int frame_length, int roi_length, int count) {
        std::vector<int> positions(static_cast<std::size_t>(count));
        if (count == 1) {
            positions[0] = 0;
            return positions;
        }
        for (int i = 0; i < count; ++i) {
            positions[static_cast<std::size_t>(i)] = static_cast<int>(
                std::lround(static_cast<double>(i) * (frame_length - roi_length) / (count - 1)));
        }
        positions.back() = frame_length - roi_length;
        return positions;
    };
    const std::vector<int> positions_x = tile_positions(frame_width, roi_width, columns);
    const std::vector<int> positions_y = tile_positions(frame_height, roi_height, rows);

    tiles_.reserve(positions_x.size() * positions_y.size());
    // 行优先（y 外层、x 内层）：每个分块尺寸 = 最小 ROI，完全在画面内
    for (const int y : positions_y) {
        for (const int x : positions_x) {
            tiles_.emplace_back(x, y, roi_width, roi_height);
        }
    }
}

void Planner::enterMode(Mode mode) {
    std::cout << "模式切换: " << modeName(mode_) << " -> " << modeName(mode) << std::endl;
    mode_ = mode;
    roi_frame_index_ = 0;
    scan_frame_index_ = 0;
}

} // namespace ArmorRoi
