// roi_planner_test.cpp — ArmorRoiPlanner 独立单元测试（不依赖 gtest）
//
// 覆盖：参数校验、OFF / NO_TARGET / TARGET_ROI / BBOX_ROI / FULL_HOLD 五种模式、
// 全画面监视间隔、最小 ROI 尺寸与网络输入比例、画面内限制与超画面退回、
// 分块无缝覆盖与最小重叠、检测去重、ROI 边界判定、reset 与目标状态清除。
#include "Armor/ArmorRoiPlanner.h"

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

std::string toText(ArmorRoi::Mode mode) {
    return ArmorRoi::modeName(mode);
}

template <typename T>
std::string toText(const T& value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

#define CHECK(condition)                                                                    \
    do {                                                                                    \
        ++g_checks;                                                                         \
        if (!(condition)) {                                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK 失败: " #condition          \
                      << std::endl;                                                         \
            ++g_failures;                                                                   \
        }                                                                                   \
    } while (false)

#define CHECK_EQ(actual, expected)                                                          \
    do {                                                                                    \
        ++g_checks;                                                                         \
        const auto& check_eq_actual = (actual);                                             \
        const auto& check_eq_expected = (expected);                                         \
        if (!(check_eq_actual == check_eq_expected)) {                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK_EQ 失败: " #actual          \
                      << " == " #expected << "（实际 " << toText(check_eq_actual)           \
                      << "，期望 " << toText(check_eq_expected) << "）" << std::endl;        \
            ++g_failures;                                                                   \
        }                                                                                   \
    } while (false)

void expectInvalidArgument(const char* file, int line, const char* expression,
                           const std::function<void()>& action) {
    ++g_checks;
    try {
        action();
    } catch (const std::invalid_argument&) {
        return;
    } catch (...) {
        std::cerr << file << ":" << line
                  << ": 期望抛出 std::invalid_argument，实际抛出其它类型异常: " << expression
                  << std::endl;
        ++g_failures;
        return;
    }
    std::cerr << file << ":" << line
              << ": 期望抛出 std::invalid_argument，实际未抛出: " << expression << std::endl;
    ++g_failures;
}

#define CHECK_THROWS_INVALID(expression)                                                     \
    expectInvalidArgument(__FILE__, __LINE__, #expression, [&]() { expression; })

ArmorRoi::Planner::Params baseParams() {
    ArmorRoi::Planner::Params params;
    params.enabled = true;
    params.targetSpanM = 1.5;
    params.monitorInterval = 5;
    params.fullFrameHoldSec = 0.5;
    params.scanInterval = 3;
    params.tileOverlapPx = 64;
    params.bboxExpandFactor = 2.0;
    params.boundaryMarginPx = 2;
    params.inputWidth = 640;
    params.inputHeight = 640;
    params.fx = 1312.0;
    params.fy = 1312.0;
    return params;
}

bool rectEqual(const cv::Rect& lhs, const cv::Rect& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.width == rhs.width &&
           lhs.height == rhs.height;
}

bool rectInside(const cv::Rect& rect, const cv::Size& frame) {
    return rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= frame.width &&
           rect.y + rect.height <= frame.height;
}

// 分块并集是否覆盖整幅画面（逐像素采样）
bool coveredBy(const std::vector<cv::Rect>& tiles, const cv::Size& frame) {
    for (int y = 0; y < frame.height; ++y) {
        for (int x = 0; x < frame.width; ++x) {
            bool covered = false;
            for (const cv::Rect& tile : tiles) {
                if (x >= tile.x && x < tile.x + tile.width && y >= tile.y &&
                    y < tile.y + tile.height) {
                    covered = true;
                    break;
                }
            }
            if (!covered) {
                return false;
            }
        }
    }
    return true;
}

// 分块网格几何断言：尺寸 = 最小 ROI、完全在画面内、行优先排列、
// 行/列内相邻重叠 >= overlap 且尽量均匀、并集覆盖整幅画面
void checkTileGrid(const std::vector<cv::Rect>& tiles, int columns, int rows, int tile_width,
                   int tile_height, int overlap, const cv::Size& frame) {
    CHECK_EQ(static_cast<int>(tiles.size()), columns * rows);
    if (static_cast<int>(tiles.size()) != columns * rows) {
        return;
    }
    for (const cv::Rect& tile : tiles) {
        CHECK_EQ(tile.width, tile_width);
        CHECK_EQ(tile.height, tile_height);
        CHECK(rectInside(tile, frame));
    }
    // 行优先（y 外层、x 内层）：同一行 y 相同、x 递增
    for (int row = 0; row < rows; ++row) {
        for (int column = 1; column < columns; ++column) {
            const cv::Rect& prev = tiles[static_cast<std::size_t>(row * columns + column - 1)];
            const cv::Rect& current = tiles[static_cast<std::size_t>(row * columns + column)];
            CHECK_EQ(current.y, prev.y);
            CHECK(current.x > prev.x);
            CHECK(prev.x + prev.width - current.x >= overlap);
        }
        if (columns > 1) {
            int min_overlap = columns * tile_width;
            int max_overlap = 0;
            for (int column = 1; column < columns; ++column) {
                const cv::Rect& prev =
                    tiles[static_cast<std::size_t>(row * columns + column - 1)];
                const cv::Rect& current = tiles[static_cast<std::size_t>(row * columns + column)];
                const int current_overlap = prev.x + prev.width - current.x;
                min_overlap = std::min(min_overlap, current_overlap);
                max_overlap = std::max(max_overlap, current_overlap);
            }
            CHECK(max_overlap - min_overlap <= 1);
        }
    }
    // 列内相邻重叠 >= overlap，且上下两行的 x 起点一致
    for (int row = 1; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const cv::Rect& prev =
                tiles[static_cast<std::size_t>((row - 1) * columns + column)];
            const cv::Rect& current = tiles[static_cast<std::size_t>(row * columns + column)];
            CHECK_EQ(current.x, prev.x);
            CHECK(current.y > prev.y);
            CHECK(prev.y + prev.height - current.y >= overlap);
        }
    }
    // 首尾对齐画面两端 → 并集覆盖整幅画面
    CHECK_EQ(tiles.front().x, 0);
    CHECK_EQ(tiles.front().y, 0);
    CHECK_EQ(tiles.back().x + tiles.back().width, frame.width);
    CHECK_EQ(tiles.back().y + tiles.back().height, frame.height);
    CHECK(coveredBy(tiles, frame));
}

void testParamsValidation() {
    const ArmorRoi::Planner::Params base = baseParams();
    ArmorRoi::Planner valid(base);
    CHECK(valid.mode() == ArmorRoi::Mode::OFF);
    CHECK_EQ(std::string(ArmorRoi::modeName(ArmorRoi::Mode::OFF)), std::string("OFF"));
    CHECK_EQ(std::string(ArmorRoi::modeName(ArmorRoi::Mode::NO_TARGET)),
             std::string("NO_TARGET"));
    CHECK_EQ(std::string(ArmorRoi::modeName(ArmorRoi::Mode::TARGET_ROI)),
             std::string("TARGET_ROI"));
    CHECK_EQ(std::string(ArmorRoi::modeName(ArmorRoi::Mode::BBOX_ROI)),
             std::string("BBOX_ROI"));
    CHECK_EQ(std::string(ArmorRoi::modeName(ArmorRoi::Mode::FULL_HOLD)),
             std::string("FULL_HOLD"));

    { auto p = base; p.inputWidth = 0; CHECK_THROWS_INVALID((void)ArmorRoi::Planner(p)); }
    { auto p = base; p.inputHeight = 0; CHECK_THROWS_INVALID((void)ArmorRoi::Planner(p)); }
    { auto p = base; p.monitorInterval = 0; CHECK_THROWS_INVALID((void)ArmorRoi::Planner(p)); }
    { auto p = base; p.scanInterval = 0; CHECK_THROWS_INVALID((void)ArmorRoi::Planner(p)); }
    { auto p = base; p.tileOverlapPx = -1; CHECK_THROWS_INVALID((void)ArmorRoi::Planner(p)); }
    { auto p = base; p.bboxExpandFactor = 0.5; CHECK_THROWS_INVALID((void)ArmorRoi::Planner(p)); }
    { auto p = base; p.fullFrameHoldSec = -0.1; CHECK_THROWS_INVALID((void)ArmorRoi::Planner(p)); }
    { auto p = base; p.targetSpanM = 0.0; CHECK_THROWS_INVALID((void)ArmorRoi::Planner(p)); }
    { auto p = base; p.boundaryMarginPx = -1; CHECK_THROWS_INVALID((void)ArmorRoi::Planner(p)); }
    {
        auto p = base;
        p.fx = 0.0;
        bool threw = false;
        try {
            ArmorRoi::Planner planner(p);
            (void)planner;
        } catch (const std::invalid_argument& error) {
            threw = true;
            CHECK(std::string(error.what()).find("fx") != std::string::npos);
        }
        CHECK(threw);
    }
    { auto p = base; p.fy = -1.0; CHECK_THROWS_INVALID((void)ArmorRoi::Planner(p)); }
}

void testOffMode() {
    ArmorRoi::Planner::Params params = baseParams();
    params.enabled = false;
    ArmorRoi::Planner planner(params);
    for (int frame_index = 0; frame_index < 6; ++frame_index) {
        const ArmorRoi::Plan plan = planner.plan(cv::Size(1920, 1080));
        CHECK_EQ(plan.mode, ArmorRoi::Mode::OFF);
        CHECK_EQ(plan.views.size(), 1u);
        CHECK(plan.views[0].full_frame);
        CHECK(!plan.hasRoi());
    }
}

void testNoTargetTiles() {
    // 1920x1080，最小 ROI 640x640，重叠 >= 64：4 列 x 2 行，每 3 帧扫描一个分块
    {
        ArmorRoi::Planner planner(baseParams());
        const cv::Size frame(1920, 1080);
        std::vector<cv::Rect> scanned;
        for (int frame_index = 0; frame_index < 30; ++frame_index) {
            const ArmorRoi::Plan plan = planner.plan(frame);
            CHECK_EQ(plan.mode, ArmorRoi::Mode::NO_TARGET);
            CHECK(!plan.views.empty());
            CHECK(plan.views[0].full_frame);
            if (frame_index % 3 != 0) {
                CHECK_EQ(plan.views.size(), 1u);
                continue;
            }
            CHECK_EQ(plan.views.size(), 2u);
            CHECK(!plan.views[1].full_frame);
            CHECK(plan.hasRoi());
            scanned.push_back(plan.views[1].rect);
        }
        CHECK_EQ(scanned.size(), 10u);
        if (scanned.size() >= 9) {
            std::vector<cv::Rect> one_round(scanned.begin(), scanned.begin() + 8);
            for (std::size_t i = 0; i < one_round.size(); ++i) {
                for (std::size_t j = i + 1; j < one_round.size(); ++j) {
                    CHECK(!rectEqual(one_round[i], one_round[j]));
                }
            }
            // 分块循环：第 9、10 个扫描帧回到第 1、2 个分块
            CHECK(rectEqual(scanned[8], scanned[0]));
            CHECK(rectEqual(scanned[9], scanned[1]));
            checkTileGrid(one_round, 4, 2, 640, 640, 64, frame);
        }
    }
    // 1280x1024：3 列 x 2 行，共 6 个分块
    {
        ArmorRoi::Planner planner(baseParams());
        const cv::Size frame(1280, 1024);
        std::vector<cv::Rect> scanned;
        for (int frame_index = 0; frame_index < 18; ++frame_index) {
            const ArmorRoi::Plan plan = planner.plan(frame);
            CHECK_EQ(plan.mode, ArmorRoi::Mode::NO_TARGET);
            CHECK(!plan.views.empty());
            CHECK(plan.views[0].full_frame);
            if (frame_index % 3 != 0) {
                CHECK_EQ(plan.views.size(), 1u);
                continue;
            }
            CHECK_EQ(plan.views.size(), 2u);
            CHECK(!plan.views[1].full_frame);
            scanned.push_back(plan.views[1].rect);
        }
        CHECK_EQ(scanned.size(), 6u);
        checkTileGrid(scanned, 3, 2, 640, 640, 64, frame);
    }
    // 画面小于最小 ROI：无法分块，只有全画面视图
    {
        ArmorRoi::Planner planner(baseParams());
        const ArmorRoi::Plan plan = planner.plan(cv::Size(320, 240));
        CHECK_EQ(plan.mode, ArmorRoi::Mode::NO_TARGET);
        CHECK_EQ(plan.views.size(), 1u);
        CHECK(plan.views[0].full_frame);
    }
}

void testTargetRoi() {
    ArmorRoi::Planner planner(baseParams());
    const cv::Size frame(1920, 1080);

    // d=10m：需求 196.8 像素小于最小 ROI → 640x640，中心 (960,540)
    planner.observeTarget(true, 10.0, cv::Point2f(960.0f, 540.0f), true);
    ArmorRoi::Plan plan = planner.plan(frame);
    CHECK_EQ(plan.mode, ArmorRoi::Mode::TARGET_ROI);
    CHECK_EQ(plan.views.size(), 2u);
    CHECK(!plan.views[0].full_frame);
    CHECK(plan.views[1].full_frame);
    CHECK(plan.hasRoi());
    CHECK(rectEqual(plan.views[0].rect, cv::Rect(640, 220, 640, 640)));

    // 接下来 4 帧只有 ROI 视图（monitorInterval = 5）
    for (int frame_index = 1; frame_index <= 4; ++frame_index) {
        plan = planner.plan(frame);
        CHECK_EQ(plan.mode, ArmorRoi::Mode::TARGET_ROI);
        CHECK_EQ(plan.views.size(), 1u);
        CHECK(!plan.views[0].full_frame);
        CHECK(rectEqual(plan.views[0].rect, cv::Rect(640, 220, 640, 640)));
    }
    // 第 5 帧再次同时识别全画面
    plan = planner.plan(frame);
    CHECK_EQ(plan.views.size(), 2u);
    CHECK(!plan.views[0].full_frame);
    CHECK(plan.views[1].full_frame);

    // 近目标 d=1m：需求 1968 像素超过画面宽 1920 → 退回全画面
    planner.observeTarget(true, 1.0, cv::Point2f(960.0f, 540.0f), true);
    plan = planner.plan(frame);
    CHECK_EQ(plan.mode, ArmorRoi::Mode::TARGET_ROI);
    CHECK_EQ(plan.views.size(), 1u);
    CHECK(plan.views[0].full_frame);
    CHECK(!plan.hasRoi());

    // 远/近之间 d=2.5m：需求约 787.2 像素 → ROI 787x787，完全在画面内
    planner.observeTarget(true, 2.5, cv::Point2f(960.0f, 540.0f), true);
    plan = planner.plan(frame);
    CHECK_EQ(plan.views.size(), 1u);
    CHECK(!plan.views[0].full_frame);
    CHECK_EQ(plan.views[0].rect.width, 787);
    CHECK_EQ(plan.views[0].rect.height, 787);
    CHECK(rectInside(plan.views[0].rect, frame));

    // 目标点位于画面左上角：ROI 仍完全在画面内（起点钳到 0）
    planner.observeTarget(true, 10.0, cv::Point2f(0.0f, 0.0f), true);
    plan = planner.plan(frame);
    CHECK_EQ(plan.views.size(), 1u);
    CHECK(!plan.views[0].full_frame);
    CHECK(rectEqual(plan.views[0].rect, cv::Rect(0, 0, 640, 640)));
}

void testRoiAspect() {
    // 网络输入 640x480：ROI 比例必须保持 640:480
    ArmorRoi::Planner::Params params = baseParams();
    params.inputWidth = 640;
    params.inputHeight = 480;
    ArmorRoi::Planner planner(params);
    planner.observeTarget(true, 2.0, cv::Point2f(960.0f, 540.0f), true);
    const ArmorRoi::Plan plan = planner.plan(cv::Size(1920, 1080));
    CHECK_EQ(plan.mode, ArmorRoi::Mode::TARGET_ROI);
    CHECK_EQ(plan.views.size(), 2u);
    CHECK(!plan.views[0].full_frame);
    // 需求 984x984 → q = 2.05 → 1312x984，与 640:480 比例一致
    CHECK_EQ(plan.views[0].rect.width, 1312);
    CHECK_EQ(plan.views[0].rect.height, 984);
    CHECK(rectInside(plan.views[0].rect, cv::Size(1920, 1080)));
}

void testBboxRoi() {
    ArmorRoi::Planner planner(baseParams());
    const cv::Size frame(1920, 1080);

    // 无目标：先得到 NO_TARGET 规划（全画面 + 分块）
    ArmorRoi::Plan plan = planner.plan(frame);
    CHECK_EQ(plan.mode, ArmorRoi::Mode::NO_TARGET);
    CHECK_EQ(plan.views.size(), 2u);

    std::vector<std::vector<ArmorRoi::Detection>> detections(plan.views.size());
    detections[0].push_back(
        ArmorRoi::Detection{cv::Rect2f(1000.0f, 600.0f, 30.0f, 20.0f), 1});
    const std::vector<std::vector<char>> keep = planner.filterDetections(plan, detections);
    CHECK_EQ(keep.size(), plan.views.size());
    CHECK_EQ(static_cast<int>(keep[0].size()), 1);
    CHECK_EQ(static_cast<int>(keep[0][0]), 1);

    // 有物体未锁定：BBOX_ROI，包围盒 30x20 放大 2 倍仍小于最小 ROI
    plan = planner.plan(frame);
    CHECK_EQ(plan.mode, ArmorRoi::Mode::BBOX_ROI);
    CHECK_EQ(plan.views.size(), 2u);
    CHECK(!plan.views[0].full_frame);
    CHECK(plan.views[1].full_frame);
    CHECK(rectEqual(plan.views[0].rect, cv::Rect(695, 290, 640, 640)));

    // 下游锁定目标后切换 TARGET_ROI（目标点即包围盒中心）
    planner.observeTarget(true, 10.0, cv::Point2f(1015.0f, 610.0f), true);
    plan = planner.plan(frame);
    CHECK_EQ(plan.mode, ArmorRoi::Mode::TARGET_ROI);
    CHECK_EQ(plan.views.size(), 2u);
    CHECK(rectEqual(plan.views[0].rect, cv::Rect(695, 290, 640, 640)));
}

void testFullHold() {
    const cv::Size frame(1920, 1080);
    // 全画面视图中 ROI 外的检测：保留并触发全画面保持，s 秒后恢复 TARGET_ROI
    {
        ArmorRoi::Planner::Params params = baseParams();
        params.fullFrameHoldSec = 0.05;
        ArmorRoi::Planner planner(params);
        planner.observeTarget(true, 10.0, cv::Point2f(960.0f, 540.0f), true);
        ArmorRoi::Plan plan = planner.plan(frame);
        CHECK_EQ(plan.mode, ArmorRoi::Mode::TARGET_ROI);
        CHECK_EQ(plan.views.size(), 2u);
        CHECK(rectEqual(plan.views[0].rect, cv::Rect(640, 220, 640, 640)));

        std::vector<std::vector<ArmorRoi::Detection>> detections(2);
        detections[1].push_back(
            ArmorRoi::Detection{cv::Rect2f(100.0f, 900.0f, 40.0f, 30.0f), 3});
        const std::vector<std::vector<char>> keep = planner.filterDetections(plan, detections);
        CHECK_EQ(static_cast<int>(keep[1].size()), 1);
        CHECK_EQ(static_cast<int>(keep[1][0]), 1);

        plan = planner.plan(frame);
        CHECK_EQ(plan.mode, ArmorRoi::Mode::FULL_HOLD);
        CHECK_EQ(plan.views.size(), 1u);
        CHECK(plan.views[0].full_frame);

        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        plan = planner.plan(frame);
        CHECK_EQ(plan.mode, ArmorRoi::Mode::TARGET_ROI);
        CHECK_EQ(plan.views.size(), 2u);
        CHECK(!plan.views[0].full_frame);
    }
    // ROI 视图中跨越 ROI 边界的检测：丢弃并触发全画面保持
    {
        ArmorRoi::Planner::Params params = baseParams();
        params.fullFrameHoldSec = 0.05;
        ArmorRoi::Planner planner(params);
        planner.observeTarget(true, 10.0, cv::Point2f(960.0f, 540.0f), true);
        const ArmorRoi::Plan plan = planner.plan(frame);
        CHECK_EQ(plan.views.size(), 2u);
        CHECK(rectEqual(plan.views[0].rect, cv::Rect(640, 220, 640, 640)));

        std::vector<std::vector<ArmorRoi::Detection>> detections(2);
        detections[0].push_back(
            ArmorRoi::Detection{cv::Rect2f(1200.0f, 400.0f, 100.0f, 60.0f), 1});
        const std::vector<std::vector<char>> keep = planner.filterDetections(plan, detections);
        CHECK_EQ(static_cast<int>(keep[0].size()), 1);
        CHECK_EQ(static_cast<int>(keep[0][0]), 0);

        const ArmorRoi::Plan next = planner.plan(frame);
        CHECK_EQ(next.mode, ArmorRoi::Mode::FULL_HOLD);
    }
}

void testDeduplication() {
    ArmorRoi::Planner planner(baseParams());
    const cv::Size frame(1920, 1080);
    planner.observeTarget(true, 10.0, cv::Point2f(960.0f, 540.0f), true);
    const ArmorRoi::Plan plan = planner.plan(frame);
    CHECK_EQ(plan.views.size(), 2u);
    CHECK(rectEqual(plan.views[0].rect, cv::Rect(640, 220, 640, 640)));

    std::vector<std::vector<ArmorRoi::Detection>> detections(2);
    detections[0].push_back(
        ArmorRoi::Detection{cv::Rect2f(700.0f, 300.0f, 100.0f, 80.0f), 1});
    // 与 ROI 结果高度重合且 label 相同：全画面结果丢弃
    detections[1].push_back(
        ArmorRoi::Detection{cv::Rect2f(705.0f, 305.0f, 95.0f, 75.0f), 1});
    // label 不同的同位置框不算重复
    detections[1].push_back(
        ArmorRoi::Detection{cv::Rect2f(705.0f, 305.0f, 95.0f, 75.0f), 2});
    // 不重复且完全在 ROI 内部：保留
    detections[1].push_back(
        ArmorRoi::Detection{cv::Rect2f(800.0f, 500.0f, 40.0f, 30.0f), 2});

    const std::vector<std::vector<char>> keep = planner.filterDetections(plan, detections);
    CHECK_EQ(static_cast<int>(keep[0].size()), 1);
    CHECK_EQ(static_cast<int>(keep[0][0]), 1);
    CHECK_EQ(static_cast<int>(keep[1].size()), 3);
    CHECK_EQ(static_cast<int>(keep[1][0]), 0);
    CHECK_EQ(static_cast<int>(keep[1][1]), 1);
    CHECK_EQ(static_cast<int>(keep[1][2]), 1);

    // 未触发全画面保持：下一帧仍为 TARGET_ROI
    const ArmorRoi::Plan next = planner.plan(frame);
    CHECK_EQ(next.mode, ArmorRoi::Mode::TARGET_ROI);
}

void testBoundaryMargin() {
    const cv::Size frame(1920, 1080);
    // 完全在 ROI 内但距边界小于 margin：触发全画面保持
    {
        ArmorRoi::Planner::Params params = baseParams();
        params.fullFrameHoldSec = 0.05;
        params.boundaryMarginPx = 10;
        ArmorRoi::Planner planner(params);
        planner.observeTarget(true, 10.0, cv::Point2f(960.0f, 540.0f), true);
        const ArmorRoi::Plan plan = planner.plan(frame);
        CHECK_EQ(plan.views.size(), 2u);
        CHECK(rectEqual(plan.views[0].rect, cv::Rect(640, 220, 640, 640)));

        std::vector<std::vector<ArmorRoi::Detection>> detections(2);
        detections[1].push_back(
            ArmorRoi::Detection{cv::Rect2f(645.0f, 400.0f, 40.0f, 30.0f), 1});
        const std::vector<std::vector<char>> keep = planner.filterDetections(plan, detections);
        CHECK_EQ(static_cast<int>(keep[1][0]), 1);

        const ArmorRoi::Plan next = planner.plan(frame);
        CHECK_EQ(next.mode, ArmorRoi::Mode::FULL_HOLD);
    }
    // 完全在内部（距边界超过 margin）：不触发
    {
        ArmorRoi::Planner::Params params = baseParams();
        params.boundaryMarginPx = 10;
        ArmorRoi::Planner planner(params);
        planner.observeTarget(true, 10.0, cv::Point2f(960.0f, 540.0f), true);
        const ArmorRoi::Plan plan = planner.plan(frame);
        CHECK_EQ(plan.views.size(), 2u);

        std::vector<std::vector<ArmorRoi::Detection>> detections(2);
        detections[1].push_back(
            ArmorRoi::Detection{cv::Rect2f(700.0f, 400.0f, 40.0f, 30.0f), 1});
        const std::vector<std::vector<char>> keep = planner.filterDetections(plan, detections);
        CHECK_EQ(static_cast<int>(keep[1][0]), 1);

        const ArmorRoi::Plan next = planner.plan(frame);
        CHECK_EQ(next.mode, ArmorRoi::Mode::TARGET_ROI);
    }
}

void testResetAndClearTarget() {
    const cv::Size frame(1920, 1080);
    // reset：清空全部跨帧状态，下一帧回到 NO_TARGET（无目标、无检测）
    {
        ArmorRoi::Planner planner(baseParams());
        planner.observeTarget(true, 10.0, cv::Point2f(960.0f, 540.0f), true);
        const ArmorRoi::Plan before = planner.plan(frame);
        CHECK_EQ(before.mode, ArmorRoi::Mode::TARGET_ROI);
        CHECK(planner.mode() == ArmorRoi::Mode::TARGET_ROI);

        planner.reset();
        CHECK(planner.mode() == ArmorRoi::Mode::OFF);

        const ArmorRoi::Plan after = planner.plan(frame);
        CHECK_EQ(after.mode, ArmorRoi::Mode::NO_TARGET);
        CHECK_EQ(after.views.size(), 2u);
        CHECK(after.views[0].full_frame);
        CHECK(!after.views[1].full_frame);
    }
    // 投影点无效：清除目标状态
    {
        ArmorRoi::Planner planner(baseParams());
        planner.observeTarget(true, 10.0, cv::Point2f(960.0f, 540.0f), true);
        const ArmorRoi::Plan before = planner.plan(frame);
        CHECK_EQ(before.mode, ArmorRoi::Mode::TARGET_ROI);
        planner.observeTarget(true, 10.0, cv::Point2f(960.0f, 540.0f), false);
        const ArmorRoi::Plan after = planner.plan(frame);
        CHECK_EQ(after.mode, ArmorRoi::Mode::NO_TARGET);
    }
    // 距离无效（0）：清除目标状态
    {
        ArmorRoi::Planner planner(baseParams());
        planner.observeTarget(true, 10.0, cv::Point2f(960.0f, 540.0f), true);
        const ArmorRoi::Plan before = planner.plan(frame);
        CHECK_EQ(before.mode, ArmorRoi::Mode::TARGET_ROI);
        planner.observeTarget(true, 0.0, cv::Point2f(960.0f, 540.0f), true);
        const ArmorRoi::Plan after = planner.plan(frame);
        CHECK_EQ(after.mode, ArmorRoi::Mode::NO_TARGET);
    }
}

void testFrameEqualsInput() {
    // 画面尺寸恰好等于网络输入尺寸：分块与 ROI 都等于整幅画面，不能产生重复视图
    const cv::Size frame(640, 640);
    // a) NO_TARGET：每帧只有全画面视图，扫描帧不再追加与全画面相同的分块
    {
        ArmorRoi::Planner planner(baseParams());
        for (int frame_index = 0; frame_index < 6; ++frame_index) {
            const ArmorRoi::Plan plan = planner.plan(frame);
            CHECK_EQ(plan.mode, ArmorRoi::Mode::NO_TARGET);
            CHECK_EQ(plan.views.size(), 1u);
            CHECK(plan.views[0].full_frame);
            CHECK(!plan.hasRoi());
        }
    }
    // b) TARGET_ROI：ROI 恰好等于整幅画面 → 退回全画面，只有全画面视图
    {
        ArmorRoi::Planner planner(baseParams());
        planner.observeTarget(true, 10.0, cv::Point2f(320.0f, 320.0f), true);
        for (int frame_index = 0; frame_index < 3; ++frame_index) {
            const ArmorRoi::Plan plan = planner.plan(frame);
            CHECK_EQ(plan.mode, ArmorRoi::Mode::TARGET_ROI);
            CHECK_EQ(plan.views.size(), 1u);
            CHECK(plan.views[0].full_frame);
            CHECK(!plan.hasRoi());
        }
    }
    // BBOX_ROI：包围盒放大后仍是最小 ROI（等于整幅画面）→ 退回全画面
    {
        ArmorRoi::Planner planner(baseParams());
        ArmorRoi::Plan plan = planner.plan(frame);
        CHECK_EQ(plan.mode, ArmorRoi::Mode::NO_TARGET);
        CHECK_EQ(plan.views.size(), 1u);
        std::vector<std::vector<ArmorRoi::Detection>> detections(plan.views.size());
        detections[0].push_back(
            ArmorRoi::Detection{cv::Rect2f(100.0f, 120.0f, 40.0f, 30.0f), 1});
        const std::vector<std::vector<char>> keep = planner.filterDetections(plan, detections);
        CHECK_EQ(static_cast<int>(keep[0].size()), 1);
        CHECK_EQ(static_cast<int>(keep[0][0]), 1);
        for (int frame_index = 0; frame_index < 3; ++frame_index) {
            plan = planner.plan(frame);
            CHECK_EQ(plan.mode, ArmorRoi::Mode::BBOX_ROI);
            CHECK_EQ(plan.views.size(), 1u);
            CHECK(plan.views[0].full_frame);
            CHECK(!plan.hasRoi());
        }
    }
}

void testFilterDetectionsMismatch() {
    ArmorRoi::Planner planner(baseParams());
    const ArmorRoi::Plan plan = planner.plan(cv::Size(1920, 1080));
    std::vector<std::vector<ArmorRoi::Detection>> too_few(plan.views.size() - 1);
    CHECK_THROWS_INVALID((void)planner.filterDetections(plan, too_few));
    std::vector<std::vector<ArmorRoi::Detection>> too_many(plan.views.size() + 1);
    CHECK_THROWS_INVALID((void)planner.filterDetections(plan, too_many));
}

} // namespace

int main() {
    testParamsValidation();
    testOffMode();
    testNoTargetTiles();
    testFrameEqualsInput();
    testTargetRoi();
    testRoiAspect();
    testBboxRoi();
    testFullHold();
    testDeduplication();
    testBoundaryMargin();
    testResetAndClearTarget();
    testFilterDetectionsMismatch();

    std::cout << "检查项: " << g_checks << "，失败: " << g_failures << std::endl;
    if (g_failures == 0) {
        std::cout << "全部通过" << std::endl;
        return 0;
    }
    return 1;
}
