#include "tcs/mpc/mcu_mpc_controller.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
using namespace std::chrono_literals;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Predicate>
tcs::McuMpcController::State waitFor(tcs::McuMpcController& controller, Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
        auto state = controller.state();
        if (predicate(state.stamp)) return state;
        std::this_thread::sleep_for(100us);
    }
    throw std::runtime_error("等待控制周期状态超时");
}
} // namespace

int main() {
    // 运行实际控制器后台循环检查参考消费及编号；通信接口为空，不进行设备发送或反馈求解。
    tcs::McuMpcController controller(nullptr, 0.01, 3, 0.1, 0, 0, 0,
                                     1, 10, 1, 1, 1, 5, 0, 0.002);
    check(!controller.state().stamp.control_tick_id, "未运行时出现了控制周期编号");
    check(controller.takeControlRecords().records.empty(), "未运行时出现了调试记录");
    controller.setControlRecordingEnabled(true);
    controller.set(true, false, std::vector<double>{0.1, 0.2, 0.3}, {}, {}, false, 10, 20);
    controller.start();
    auto state = waitFor(controller, [](const auto& stamp) {
        return stamp.control_tick_id && *stamp.control_tick_id >= 5;
    });
    check(state.stamp.frame_id == 10 && state.stamp.plan_id == 20 && state.stamp.reference_index == 2,
          "序列耗尽后未保留最后参考下标和计划编号");
    controller.stop();
    const auto previous_tick = *controller.state().stamp.control_tick_id;

    controller.set(true, false, std::vector<double>{0.4, 0.5}, {}, {}, false, 11, 21);
    controller.start();
    state = waitFor(controller, [previous_tick](const auto& stamp) {
        return stamp.control_tick_id && *stamp.control_tick_id > previous_tick;
    });
    const auto consumed = *state.stamp.control_tick_id - previous_tick - 1;
    check(state.stamp.frame_id == 11 && state.stamp.plan_id == 21 &&
          state.stamp.reference_index == std::min<std::uint64_t>(consumed, 1),
          "新计划下标没有重新开始，或者重启后周期编号发生清零");
    controller.stop();
    const auto sequence_tick = *controller.state().stamp.control_tick_id;

    controller.set(false, false, 0.6, 0.1, false, false, 12, 22);
    controller.start();
    state = waitFor(controller, [sequence_tick](const auto& stamp) {
        return stamp.control_tick_id && *stamp.control_tick_id > sequence_tick;
    });
    check(state.stamp.frame_id == 12 && state.stamp.plan_id == 22 && !state.stamp.reference_index,
          "单目标参考继承了旧序列下标");

    // 连续替换不同长度的计划，同时检查结果关联信息始终来自同一次 set()。
    for (std::uint64_t i = 100; i < 200; ++i) {
        controller.set(true, false, std::vector<double>(i % 3 + 1, 0.1), {}, {}, false, i, i + 1000);
        state = waitFor(controller, [i](const auto& stamp) { return stamp.frame_id == i; });
        check(state.stamp.plan_id == i + 1000 && state.stamp.reference_index &&
              *state.stamp.reference_index < i % 3 + 1,
              "计划替换时出现跨计划的编号或参考下标");
    }
    controller.stop();
    const auto replacement_tick = *controller.state().stamp.control_tick_id;

    controller.set(false, false, std::vector<double>{}, {}, {}, false);
    controller.start();
    state = waitFor(controller, [replacement_tick](const auto& stamp) {
        return stamp.control_tick_id && *stamp.control_tick_id > replacement_tick;
    });
    check(!state.stamp.frame_id && !state.stamp.plan_id && !state.stamp.reference_index,
          "空序列或省略编号时继承了旧计划关联");
    controller.stop();
    const auto last_tick = *controller.state().stamp.control_tick_id;
    const auto batch = controller.takeControlRecords();
    check(batch.dropped_records == 0 && batch.records.size() == last_tick + 1,
          "批量取出出现重复、遗漏或错误的溢出数量");
    for (std::size_t i = 0; i < batch.records.size(); ++i) {
        const auto& record = batch.records[i];
        check(record.stamp.control_tick_id == i, "批量记录没有按控制周期排列");
        check(record.started_at <= record.command_submitted_at &&
              record.command_submitted_at <= record.finished_at, "周期时间戳顺序错误");
        check(!record.command_send_succeeded && !record.reference_position,
              "未连接通信设备时出现了发送或求解数据");
        if (record.stamp.frame_id && *record.stamp.frame_id >= 100) {
            const auto frame = *record.stamp.frame_id;
            check(record.stamp.plan_id == frame + 1000 && record.stamp.reference_index &&
                  *record.stamp.reference_index < frame % 3 + 1,
                  "批量记录的计划编号与参考下标不匹配");
        }
    }
    check(controller.takeControlRecords().records.empty(), "重复读取返回了已取出的记录");
    check(controller.state().stamp.control_tick_id == last_tick, "取出调试记录改变了最新状态");

    controller.setControlRecordingEnabled(false);
    controller.start();
    waitFor(controller, [last_tick](const auto& stamp) {
        return stamp.control_tick_id && *stamp.control_tick_id >= last_tick + 3;
    });
    controller.stop();
    check(controller.takeControlRecords().records.empty(), "关闭记录后仍在入队");
    controller.setControlRecordingEnabled(true);
    controller.start();
    const auto disabled_tick = *controller.state().stamp.control_tick_id;
    waitFor(controller, [disabled_tick](const auto& stamp) {
        return stamp.control_tick_id && *stamp.control_tick_id > disabled_tick;
    });
    controller.stop();
    check(!controller.takeControlRecords().records.empty(), "重新开启后没有产生调试记录");

    tcs::McuMpcController bounded(nullptr, 0.01, 3, 0.1, 0, 0, 0,
                                  1, 10, 1, 1, 1, 5, 0, 0.000001);
    bounded.start();
    waitFor(bounded, [](const auto& stamp) { return stamp.control_tick_id.has_value(); });
    bounded.stop();
    check(bounded.takeControlRecords().records.empty(), "默认状态开启了调试记录");
    const auto initial_tick = *bounded.state().stamp.control_tick_id;
    bounded.setControlRecordingEnabled(true);
    bounded.start();
    waitFor(bounded, [initial_tick](const auto& stamp) {
        return stamp.control_tick_id &&
               *stamp.control_tick_id > initial_tick + tcs::McuMpcController::kControlRecordCapacity + 10;
    });
    bounded.stop();
    const auto final_tick = *bounded.state().stamp.control_tick_id;
    const auto overflow = bounded.takeControlRecords();
    check(overflow.records.size() == tcs::McuMpcController::kControlRecordCapacity &&
          overflow.dropped_records + overflow.records.size() == final_tick - initial_tick,
          "队列容量或溢出数量错误");
    check(overflow.records.front().stamp.control_tick_id == initial_tick + 1 + overflow.dropped_records &&
          overflow.records.back().stamp.control_tick_id == final_tick,
          "队列溢出没有保留最新的连续周期");
    const auto drained = bounded.takeControlRecords();
    check(drained.records.empty() && drained.dropped_records == 0, "读取后没有清空队列及溢出计数");

    std::vector<std::uint64_t> observed;
    std::atomic<bool> reader_done{false};
    auto consume = [&]() {
        auto records = bounded.takeControlRecords();
        check(records.dropped_records == 0, "并发读取期间发生溢出");
        for (const auto& record : records.records) observed.push_back(*record.stamp.control_tick_id);
    };
    std::thread reader([&] {
        while (!reader_done.load()) {
            consume();
            std::this_thread::sleep_for(100us);
        }
    });
    bounded.start();
    waitFor(bounded, [final_tick](const auto& stamp) {
        return stamp.control_tick_id && *stamp.control_tick_id >= final_tick + 100;
    });
    bounded.stop();
    reader_done.store(true);
    reader.join();
    consume();
    const auto concurrent_last_tick = *bounded.state().stamp.control_tick_id;
    check(observed.size() == concurrent_last_tick - final_tick, "并发批量读取遗漏或重复了周期");
    for (std::size_t i = 0; i < observed.size(); ++i)
        check(observed[i] == final_tick + 1 + i, "并发批量读取顺序错误");
    std::cout << "实际后台循环的周期关联、批量取出、开关、并发读取及队列容量检查通过\n";
}
