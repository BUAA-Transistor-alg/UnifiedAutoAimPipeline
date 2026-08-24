// LatestSlot 单元测试（独立编译，验证缓冲位语义）
// 验证点：
//  1) publish 覆盖未取走的旧数据（latest-wins）；
//  2) take 取走即清空；
//  3) 空槽阻塞、stop 唤醒并退出；
//  4) 生产者快于消费者时只保留最新数据（丢弃中间帧）。
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include "common/LatestSlot.h"

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { ++g_fail; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)

int main() {
    // 1. latest-wins：连续 publish 3 次不消费，只保留最后一个
    {
        LatestSlot<int> slot;
        slot.publish(1);
        slot.publish(2);
        slot.publish(3);
        int out = -1;
        bool ok = slot.take(out);
        CHECK(ok && out == 3);
        // 取走后槽空：非阻塞验证——用第二个 take 开新线程？这里用 stop 语义即可
        slot.stop();
        CHECK(!slot.take(out));
    }

    // 2. 消费线程 drain：publish 多次，消费者逐个取完
    {
        LatestSlot<int> slot;
        std::vector<int> got;
        std::thread t([&] {
            int v;
            while (slot.take(v)) got.push_back(v);
        });
        slot.publish(10);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        slot.publish(20);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        slot.stop();
        t.join();
        CHECK(got == std::vector<int>({10, 20}));
    }

    // 3. 覆盖语义：消费者慢，中间值被丢弃
    {
        LatestSlot<int> slot;
        std::atomic<int> consumed{0};
        std::thread t([&] {
            int v;
            while (slot.take(v)) {
                consumed.store(v);
                std::this_thread::sleep_for(std::chrono::milliseconds(30));  // 慢消费
            }
        });
        slot.publish(1);
        slot.publish(2);   // 未取走 → 覆盖
        slot.publish(3);   // 覆盖
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        slot.stop();
        t.join();
        CHECK(consumed.load() == 3);   // 最终消费到的是最新值
    }

    // 4. stop 时槽内仍有数据：take 先返回数据，再返回 false
    {
        LatestSlot<int> slot;
        slot.publish(7);
        slot.stop();
        int out = -1;
        bool ok = slot.take(out);
        CHECK(ok && out == 7);
        CHECK(!slot.take(out));
    }

    // 5. tryTake：非阻塞；无数据立即 false，有数据取走即清空
    {
        LatestSlot<int> slot;
        int out = -1;
        CHECK(!slot.tryTake(out));          // 空槽
        slot.publish(5);
        CHECK(slot.tryTake(out) && out == 5);
        CHECK(!slot.tryTake(out));          // 已取走
        slot.publish(6);
        slot.publish(7);                    // 覆盖未取走的 6
        CHECK(slot.tryTake(out) && out == 7);
        slot.stop();
        CHECK(!slot.tryTake(out));
    }

    if (g_fail == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("%d FAILED\n", g_fail);
    return 1;
}
