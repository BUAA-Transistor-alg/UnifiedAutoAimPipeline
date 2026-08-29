// infer_force_restart_test.cpp — 推理挂死强制重启功能验证（临时测试，不入 CMake）
//
// 覆盖：
//   1) InferShmClient：正常返回计时 / 响应超时检测 / 间隔判定 / 节流 / reconnect
//      重置 / 功能关闭（timeout <= 0）；配合进程内假服务端（InferShm 协议）。
//   2) InferProcessManager：startAndWait / forceRestart 并发去重 / 重复重启 /
//      切换完成后丢弃过时强制重启（switch_count_ 规则）/ shutdown。
//      （需 bin/armor_infer_process 临时替换为"打印 ready 后 sleep"的伪造进程，
//        伪造进程就绪延迟由环境变量 FAKE_READY_DELAY 控制，单位秒。）
//
// 编译（参考 build/CMakeFiles/unified_auto_aim.dir/flags.make）：
//   c++ -O2 -g -std=gnu++17 -DPROJECT_ROOT=\"<项目根>\" \
//       -I include -I third_party -isystem /usr/include/opencv4 \
//       test/infer_force_restart_test.cpp \
//       src/common/Infer/InferShmClient.cpp src/common/Infer/InferProcessManager.cpp \
//       src/common/PathResolver.cpp -lopencv_core -lpthread -o build/infer_force_restart_test
#include "common/Infer/InferShmClient.h"
#include "common/Infer/InferProcessManager.h"

#include <sys/ipc.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

using namespace std::chrono;

static int g_failures = 0;
#define CHECK(cond, msg)                                        \
    do {                                                        \
        if (cond) {                                             \
            std::cout << "[OK]   " << msg << std::endl;         \
        } else {                                                \
            std::cout << "[FAIL] " << msg << std::endl;         \
            ++g_failures;                                       \
        }                                                       \
    } while (0)

// ── 进程内假服务端：按 InferShm 协议响应请求，可切换为挂死（不再响应）──
static void fakeServerLoop(int shm_key, std::atomic<bool>& hang) {
    const size_t size = InferShm::totalSize();
    int shm_id = shmget(shm_key, size, IPC_CREAT | 0666);
    if (shm_id == -1) { std::cerr << "[server] shmget failed" << std::endl; return; }
    auto* shm = static_cast<InferShm::ShmLayout*>(shmat(shm_id, nullptr, 0));
    if (shm == reinterpret_cast<void*>(-1)) { std::cerr << "[server] shmat failed" << std::endl; return; }
    sem_unlink(InferShm::reqSemName(shm_key).c_str());
    sem_unlink(InferShm::respSemName(shm_key).c_str());
    sem_t* req  = sem_open(InferShm::reqSemName(shm_key).c_str(), O_CREAT | O_EXCL, 0666, 0);
    sem_t* resp = sem_open(InferShm::respSemName(shm_key).c_str(), O_CREAT | O_EXCL, 0666, 0);
    if (req == SEM_FAILED || resp == SEM_FAILED) { std::cerr << "[server] sem_open failed" << std::endl; return; }
    std::cout << "[server] ready, waiting for requests" << std::endl;
    while (true) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 1;
        if (sem_timedwait(req, &ts) != 0) {
            if (hang.load()) { std::cout << "[server] hang requested, stop responding" << std::endl; break; }
            continue;
        }
        if (hang.load()) { continue; }   // 已挂死：不响应
        // 响应一个 batch=1 的 [1,1] f32 张量
        shm->result_batches = 1;
        shm->batch_size[0] = 1;
        shm->out_rows[0] = 1;
        shm->out_cols[0] = 1;
        float val = 1.0f;
        std::memcpy(reinterpret_cast<char*>(shm) + InferShm::outputOffset(), &val, sizeof(val));
        sem_post(resp);
    }
    sem_close(req);
    sem_close(resp);
    shmdt(shm);
}

// ── 1) 推理客户端挂死检测 ──
static int testClient() {
    std::cout << "\n===== 客户端挂死检测（limit = 5.0s，响应超时 2s）=====" << std::endl;
    const int key = 23401;
    std::atomic<bool> hang{false};
    std::thread server(fakeServerLoop, key, std::ref(hang));
    std::this_thread::sleep_for(milliseconds(500));

    Infer::InferShmClient client(key, /*force_restart_timeout_sec=*/5.0);
    std::atomic<int> calls{0};
    client.setForceRestartHandler([&]() { ++calls; std::cout << "[client] 触发强制重启回调!" << std::endl; });

    cv::Mat img(1, 1, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<const cv::Mat*> imgs{&img};
    Infer::OutputBuffer out;
    std::vector<Infer::OutputBuffer*> outs{&out};

    // 正常返回两次（记录"上次正常返回"时间）
    const bool ok1 = client.runInference(imgs, outs);
    const bool ok2 = client.runInference(imgs, outs);
    CHECK(ok1 && ok2, "正常推理返回两次成功");
    CHECK(calls.load() == 0, "正常返回不触发强制重启（不计算间隔）");

    // 服务端挂死：第一次超时距上次正常返回 ≈2s < 5s，不应触发
    hang.store(true);
    const bool r1 = client.runInference(imgs, outs);
    CHECK(!r1, "挂死后推理超时返回 false");
    CHECK(calls.load() == 0, "超时但距上次正常返回未超时限 → 不触发");

    // 间隔超过时限 → 触发一次
    std::this_thread::sleep_for(seconds(4));
    const bool r2 = client.runInference(imgs, outs);
    CHECK(!r2, "挂死后推理再次超时返回 false");
    CHECK(calls.load() == 1, "距上次正常返回超过时限 → 触发强制重启");

    // 节流：紧随其后的超时不再触发（两次触发至少间隔时限）
    const bool r3 = client.runInference(imgs, outs);
    CHECK(!r3, "节流验证：推理再次超时");
    CHECK(calls.load() == 1, "节流：同一客户端短时间内不重复触发");

    // reconnect 重置计时：进程（重新）就绪后短间隔超时不再触发
    client.reconnect();
    const bool r4 = client.runInference(imgs, outs);
    CHECK(!r4, "reconnect 后推理超时");
    CHECK(calls.load() == 1, "reconnect 重置'上次正常返回' → 短间隔不触发");

    // 再次等待超过时限 → 可再次触发（重启失败/进程再挂死的重试路径）
    std::this_thread::sleep_for(seconds(4));
    const bool r5 = client.runInference(imgs, outs);
    CHECK(!r5, "重试验证：推理再次超时");
    CHECK(calls.load() == 2, "超过时限后再次触发（重试）");

    // 功能关闭（timeout <= 0）：永不触发
    {
        std::atomic<bool> hang2{false};
        std::thread server2(fakeServerLoop, 23402, std::ref(hang2));
        std::this_thread::sleep_for(milliseconds(300));
        Infer::InferShmClient off(23402, /*force_restart_timeout_sec=*/0.0);
        std::atomic<int> off_calls{0};
        off.setForceRestartHandler([&]() { ++off_calls; });
        hang2.store(true);
        const bool rr = off.runInference(imgs, outs);
        CHECK(!rr && off_calls.load() == 0, "timeout=0 关闭功能：超时也不触发");
        server2.join();
    }

    server.join();
    return 0;
}

// ── 2) InferProcessManager 强制重启 ──
// 需要 bin/armor_infer_process 与 bin/power_rune_infer_process 均为伪造进程
// （打印 ready 后 sleep；就绪延迟见环境变量 FAKE_READY_DELAY）
static int testManager() {
    std::cout << "\n===== InferProcessManager 强制重启 =====" << std::endl;
    std::atomic<int> ready_calls{0};
    Infer::InferProcessManager mgr([&](Infer::InferProcessManager::Kind k) {
        ++ready_calls;
        std::cout << "[manager] on_ready kind=" << static_cast<int>(k)
                  << "（累计 " << ready_calls.load() << " 次）" << std::endl;
    });

    // 启动
    CHECK(mgr.startAndWait(Infer::InferProcessManager::Kind::ARMOR, 30), "startAndWait(ARMOR) 成功");
    CHECK(ready_calls.load() == 1, "启动就绪回调 1 次");

    // 并发强制重启 ×4：去重后只执行一次
    std::vector<std::thread> ts;
    for (int i = 0; i < 4; ++i) {
        ts.emplace_back([&]() { mgr.forceRestart(Infer::InferProcessManager::Kind::ARMOR); });
    }
    for (auto& t : ts) t.join();
    std::this_thread::sleep_for(milliseconds(1500));   // 等 worker 完成重启
    CHECK(ready_calls.load() == 2, "并发 forceRestart×4 去重 → 只重启一次");

    // 再次强制重启（可重复）
    mgr.forceRestart(Infer::InferProcessManager::Kind::ARMOR);
    std::this_thread::sleep_for(milliseconds(1500));
    CHECK(ready_calls.load() == 3, "再次 forceRestart 可重复重启");

    // 切换完成后丢弃排队中的过时强制重启：
    //   切到 POWER_RUNE（停止 armor、拉起 power_rune），再切回 ARMOR，并在
    //   ARMOR 进程重新拉起的耗时窗口内排队一个强制重启——该切换本身已重新拉起
    //   armor 进程，排队中的强制重启针对的是旧进程，应被丢弃（不重复拉起）。
    const int before = ready_calls.load();            // = 3
    mgr.switchTo(Infer::InferProcessManager::Kind::POWER_RUNE);
    std::this_thread::sleep_for(milliseconds(1500));  // 等 power_rune 就绪（on_ready +1）
    mgr.switchTo(Infer::InferProcessManager::Kind::ARMOR);
    std::this_thread::sleep_for(milliseconds(100));   // 切换（spawn）进行中
    mgr.forceRestart(Infer::InferProcessManager::Kind::ARMOR);
    std::this_thread::sleep_for(milliseconds(2500));  // 等切换完成 + 过时重启被丢弃
    CHECK(ready_calls.load() == before + 2,
          "切换完成并丢弃过时的强制重启 → 仅两次切换的 on_ready 计入（+2）");

    // 丢弃后去重标志已复位：再次强制重启仍可执行
    mgr.forceRestart(Infer::InferProcessManager::Kind::ARMOR);
    std::this_thread::sleep_for(milliseconds(1500));
    CHECK(ready_calls.load() == before + 3, "丢弃后 forceRestart 仍可执行");

    mgr.shutdown();
    std::cout << "[manager] shutdown 完成" << std::endl;
    return 0;
}

int main() {
    std::cout << "推理挂死强制重启功能测试" << std::endl;
    testClient();
    testManager();
    std::cout << "\n结果：" << (g_failures == 0 ? "全部通过" : "存在失败") << std::endl;
    return g_failures == 0 ? 0 : 1;
}
