// outpost_infer_process_main.cpp — Outpost 推理进程
//
// 独立进程：仅做 Outpost 模型推理（编译 + 预热 + 循环处理共享内存请求）。
// 预处理/后处理仍在主进程。启动方式（先于主程序）：
//   ./bin/outpost_infer_process
// 共享内存 Key 取 config outpost.inference.shm_key。
#include "common/RobotConfig.h"
#include "common/PathResolver.h"
#include "common/Infer/InferShmServer.h"
#include "Outpost/OpenvinoInfer.h"

#include <csignal>
#include <iostream>
#include <memory>

namespace {
volatile std::sig_atomic_t g_running = 1;
void onSignal(int) { g_running = 0; }
} // namespace

int main() {
    signal(SIGINT, onSignal);
    signal(SIGTERM, onSignal);

    const RobotConfig& cfg = RobotConfig::instance();
    std::string model_path = (!cfg.outpost.modelPath.empty() && cfg.outpost.modelPath[0] == '/')
        ? cfg.outpost.modelPath
        : PathResolver::resolvePath(cfg.outpost.modelPath);

    std::cout << "========================================" << std::endl;
    std::cout << "Outpost Infer Process" << std::endl;
    std::cout << "    Model:   " << model_path << std::endl;
    std::cout << "    Device:  " << cfg.outpost.device << std::endl;
    std::cout << "    Input:   " << cfg.outpost.inputWidth << "x" << cfg.outpost.inputHeight
              << "  max_batch=" << cfg.outpost.maxBatch << std::endl;
    std::cout << "    Shm key: " << cfg.outpost.shmKey << std::endl;
    std::cout << "========================================" << std::endl;

    auto engine = std::make_unique<OutpostDetect::OutpostInfer>(
        model_path, cfg.outpost.device,
        cfg.outpost.inputWidth, cfg.outpost.inputHeight, cfg.outpost.maxBatch);

    Infer::InferShmServer server(cfg.outpost.shmKey,
        [&](const std::vector<const cv::Mat*>& imgs) {
            return engine->runInference(imgs);
        });
    server.run([]() { return g_running != 0; });

    std::cout << "[InferProcess] Outpost inference process exiting." << std::endl;
    return 0;
}
