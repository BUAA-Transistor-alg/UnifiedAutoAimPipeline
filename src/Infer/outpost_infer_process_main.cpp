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

    // 本进程专属模型缓存目录（OpenVINO 编译缓存 + ONNX→IR 转换产物存放处）：
    // <项目根>/cache/outpost，不存在时自动创建（InferEngine::init 中创建）
    std::string cache_dir = PathResolver::resolvePath("cache/outpost");

    std::cout << "========================================" << std::endl;
    std::cout << "Outpost Infer Process" << std::endl;
    std::cout << "    Model:   " << model_path << std::endl;
    std::cout << "    Device:  " << cfg.outpost.device << std::endl;
    std::cout << "    Input:   " << cfg.outpost.inputWidth << "x" << cfg.outpost.inputHeight
              << "  max_batch=" << cfg.outpost.maxBatch << std::endl;
    std::cout << "    Shm key: " << cfg.outpost.shmKey << std::endl;
    std::cout << "    缓存目录: " << cache_dir << std::endl;
    std::cout << "========================================" << std::endl;

    auto engine = std::make_unique<OutpostDetect::OutpostInfer>(
        model_path, cfg.outpost.device,
        cfg.outpost.inputWidth, cfg.outpost.inputHeight, cfg.outpost.maxBatch,
        nullptr, cache_dir);

    Infer::InferShmServer server(cfg.outpost.shmKey,
        [&](const std::vector<const cv::Mat*>& imgs, char* out_area, size_t out_cap) {
            return engine->runInference(imgs, out_area, out_cap);
        });
    server.run([]() { return g_running != 0; });

    std::cout << "[InferProcess] Outpost inference process exiting." << std::endl;
    return 0;
}
