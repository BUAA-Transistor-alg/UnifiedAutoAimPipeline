// power_rune_infer_process_main.cpp — PowerRune 推理进程
//
// 独立进程：仅做 PowerRune 模型推理（编译 + 预热 + 循环处理共享内存请求）。
// 预处理/后处理仍在主进程。启动方式（先于主程序）：
//   ./bin/power_rune_infer_process
// 共享内存 Key 取 config power_rune.inference.shm_key。
#include "common/RobotConfig.h"
#include "common/PathResolver.h"
#include "common/Infer/InferShmServer.h"
#include "PowerRune/YoloPoseInfer.h"

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
    std::string model_path = (!cfg.powerRune.modelPath.empty() && cfg.powerRune.modelPath[0] == '/')
        ? cfg.powerRune.modelPath
        : PathResolver::resolvePath(cfg.powerRune.modelPath);

    // 本进程专属模型缓存目录（OpenVINO 编译缓存 + ONNX→IR 转换产物存放处）：
    // <项目根>/cache/power_rune，不存在时自动创建（InferEngine::init 中创建）
    std::string cache_dir = PathResolver::resolvePath("cache/power_rune");

    std::cout << "========================================" << std::endl;
    std::cout << "PowerRune Infer Process" << std::endl;
    std::cout << "    Model:   " << model_path << std::endl;
    std::cout << "    Device:  " << cfg.powerRune.device << std::endl;
    std::cout << "    Input:   " << cfg.powerRune.inputWidth << "x" << cfg.powerRune.inputHeight
              << "  max_batch=" << cfg.powerRune.maxBatch << std::endl;
    std::cout << "    Shm key: " << cfg.powerRune.shmKey << std::endl;
    std::cout << "    缓存目录: " << cache_dir << std::endl;
    std::cout << "========================================" << std::endl;

    auto engine = std::make_unique<YoloPose::YoloPoseInfer>(
        model_path, cfg.powerRune.device,
        cfg.powerRune.inputWidth, cfg.powerRune.inputHeight, cfg.powerRune.maxBatch,
        nullptr, cache_dir);

    Infer::InferShmServer server(cfg.powerRune.shmKey,
        [&](const std::vector<const cv::Mat*>& imgs) {
            return engine->runInference(imgs);
        });
    server.run([]() { return g_running != 0; });

    std::cout << "[InferProcess] PowerRune inference process exiting." << std::endl;
    return 0;
}
