// armor_infer_process_main.cpp — Armor 推理进程
//
// 独立进程：仅做 Armor 模型推理（编译 + 预热 + 循环处理共享内存请求）。
// 预处理/后处理仍在主进程。启动方式（先于主程序）：
//   ./bin/armor_infer_process
// 共享内存 Key 取 config armor.inference.shm_key。
#include "common/RobotConfig.h"
#include "common/PathResolver.h"
#include "common/Infer/InferShmServer.h"
#include "Armor/ArmorInfer.h"

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
    std::string model_path = (!cfg.armor.modelPath.empty() && cfg.armor.modelPath[0] == '/')
        ? cfg.armor.modelPath
        : PathResolver::resolvePath(cfg.armor.modelPath);

    // 本进程专属模型缓存目录（OpenVINO 编译缓存 + ONNX→IR 转换产物存放处）：
    // <项目根>/cache/armor，不存在时自动创建（InferEngine::init 中创建）
    std::string cache_dir = PathResolver::resolvePath("cache/armor");

    std::cout << "========================================" << std::endl;
    std::cout << "Armor Infer Process" << std::endl;
    std::cout << "    Model:   " << model_path << std::endl;
    std::cout << "    Device:  " << cfg.armor.device << std::endl;
    std::cout << "    Input:   " << cfg.armor.inputWidth << "x" << cfg.armor.inputHeight
              << "  max_batch=" << cfg.armor.maxBatch << std::endl;
    std::cout << "    Shm key: " << cfg.armor.shmKey << std::endl;
    std::cout << "    缓存目录: " << cache_dir << std::endl;
    std::cout << "========================================" << std::endl;

    auto engine = std::make_unique<ArmorDetect::ArmorInfer>(
        model_path, cfg.armor.device,
        cfg.armor.inputWidth, cfg.armor.inputHeight, cfg.armor.maxBatch,
        nullptr, cache_dir);

    Infer::InferShmServer server(cfg.armor.shmKey,
        [&](const std::vector<const cv::Mat*>& imgs, char* out_area, size_t out_cap) {
            return engine->runInference(imgs, out_area, out_cap);
        });
    server.run([]() { return g_running != 0; });

    std::cout << "[InferProcess] Armor inference process exiting." << std::endl;
    return 0;
}
