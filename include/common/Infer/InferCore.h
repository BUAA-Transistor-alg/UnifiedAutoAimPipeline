// InferCore.h — 公共推理核心（预处理 + 推理引擎）
//
// 从 Outpost 的 OpenvinoInfer 与 PowerRune 的 YoloPoseInfer 中抽取的公共部分：
//   1. Preprocessor  — 批量预处理（线程池并行 resize 到固定 W×H）
//   2. InferEngine   — 动态 batch 1..N：每种 batch 编译一个静态模型 + 白图预热 +
//                      THROUGHPUT 性能模式 + ONNX→IR 转换 + 贪心拆批
//
// 各模型的常量（输入尺寸 / 输出解码 / NMS）由各项目自己的 Infer/Postprocess
// 类提供，本核心只负责"预处理 + 推理 + 批量调度"。
#ifndef INFER_CORE_H
#define INFER_CORE_H

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include <string>
#include <utility>
#include <memory>

#include "common/TaskPool.h"

namespace Infer {

// 推理输出：一个 batch 的输出张量 + 该图在此 batch 中的索引（同 batch 共享张量）
using InferenceOutput = std::pair<std::shared_ptr<ov::Tensor>, int>;

// ==========================================================================
// Preprocessor – 批量预处理（多线程 resize）
// ==========================================================================
class Preprocessor {
public:
    /// @param width/height 目标输入尺寸；num_threads 内部 TaskPool 线程数, 0 = 自动
    explicit Preprocessor(int width, int height, int num_threads = 0);

    /// 批量预处理：将原始图像 resize 到 width × height
    void preprocess(const std::vector<const cv::Mat*>& imgs,
                    std::vector<cv::Mat*>& out);

    int width() const { return width_; }
    int height() const { return height_; }

private:
    int width_;
    int height_;
    TaskPool pool_;
};

// ==========================================================================
// InferEngine – 推理（接收已预处理图像，动态 batch 编译 + 贪心拆批）
// ==========================================================================
class InferEngine {
public:
    /// 直接使用 IR 模型（xml + bin）
    /// @param shared_core 可选的共享 ov::Core：多条流水线共用同一 Core（同一 GPU
    ///                    context），避免同进程多 Core 共存对彼此推理的显著拖慢
    ///                    （2026.3 GPU 插件实测：同进程两套推理器各建 Core 时，
    ///                    outpost 帧率被拖到 ~62；共享一个 Core 后无此问题）。
    ///                    为空时自建一个。
    InferEngine(const std::string& model_path_xml,
                const std::string& model_path_bin,
                const std::string& device,
                int width, int height,
                int max_batch = 4,
                std::shared_ptr<ov::Core> shared_core = nullptr);

    /// 使用 ONNX 模型（若同目录 IR 不存在则先转换）
    InferEngine(const std::string& model_path_onnx,
                const std::string& device,
                int width, int height,
                int max_batch = 4,
                std::shared_ptr<ov::Core> shared_core = nullptr);

    /// 主推理接口：接收已预处理（resize 到 width×height）的图像指针向量
    /// @return 每个输入图对应一个 InferenceOutput（同一 batch 共享同一张量）
    std::vector<InferenceOutput> runInference(
        const std::vector<const cv::Mat*>& preprocessed_imgs);

    int inputWidth() const { return width_; }
    int inputHeight() const { return height_; }
    int maxBatch() const { return max_batch_; }
    /// 实际可用的最大 batch（init 时若高 batch 编译失败已回退）
    int usableMaxBatch() const { return static_cast<int>(compiled_models_.size()); }

private:
    void init(const std::string& model_path_xml,
              const std::string& model_path_bin,
              const std::string& device,
              int width, int height, int max_batch,
              std::shared_ptr<ov::Core> shared_core);

    static std::pair<std::string, std::string> convertOnnxToIR(const std::string& onnx_path);

    /// 执行单次确定 batch 的推理，返回该 batch 的输出张量
    std::shared_ptr<ov::Tensor> inferBatch(
        const std::vector<const cv::Mat*>& preprocessed_imgs,
        int batch_idx);

    std::shared_ptr<ov::Core> core_;
    int width_;
    int height_;
    int max_batch_;
    // index i 对应 batch_size = i+1
    std::vector<ov::CompiledModel> compiled_models_;
    std::vector<ov::InferRequest> infer_requests_;
};

} // namespace Infer

#endif // INFER_CORE_H
