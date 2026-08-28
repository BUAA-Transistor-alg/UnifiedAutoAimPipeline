#ifndef POWER_RUNE_INFER_H
#define POWER_RUNE_INFER_H

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "common/TaskPool.h"
#include "common/Infer/InferCore.h"
#include <vector>
#include <string>
#include <utility>
#include <memory>

struct PoseDetection {
    cv::Rect2f rect;                 // 边界框 (x, y, width, height) 基于原图尺寸
    int class_id;                    // 类别ID (0~7)
    float confidence;                // 置信度 (0~1)
    std::vector<cv::Point3f> keypoints; // 关键点: (x, y, visibility) 基于原图尺寸
};

/// PowerRune 命名空间：预处理类 与 推理类 共享的常量
namespace PowerRune {

// ---- 共享参数 ----
// 输入分辨率 INPUT_WIDTH × INPUT_HEIGHT 不再硬编码为 512×512：由机器配置文件
// （config/robots/*.yaml）的 power_rune.inference.resolution 提供（见 PowerRunePipeline 构造）。
constexpr int MAX_DET       = 300;
constexpr int NUM_KEYPOINTS = 32;
constexpr int NUM_CLASSES   = 8;   // 类别数（当前模型 8 分类）
constexpr int OUTPUT_DIM    = 6 + NUM_KEYPOINTS * 3;
constexpr int RAW_OUTPUT_DIM = 4 + NUM_CLASSES + NUM_KEYPOINTS * 3;  // 无 NMS 导出的原始输出通道数 (108)

// ---- 推理输出：一个 batch 的输出张量 + 该图在此 batch 中的索引 ----
using InferenceOutput = Infer::InferenceOutput;

// ==========================================================================
// PowerRunePreprocessor – 批量预处理（多线程 resize，公共 InferCore）
// ==========================================================================
class PowerRunePreprocessor {
public:
    /// @param input_width/input_height  模型输入分辨率（像素，须与模型输入一致）
    /// @param num_threads               内部 TaskPool 线程数, 0 = 自动
    PowerRunePreprocessor(int input_width, int input_height, int num_threads = 0);

    /// 批量预处理：将原始图像 resize 到 input_width × input_height
    void preprocess(const std::vector<const cv::Mat*>& imgs,
                    std::vector<cv::Mat*>& out);

private:
    Infer::Preprocessor impl_;
};

// ==========================================================================
// PowerRunePostprocessor – 后处理（输出张量 → PoseDetection）
// ==========================================================================

/// 单图推理输出视图：指向 PipelineData 自持的输出缓冲（行优先 f32）
struct PoseBatchOutput {
    const float* data;   // 输出数据
    int rows;            // 张量 shape[1]
    int cols;            // 张量 shape[2]
};

class PowerRunePostprocessor {
public:
    /// @param manual_nms    true: 模型为无 NMS 的原始输出，需手动 NMS；false: NMS 已内嵌
    /// @param input_width/input_height  模型输入分辨率（后处理坐标缩放基准）
    /// @param num_threads   线程池线程数，0 = 自动
    PowerRunePostprocessor(bool manual_nms, int input_width, int input_height, int num_threads = 0);

    /// 处理单个推理输出（每图一个独立输出缓冲）
    /// @return 该图的所有检测结果（坐标已缩放到原图尺寸）
    std::vector<PoseDetection> postprocess(const float* data, int rows, int cols,
                                           int orig_w,
                                           int orig_h,
                                           float conf_threshold);

    /// 批量并行后处理（多线程）
    /// @param out 输出，out[i] 为第 i 个输入图对应的检测结果
    void postprocessBatch(const std::vector<PoseBatchOutput>& outputs,
                          const std::vector<int>& orig_ws,
                          const std::vector<int>& orig_hs,
                          float conf_threshold,
                          std::vector<std::vector<PoseDetection>>& out);

private:
    bool manual_nms_;
    int input_width_;
    int input_height_;
    TaskPool pool_;

    std::vector<PoseDetection> postprocessNms(const float* data, int num_dets,
                                              int orig_w, int orig_h, float conf_threshold);

    std::vector<PoseDetection> postprocessRaw(const float* data, int num_anchors,
                                              int orig_w, int orig_h, float conf_threshold);
};

// ==========================================================================
// PowerRuneInfer – 推理（接收已预处理的图像，动态 batch，公共 InferCore）
// ==========================================================================
class PowerRuneInfer {
public:
    /// @param shared_core 可选的共享 ov::Core（多流水线共用同一 GPU context，避免
    ///                    同进程多 Core 对彼此推理的显著拖慢），为空则自建
    /// @param cache_dir 可选的模型缓存目录（不存在时自动创建）：作为 OpenVINO
    ///                  编译缓存（ov::cache_dir），且 ONNX→IR 转换产物（xml/bin）
    ///                  也写入该目录；为空时保持旧行为
    PowerRuneInfer(const std::string& model_path_xml,
                  const std::string& model_path_bin,
                  const std::string& device,
                  int input_width, int input_height,
                  int max_batch = 4,
                  std::shared_ptr<ov::Core> shared_core = nullptr,
                  const std::string& cache_dir = "");

    PowerRuneInfer(const std::string& model_path_onnx,
                  const std::string& device,
                  int input_width, int input_height,
                  int max_batch = 4,
                  std::shared_ptr<ov::Core> shared_core = nullptr,
                  const std::string& cache_dir = "");

    /// 主推理接口：接收已预处理（resize 到 input_width×input_height）的图像指针向量
    /// @param out_area 可选外部输出缓冲（共享内存输出区）；非空时输出零拷贝直写
    /// @param out_cap  out_area 容量（字节），out_area 为空时忽略
    /// @return 每个输入图对应一个 InferenceOutput（同一 batch 共享同一张量）
    std::vector<InferenceOutput> runInference(
        const std::vector<const cv::Mat*>& preprocessed_imgs,
        char* out_area = nullptr, size_t out_cap = 0);

private:
    std::unique_ptr<Infer::InferEngine> engine_;
};

} // namespace PowerRune

#endif // POWER_RUNE_INFER_H
