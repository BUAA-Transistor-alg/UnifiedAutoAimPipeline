// OpenvinoInfer.h — 前哨站检测推理封装（预处理/推理使用公共 InferCore，解码为 Outpost 特有）
//
// 模型：输入 (bs, 3, H, W)，输出 (bs, num_anchors, 22)；输入分辨率由 config.yaml
// 的 outpost.inference.resolution 提供（当前模型 640×640 → 25200 anchors，
// 512×512 → 16128，320×320 → 6300；anchor 数随分辨率变化，后处理按输出形状动态读取）：
//   col 0-7    4 个关键点 xy（左上/左下/右下/右上）
//   col 8      obj 置信度（需 sigmoid）
//   col 9-12   颜色独热（0=red, 1=blue, 2/3=丢弃）
//   col 13-21  类别独热（9 类）
#ifndef OPENVINO_INFER_H
#define OPENVINO_INFER_H

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include <string>
#include <utility>
#include <memory>

#include "common/TaskPool.h"
#include "common/Infer/InferCore.h"

namespace OutpostDetect {

// ---- 共享常量（输出列数固定，anchor 数随分辨率动态变化，见 postprocess） ----
constexpr int OUTPUT_DIM    = 22;    // 模型输出列数（8 kpts + 1 conf + 4 color + 9 class）
constexpr int NUM_COLOR     = 4;
constexpr int NUM_CLASSES   = 9;

// 类别映射（模型训练标签）：0-5 哨兵/1~5号机器人，6 前哨站，7-8 基地/基地大装甲。
// 识别流程仅保留前哨站类别（label 6）的装甲板，其余类别直接丢弃。
constexpr int OUTPOST_CLASS = 6;

// 检测结果
struct Object {
    cv::Rect_<float> rect;         // 边界框（原图尺寸）
    float landmarks[8];            // 4 个关键点（左上/左下/右下/右上）
    int label;                     // 类别 0~8
    float prob;                    // 置信度
    int color;                     // blue:1, red:0
    double length;                 // 长
    double width;                  // 宽
    double ratio;                  // 长宽比
};

// 推理输出：一个 batch 的输出张量 + 该图在此 batch 中的索引
using InferenceOutput = Infer::InferenceOutput;

// ==========================================================================
// OutpostPreprocessor – 批量预处理（多线程 resize，公共 InferCore）
// ==========================================================================
class OutpostPreprocessor {
public:
    /// @param input_width/input_height  模型输入分辨率（像素，须与模型输入一致）
    /// @param num_threads               内部 TaskPool 线程数, 0 = 自动
    OutpostPreprocessor(int input_width, int input_height, int num_threads = 0);

    /// 批量预处理：将原始图像 resize 到 input_width × input_height
    void preprocess(const std::vector<const cv::Mat*>& imgs,
                    std::vector<cv::Mat*>& out);

private:
    Infer::Preprocessor impl_;
};

// ==========================================================================
// OutpostInfer – 推理（接收已预处理的图像，动态 batch，公共 InferCore）
// ==========================================================================
class OutpostInfer {
public:
    /// @param shared_core 可选的共享 ov::Core（多流水线共用同一 GPU context，避免
    ///                    同进程多 Core 对彼此推理的显著拖慢），为空则自建
    OutpostInfer(const std::string& model_path_xml,
                 const std::string& model_path_bin,
                 const std::string& device,
                 int input_width, int input_height,
                 int max_batch = 4,
                 std::shared_ptr<ov::Core> shared_core = nullptr);

    OutpostInfer(const std::string& model_path_onnx,
                 const std::string& device,
                 int input_width, int input_height,
                 int max_batch = 4,
                 std::shared_ptr<ov::Core> shared_core = nullptr);

    /// 主推理接口：接收已预处理（resize 到 input_width×input_height）的图像指针向量
    /// @return 每个输入图对应一个 InferenceOutput（同一 batch 共享同一张量）
    std::vector<InferenceOutput> runInference(
        const std::vector<const cv::Mat*>& preprocessed_imgs);

private:
    std::unique_ptr<Infer::InferEngine> engine_;
};

// ==========================================================================
// OutpostPostprocessor – 后处理（输出张量 → Object 列表，含手动 NMS）
// ==========================================================================
class OutpostPostprocessor {
public:
    /// @param input_width/input_height  模型输入分辨率（后处理坐标缩放基准）
    /// @param num_threads               线程池线程数，0 = 自动
    OutpostPostprocessor(int input_width, int input_height, int num_threads = 0);

    /// 处理单个推理输出（一个 batch 张量 + 该图在 batch 中的索引）
    /// @param detect_color 0=仅红, 1=仅蓝, 2=双色
    /// @return 该图的所有检测结果（坐标已缩放到原图尺寸）
    std::vector<Object> postprocess(const std::shared_ptr<ov::Tensor>& output_tensor,
                                    int batch_index,
                                    int orig_w,
                                    int orig_h,
                                    int detect_color,
                                    float conf_threshold,
                                    float nms_threshold);

    /// 批量并行后处理（多线程）
    void postprocessBatch(const std::vector<std::shared_ptr<ov::Tensor>>& tensors,
                          const std::vector<int>& batch_indices,
                          const std::vector<int>& orig_ws,
                          const std::vector<int>& orig_hs,
                          int detect_color,
                          float conf_threshold,
                          float nms_threshold,
                          std::vector<std::vector<Object>>& out);

private:
    int input_width_;
    int input_height_;
    TaskPool pool_;
};

} // namespace OutpostDetect

#endif // OPENVINO_INFER_H
