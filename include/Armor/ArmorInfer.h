// ArmorInfer.h — 装甲板检测推理封装（预处理/推理使用公共 InferCore，解码为 Armor 特有）
//
// 模型：输入 (bs, 3, H, W)，输出 (bs, num_anchors, 22)；输入分辨率由机器配置文件
// （config/robots/*.yaml）的 armor.inference.resolution 提供（当前模型 640×640 → 25200 anchors，
// 512×512 → 16128，320×320 → 6300；anchor 数随分辨率变化，后处理按输出形状动态读取）：
//   col 0-7    4 个关键点 xy（左上/左下/右下/右上）
//   col 8      obj 置信度（需 sigmoid）
//   col 9-12   颜色独热（0=red, 1=blue, 2/3=丢弃）
//   col 13-21  类别独热（9 类）
#ifndef ARMOR_INFER_H
#define ARMOR_INFER_H

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include <string>
#include <utility>
#include <memory>

#include "common/TaskPool.h"
#include "common/Infer/InferCore.h"

namespace ArmorDetect {

// ---- 共享常量（输出列数固定，anchor 数随分辨率动态变化，见 postprocess） ----
constexpr int OUTPUT_DIM    = 22;    // 模型输出列数（8 kpts + 1 conf + 4 color + 9 class）
constexpr int NUM_COLOR     = 4;
constexpr int NUM_CLASSES   = 9;

// 类别映射（模型训练标签）：0-5 哨兵/1~5号机器人，6 装甲板，7-8 基地/基地大装甲。
// 后处理保留所有类别（label 0~8），类别分类由下游流水线 processStage4 按 obj.label
// 处理；ARMOR_CLASS（label 6 装甲板）用于 OutpostESEKF 等仅关注装甲板的目标。
constexpr int ARMOR_CLASS = 6;
constexpr int BASE_CLASS = 7;        // 基地
constexpr int BASE_LARGE_CLASS = 8;  // 基地大装甲

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
// ArmorPreprocessor – 批量预处理（多线程 resize，公共 InferCore）
// ==========================================================================
class ArmorPreprocessor {
public:
    /// @param input_width/input_height  模型输入分辨率（像素，须与模型输入一致）
    /// @param num_threads               内部 TaskPool 线程数, 0 = 自动
    ArmorPreprocessor(int input_width, int input_height, int num_threads = 0);

    /// 批量预处理：将原始图像 resize 到 input_width × input_height
    void preprocess(const std::vector<const cv::Mat*>& imgs,
                    std::vector<cv::Mat*>& out);

private:
    Infer::Preprocessor impl_;
};

// ==========================================================================
// ArmorInfer – 推理（接收已预处理的图像，动态 batch，公共 InferCore）
// ==========================================================================
class ArmorInfer {
public:
    /// @param shared_core 可选的共享 ov::Core（多流水线共用同一 GPU context，避免
    ///                    同进程多 Core 对彼此推理的显著拖慢），为空则自建
    /// @param cache_dir 可选的模型缓存目录（不存在时自动创建）：作为 OpenVINO
    ///                  编译缓存（ov::cache_dir），且 ONNX→IR 转换产物（xml/bin）
    ///                  也写入该目录；为空时保持旧行为
    ArmorInfer(const std::string& model_path_xml,
                 const std::string& model_path_bin,
                 const std::string& device,
                 int input_width, int input_height,
                 int max_batch = 4,
                 std::shared_ptr<ov::Core> shared_core = nullptr,
                 const std::string& cache_dir = "");

    ArmorInfer(const std::string& model_path_onnx,
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

// ==========================================================================
// ArmorPostprocessor – 后处理（输出张量 → Object 列表，含手动 NMS）
// ==========================================================================

/// 单图推理输出视图：指向 PipelineData 自持的输出缓冲（行优先 f32）
struct BatchOutput {
    const float* data;   // 输出数据（num_anchors × OUTPUT_DIM）
    int rows;            // 张量 shape[1]（armor 为 num_anchors）
    int cols;            // 张量 shape[2]（= OUTPUT_DIM）
};

class ArmorPostprocessor {
public:
    /// @param input_width/input_height  模型输入分辨率（后处理坐标缩放基准）
    /// @param num_threads               线程池线程数，0 = 自动
    ArmorPostprocessor(int input_width, int input_height, int num_threads = 0);

    /// 处理单个推理输出（每图一个独立输出缓冲）
    /// @param detect_color 0=仅红, 1=仅蓝, 2=双色
    /// @return 该图的所有检测结果（坐标已缩放到原图尺寸）
    std::vector<Object> postprocess(const float* data, int rows, int cols,
                                    int orig_w,
                                    int orig_h,
                                    int detect_color,
                                    float conf_threshold,
                                    float nms_threshold);

    /// 批量并行后处理（多线程）
    void postprocessBatch(const std::vector<BatchOutput>& outputs,
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

} // namespace ArmorDetect

#endif // ARMOR_INFER_H
