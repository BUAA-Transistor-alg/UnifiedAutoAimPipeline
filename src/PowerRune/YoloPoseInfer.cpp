#include "YoloPoseInfer.h"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace YoloPose;

// ==========================================================================
// YoloPosePreprocessor 实现（公共 InferCore 预处理）
// ==========================================================================

YoloPosePreprocessor::YoloPosePreprocessor(int input_width, int input_height, int num_threads)
    : impl_(input_width, input_height, num_threads) {}

void YoloPosePreprocessor::preprocess(const std::vector<const cv::Mat*>& imgs,
                                      std::vector<cv::Mat*>& out) {
    impl_.preprocess(imgs, out);
}

// ==========================================================================
// YoloPoseInfer 实现（公共 InferCore 推理引擎）
// ==========================================================================

YoloPoseInfer::YoloPoseInfer(const std::string& model_path_xml,
                             const std::string& model_path_bin,
                             const std::string& device,
                             int input_width, int input_height,
                             int max_batch)
    : engine_(std::make_unique<Infer::InferEngine>(
          model_path_xml, model_path_bin, device, input_width, input_height, max_batch)) {}

YoloPoseInfer::YoloPoseInfer(const std::string& model_path_onnx,
                             const std::string& device,
                             int input_width, int input_height,
                             int max_batch)
    : engine_(std::make_unique<Infer::InferEngine>(
          model_path_onnx, device, input_width, input_height, max_batch)) {}

std::vector<InferenceOutput> YoloPoseInfer::runInference(
    const std::vector<const cv::Mat*>& preprocessed_imgs) {
    return engine_->runInference(preprocessed_imgs);
}

// ==========================================================================
// YoloPosePostprocessor 实现
// ==========================================================================

YoloPosePostprocessor::YoloPosePostprocessor(bool manual_nms, int input_width, int input_height,
                                             int num_threads)
    : manual_nms_(manual_nms), input_width_(input_width), input_height_(input_height),
      pool_(num_threads) {}

void YoloPosePostprocessor::postprocessBatch(
    const std::vector<std::shared_ptr<ov::Tensor>>& tensors,
    const std::vector<int>& batch_indices,
    const std::vector<int>& orig_ws,
    const std::vector<int>& orig_hs,
    float conf_threshold,
    std::vector<std::vector<PoseDetection>>& out) {
    int n = (int)tensors.size();
    out.resize(n);
    pool_.run_parallel(n, [&](int i) {
        out[i] = postprocess(tensors[i], batch_indices[i], orig_ws[i], orig_hs[i], conf_threshold);
    });
}

std::vector<PoseDetection> YoloPosePostprocessor::postprocess(
    const std::shared_ptr<ov::Tensor>& output_tensor,
    int batch_index,
    int orig_w,
    int orig_h,
    float conf_threshold) {
    if (!output_tensor)
        return {};

    auto shape = output_tensor->get_shape();
    if (shape.size() != 3 || batch_index < 0 || (size_t)batch_index >= shape[0]) {
        std::cerr << "[ERROR] postprocess: invalid tensor shape or batch_index" << std::endl;
        return {};
    }

    float* data = output_tensor->data<float>();
    int dim1 = (int)shape[1];
    int dim2 = (int)shape[2];

    if (manual_nms_) {
        // 无 NMS 导出的原始输出: (bs, RAW_OUTPUT_DIM, num_anchors)，通道优先
        if (dim1 != RAW_OUTPUT_DIM) {
            std::cerr << "[ERROR] Raw output channel mismatch! expected " << RAW_OUTPUT_DIM
                      << " got " << dim1 << std::endl;
            return {};
        }
        int num_anchors = dim2;
        float* batch_data = data + batch_index * RAW_OUTPUT_DIM * num_anchors;
        return postprocessRaw(batch_data, num_anchors, orig_w, orig_h, conf_threshold);
    } else {
        // NMS 已内嵌的输出: (bs, num_dets, OUTPUT_DIM)
        if (dim2 != OUTPUT_DIM) {
            std::cerr << "[ERROR] NMS output dim mismatch! expected " << OUTPUT_DIM
                      << " got " << dim2 << std::endl;
            return {};
        }
        int num_dets = dim1;
        float* batch_data = data + batch_index * num_dets * OUTPUT_DIM;
        return postprocessNms(batch_data, num_dets, orig_w, orig_h, conf_threshold);
    }
}

std::vector<PoseDetection> YoloPosePostprocessor::postprocessNms(const float* data, int num_dets,
                                                                  int orig_w, int orig_h,
                                                                  float conf_threshold) {
    std::vector<PoseDetection> detections;
    float scale_x = (float)orig_w / input_width_;
    float scale_y = (float)orig_h / input_height_;

    for (int i = 0; i < num_dets; ++i) {
        const float* det = data + i * OUTPUT_DIM;
        float x1 = det[0], y1 = det[1], x2 = det[2], y2 = det[3];
        float confidence = det[4];
        int class_id = static_cast<int>(std::round(det[5]));

        if (confidence < conf_threshold || x2 <= x1 || y2 <= y1)
            continue;

        cv::Rect2f rect;
        rect.x = x1 * scale_x;
        rect.y = y1 * scale_y;
        rect.width = (x2 - x1) * scale_x;
        rect.height = (y2 - y1) * scale_y;

        std::vector<cv::Point3f> kpts;
        kpts.reserve(NUM_KEYPOINTS);
        for (int k = 0; k < NUM_KEYPOINTS; ++k) {
            float kx = det[6 + k*3 + 0] * scale_x;
            float ky = det[6 + k*3 + 1] * scale_y;
            float vis = det[6 + k*3 + 2];
            kpts.emplace_back(kx, ky, vis);
        }

        detections.push_back({rect, class_id, confidence, std::move(kpts)});
    }
    return detections;
}

std::vector<PoseDetection> YoloPosePostprocessor::postprocessRaw(const float* data, int num_anchors,
                                                                 int orig_w, int orig_h,
                                                                 float conf_threshold) {
    // data 布局 (RAW_OUTPUT_DIM, num_anchors)，通道优先：
    //   [0:4]              box (cx, cy, w, h)，已是模型输入分辨率空间像素坐标
    //   [4:4+NUM_CLASSES]  类别分数（已 sigmoid）
    //   [12:...]           关键点 (x, y, vis)，x/y 为像素坐标，vis 已 sigmoid
    const float* box_ch = data;
    const float* cls_ch = data + 4 * num_anchors;
    const float* kpt_ch = data + (4 + NUM_CLASSES) * num_anchors;

    const float scale_x = (float)orig_w / input_width_;
    const float scale_y = (float)orig_h / input_height_;

    struct Candidate {
        cv::Rect2f rect;
        int class_id = 0;
        float confidence = 0.0f;
        std::vector<cv::Point3f> keypoints;
    };
    std::vector<Candidate> candidates;

    for (int a = 0; a < num_anchors; ++a) {
        // 取 8 类中分数最大的类别
        float max_score = -1.0f;
        int cls_id = -1;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float s = cls_ch[c * num_anchors + a];
            if (s > max_score) {
                max_score = s;
                cls_id = c;
            }
        }
        if (max_score < conf_threshold) continue;

        float cx = box_ch[0 * num_anchors + a];
        float cy = box_ch[1 * num_anchors + a];
        float w  = box_ch[2 * num_anchors + a];
        float h  = box_ch[3 * num_anchors + a];
        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;
        if (x2 <= x1 || y2 <= y1) continue;

        Candidate cand;
        cand.class_id = cls_id;
        cand.confidence = max_score;
        cand.rect = cv::Rect2f(x1 * scale_x, y1 * scale_y, (x2 - x1) * scale_x, (y2 - y1) * scale_y);
        cand.keypoints.reserve(NUM_KEYPOINTS);
        for (int k = 0; k < NUM_KEYPOINTS; ++k) {
            float kx = kpt_ch[(k * 3 + 0) * num_anchors + a] * scale_x;
            float ky = kpt_ch[(k * 3 + 1) * num_anchors + a] * scale_y;
            float vis = kpt_ch[(k * 3 + 2) * num_anchors + a];
            cand.keypoints.emplace_back(kx, ky, vis);
        }
        candidates.push_back(std::move(cand));
    }

    // 按置信度降序排序，做类内 NMS
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.confidence > b.confidence; });

    constexpr float kIouThreshold = 0.7f;  // 与导出时 iou=0.7 保持一致
    std::vector<Candidate> kept;
    kept.reserve(candidates.size());
    for (const Candidate& cand : candidates) {
        bool suppressed = false;
        for (const Candidate& k : kept) {
            if (k.class_id != cand.class_id) continue;  // 仅同类之间做 NMS
            float ix1 = std::max(cand.rect.x, k.rect.x);
            float iy1 = std::max(cand.rect.y, k.rect.y);
            float ix2 = std::min(cand.rect.x + cand.rect.width, k.rect.x + k.rect.width);
            float iy2 = std::min(cand.rect.y + cand.rect.height, k.rect.y + k.rect.height);
            float iw = std::max(0.0f, ix2 - ix1);
            float ih = std::max(0.0f, iy2 - iy1);
            float inter = iw * ih;
            float area_c = cand.rect.width * cand.rect.height;
            float area_k = k.rect.width * k.rect.height;
            float iou = inter / (area_c + area_k - inter + 1e-6f);
            if (iou > kIouThreshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(cand);
            if ((int)kept.size() >= MAX_DET) break;
        }
    }

    std::vector<PoseDetection> detections;
    detections.reserve(kept.size());
    for (const Candidate& k : kept) {
        PoseDetection d;
        d.rect = k.rect;
        d.class_id = k.class_id;
        d.confidence = k.confidence;
        d.keypoints = k.keypoints;
        detections.push_back(std::move(d));
    }
    return detections;
}
