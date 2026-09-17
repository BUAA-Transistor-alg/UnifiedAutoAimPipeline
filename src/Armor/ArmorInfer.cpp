// ArmorInfer.cpp — 装甲板检测推理实现（预处理/推理委托公共 InferCore，后处理为 Armor 特有）
#include "Armor/ArmorInfer.h"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace ArmorDetect;

namespace {
void validateModelName(const std::string& name) {
    if (name != "0526" && name != "0726")
        throw std::invalid_argument("Unsupported Armor model: " + name);
}
// Identical rounding and padding for image preprocessing and inverse coordinates.
struct LetterboxTransform {
    float scale;
    int width, height, left, top;
};
LetterboxTransform letterboxTransform(int w, int h, int target_w, int target_h) {
    const float scale = std::min(float(target_w) / w, float(target_h) / h);
    const int width = std::min(target_w, int(std::round(w * scale)));
    const int height = std::min(target_h, int(std::round(h * scale)));
    if (width < 1 || height < 1)
        throw std::runtime_error("Armor letterbox: invalid image dimensions");
    return {scale, width, height, (target_w - width) / 2, (target_h - height) / 2};
}
} // namespace

// ==========================================================================
// ArmorPreprocessor 实现（公共 InferCore 预处理）
// ==========================================================================

ArmorPreprocessor::ArmorPreprocessor(int input_width, int input_height, int num_threads,
                                     const std::string& model_name)
    : width_(input_width), height_(input_height),
      model_name_(model_name), pool_(num_threads) {
    validateModelName(model_name_);
}

void ArmorPreprocessor::preprocess(const std::vector<const cv::Mat*>& imgs,
                                     std::vector<cv::Mat*>& out) {
    pool_.run_parallel(int(imgs.size()), [&](int i) {
        if (model_name_ == "0526") {
            cv::resize(*imgs[i], *out[i], cv::Size(width_, height_));
            return;
        }
        if (imgs[i]->empty()) throw std::runtime_error("Armor: empty input image");
        const auto t = letterboxTransform(imgs[i]->cols, imgs[i]->rows, width_, height_);
        cv::Mat resized;
        cv::resize(*imgs[i], resized, cv::Size(t.width, t.height), 0, 0, cv::INTER_LINEAR);
        cv::copyMakeBorder(resized, *out[i], t.top, height_ - t.height - t.top,
                          t.left, width_ - t.width - t.left,
                          cv::BORDER_CONSTANT, cv::Scalar(124, 124, 124));
    });
}

// ==========================================================================
// ArmorInfer 实现（公共 InferCore 推理引擎）
// ==========================================================================

ArmorInfer::ArmorInfer(const std::string& model_path_xml,
                           const std::string& model_path_bin,
                           const std::string& device,
                           int input_width, int input_height,
                           int max_batch,
                           std::shared_ptr<ov::Core> shared_core,
                           const std::string& cache_dir)
    : engine_(std::make_unique<Infer::InferEngine>(
          model_path_xml, model_path_bin, device, input_width, input_height,
          max_batch, std::move(shared_core), cache_dir)) {}

ArmorInfer::ArmorInfer(const std::string& model_path_onnx,
                           const std::string& device,
                           int input_width, int input_height,
                           int max_batch,
                           std::shared_ptr<ov::Core> shared_core,
                           const std::string& cache_dir)
    : engine_(std::make_unique<Infer::InferEngine>(
          model_path_onnx, device, input_width, input_height,
          max_batch, std::move(shared_core), cache_dir)) {}

std::vector<InferenceOutput> ArmorInfer::runInference(
    const std::vector<const cv::Mat*>& preprocessed_imgs,
    char* out_area, size_t out_cap) {
    return engine_->runInference(preprocessed_imgs, out_area, out_cap);
}

// ==========================================================================
// ArmorPostprocessor 实现
// ==========================================================================

ArmorPostprocessor::ArmorPostprocessor(int input_width, int input_height, int num_threads,
                                       const std::string& model_name)
    : model_name_(model_name), input_width_(input_width), input_height_(input_height),
      pool_(num_threads) {
    validateModelName(model_name_);
}

void ArmorPostprocessor::postprocessBatch(
    const std::vector<BatchOutput>& outputs,
    const std::vector<int>& orig_ws,
    const std::vector<int>& orig_hs,
    int detect_color,
    float conf_threshold,
    float nms_threshold,
    std::vector<std::vector<Object>>& out) {
    int n = (int)outputs.size();
    out.resize(n);
    pool_.run_parallel(n, [&](int i) {
        out[i] = postprocess(outputs[i].data, outputs[i].rows, outputs[i].cols,
                             orig_ws[i], orig_hs[i],
                             detect_color, conf_threshold, nms_threshold);
    });
}

std::vector<Object> ArmorPostprocessor::postprocess(
    const float* data,
    int rows,
    int cols,
    int orig_w,
    int orig_h,
    int detect_color,
    float conf_threshold,
    float nms_threshold) {
    if (model_name_ == "0726")
        return postprocess0726(data, rows, cols, orig_w, orig_h, detect_color,
                             conf_threshold, nms_threshold);
    std::vector<Object> detections;
    if (!data)
        return detections;

    const int num_anchors = rows;
    const int out_dim     = cols;
    if (out_dim != OUTPUT_DIM) {
        std::cerr << "[ERROR] postprocess: output dim mismatch! expected ("
                  << "num_anchors, " << OUTPUT_DIM << ") got ("
                  << num_anchors << ", " << out_dim << ")" << std::endl;
        return detections;
    }
    if (num_anchors <= 0) {
        std::cerr << "[ERROR] postprocess: invalid num_anchors: " << num_anchors << std::endl;
        return detections;
    }

    // 行优先布局 (bs, num_anchors, OUTPUT_DIM)；anchor 数随输入分辨率动态变化
    // （640→25200，512→16128，320→6300），一律以实际输出形状为准
    cv::Mat output_buffer(num_anchors, OUTPUT_DIM, CV_32F, const_cast<float*>(data));

    const float sx = (float)orig_w / input_width_;
    const float sy = (float)orig_h / input_height_;

    std::vector<Object> objects;
    std::vector<cv::Rect> boxes;
    std::vector<float> class_scores;

    auto sigmoid = [](double x) {
        if (x > 0) return 1.0 / (1.0 + std::exp(-x));
        else       return std::exp(x) / (1.0 + std::exp(x));
    };

    for (int i = 0; i < num_anchors; ++i) {
        // 置信度（需 sigmoid）
        float confidence = output_buffer.at<float>(i, 8);
        confidence = (float)sigmoid(confidence);
        if (confidence < conf_threshold)
            continue;

        // 颜色和类别独热向量
        cv::Mat color_scores   = output_buffer.row(i).colRange(9, 9 + NUM_COLOR);
        cv::Mat classes_scores = output_buffer.row(i).colRange(13, 13 + NUM_CLASSES);
        cv::Point class_id, color_id;
        int _class_id, _color_id;
        double score_color, score_num;
        cv::minMaxLoc(classes_scores, NULL, &score_num, NULL, &class_id);
        cv::minMaxLoc(color_scores, NULL, &score_color, NULL, &color_id);

        // None / Purple 丢掉
        if (color_id.x == 2 || color_id.x == 3)
            continue;
        else if (detect_color == 0 && color_id.x == 1)   // detect blue
            continue;
        else if (detect_color == 1 && color_id.x == 0)   // detect red
            continue;

        _class_id = class_id.x;
        _color_id = color_id.x;

        // 保留所有类别的物体（哨兵/1~5号机器人/装甲板/基地，label 0~8）；
        // 类别分类由下游阶段（流水线 processStage4）按 obj.label 处理
        Object obj;
        obj.prob = confidence;
        obj.color = _color_id;
        obj.label = _class_id;
        for (int k = 0; k < 8; ++k)
            obj.landmarks[k] = output_buffer.at<float>(i, k);
        obj.length = cv::norm(cv::Point2f(obj.landmarks[0] - obj.landmarks[6]) -
                              cv::Point2f(obj.landmarks[1] - obj.landmarks[7]));
        obj.width  = cv::norm(cv::Point2f(obj.landmarks[0] - obj.landmarks[2]) -
                              cv::Point2f(obj.landmarks[1] - obj.landmarks[3]));
        obj.ratio  = obj.length / obj.width;

        // landmarks 为左上逆时针，points 应为左上顺时针
        std::vector<cv::Point2f> points;
        points.push_back(cv::Point2f(obj.landmarks[0], obj.landmarks[1]));
        points.push_back(cv::Point2f(obj.landmarks[6], obj.landmarks[7]));
        points.push_back(cv::Point2f(obj.landmarks[4], obj.landmarks[5]));
        points.push_back(cv::Point2f(obj.landmarks[2], obj.landmarks[3]));

        float min_x = points[0].x, max_x = points[0].x;
        float min_y = points[0].y, max_y = points[0].y;
        for (size_t p = 1; p < points.size(); ++p) {
            min_x = std::min(min_x, points[p].x);
            max_x = std::max(max_x, points[p].x);
            min_y = std::min(min_y, points[p].y);
            max_y = std::max(max_y, points[p].y);
        }
        cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);
        obj.rect = rect;
        objects.push_back(obj);
        boxes.push_back(rect);
        class_scores.push_back((float)score_num);
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, class_scores, conf_threshold, nms_threshold, indices);
    detections.reserve(indices.size());
    for (int valid_index : indices) {
        if (valid_index >= 0 && valid_index < (int)objects.size())
            detections.push_back(objects[valid_index]);
    }

    // 坐标缩放：模型输入分辨率 -> 原图
    for (auto& obj : detections) {
        obj.rect.x      = obj.rect.x * sx;
        obj.rect.y      = obj.rect.y * sy;
        obj.rect.width  = obj.rect.width  * sx;
        obj.rect.height = obj.rect.height * sy;
        for (int k = 0; k < 8; k += 2) {
            obj.landmarks[k]   *= sx;
            obj.landmarks[k+1] *= sy;
        }
        obj.length *= sx;
        obj.width  *= sy;
    }

    return detections;
}

// Infantry 0726: per-image [21,N] view of [B,21,N], already-sigmoid color/class scores followed by xy points.
std::vector<Object> ArmorPostprocessor::postprocess0726(
    const float* data, int rows, int cols, int orig_w, int orig_h,
    int detect_color, float conf_threshold, float nms_threshold) {
    const int expected_anchors = (input_width_ / 8) * (input_height_ / 8)
                               + (input_width_ / 16) * (input_height_ / 16)
                               + (input_width_ / 32) * (input_height_ / 32);
    if (!data || rows != 21 || cols != expected_anchors || orig_w <= 0 || orig_h <= 0) {
        std::cerr << "[ERROR] Armor 0726: expected per-image [21," << expected_anchors
                  << "] and a valid source image, got [" << rows << "," << cols << "]" << std::endl;
        return {};
    }
    const auto t = letterboxTransform(orig_w, orig_h, input_width_, input_height_);
    std::vector<Object> candidates;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for (int i = 0; i < cols; ++i) {
        auto value = [&](int channel) { return data[channel * cols + i]; };
        bool finite = true;
        for (int c = 0; c < 21; ++c) finite = finite && std::isfinite(value(c));
        if (!finite) continue;
        int label = 0, color = 0;
        for (int c = 1; c < 9; ++c)
            if (value(4 + c) > value(4 + label)) label = c;
        const float score = value(4 + label);
        if (score < conf_threshold) continue;
        for (int c = 1; c < 4; ++c)
            if (value(c) > value(color)) color = c;
        if (color >= 2) continue; // White/purple are intentionally unsupported.
        const int internal_color = 1 - color; // model blue=0/red=1 -> Object blue=1/red=0
        if (detect_color != 2 && internal_color != detect_color) continue;
        Object obj{};
        obj.label = label;
        obj.color = internal_color;
        obj.prob = score;
        std::vector<cv::Point2f> points;
        bool valid = true;
        for (int k = 0; k < 4; ++k) {
            const float x = (value(13 + 2*k) - t.left) / t.scale;
            const float y = (value(14 + 2*k) - t.top) / t.scale;
            if (x < 0 || x > orig_w || y < 0 || y > orig_h) valid = false;
            obj.landmarks[2*k] = x;
            obj.landmarks[2*k+1] = y;
            points.emplace_back(x, y);
        }
        if (!valid) continue;
        obj.length = cv::norm(points[0] - points[3]);
        obj.width = cv::norm(points[0] - points[1]);
        if (obj.length <= 0 || obj.width <= 0 || std::abs(cv::contourArea(points)) < 1e-3) continue;
        obj.ratio = obj.length / obj.width;
        const cv::Rect box = cv::boundingRect(points);
        obj.rect = box;
        candidates.push_back(obj);
        boxes.push_back(box);
        scores.push_back(score);
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);
    std::vector<Object> result;
    result.reserve(indices.size());
    for (int index : indices) result.push_back(candidates[index]);
    return result;
}
