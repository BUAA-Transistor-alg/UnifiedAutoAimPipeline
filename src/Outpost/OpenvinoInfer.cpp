// OpenvinoInfer.cpp — 前哨站检测推理实现（预处理/推理委托公共 InferCore，后处理为 Outpost 特有）
#include "Outpost/OpenvinoInfer.h"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>

using namespace OutpostDetect;

// ==========================================================================
// OutpostPreprocessor 实现（公共 InferCore 预处理）
// ==========================================================================

OutpostPreprocessor::OutpostPreprocessor(int input_width, int input_height, int num_threads)
    : impl_(input_width, input_height, num_threads) {}

void OutpostPreprocessor::preprocess(const std::vector<const cv::Mat*>& imgs,
                                     std::vector<cv::Mat*>& out) {
    impl_.preprocess(imgs, out);
}

// ==========================================================================
// OutpostInfer 实现（公共 InferCore 推理引擎）
// ==========================================================================

OutpostInfer::OutpostInfer(const std::string& model_path_xml,
                           const std::string& model_path_bin,
                           const std::string& device,
                           int input_width, int input_height,
                           int max_batch,
                           std::shared_ptr<ov::Core> shared_core,
                           const std::string& cache_dir)
    : engine_(std::make_unique<Infer::InferEngine>(
          model_path_xml, model_path_bin, device, input_width, input_height,
          max_batch, std::move(shared_core), cache_dir)) {}

OutpostInfer::OutpostInfer(const std::string& model_path_onnx,
                           const std::string& device,
                           int input_width, int input_height,
                           int max_batch,
                           std::shared_ptr<ov::Core> shared_core,
                           const std::string& cache_dir)
    : engine_(std::make_unique<Infer::InferEngine>(
          model_path_onnx, device, input_width, input_height,
          max_batch, std::move(shared_core), cache_dir)) {}

std::vector<InferenceOutput> OutpostInfer::runInference(
    const std::vector<const cv::Mat*>& preprocessed_imgs) {
    return engine_->runInference(preprocessed_imgs);
}

// ==========================================================================
// OutpostPostprocessor 实现
// ==========================================================================

OutpostPostprocessor::OutpostPostprocessor(int input_width, int input_height, int num_threads)
    : input_width_(input_width), input_height_(input_height), pool_(num_threads) {}

void OutpostPostprocessor::postprocessBatch(
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

std::vector<Object> OutpostPostprocessor::postprocess(
    const float* data,
    int rows,
    int cols,
    int orig_w,
    int orig_h,
    int detect_color,
    float conf_threshold,
    float nms_threshold) {
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

        // 仅保留前哨站类别（label 6）的装甲板，其余类别（哨兵/1~5号机器人/基地）直接丢弃
        if (_class_id != OUTPOST_CLASS)
            continue;

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
