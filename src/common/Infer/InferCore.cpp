// InferCore.cpp — 公共推理核心实现
#include "Infer/InferCore.h"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace Infer {

// ==========================================================================
// Preprocessor 实现
// ==========================================================================

Preprocessor::Preprocessor(int width, int height, int num_threads)
    : width_(width), height_(height), pool_(num_threads) {}

void Preprocessor::preprocess(const std::vector<const cv::Mat*>& imgs,
                              std::vector<cv::Mat*>& out) {
    int n = (int)imgs.size();
    pool_.run_parallel(n, [&](int i) {
        cv::resize(*imgs[i], *out[i], cv::Size(width_, height_));
    });
}

// ==========================================================================
// InferEngine 实现
// ==========================================================================

void InferEngine::init(const std::string& model_path_xml,
                       const std::string& model_path_bin,
                       const std::string& device,
                       int width, int height,
                       int max_batch) {
    width_ = width;
    height_ = height;
    max_batch_ = max_batch;
    compiled_models_.reserve(max_batch_);
    infer_requests_.reserve(max_batch_);

    // 为每种 batch_size (1..max_batch_) 编译一个静态模型。
    // 模型内部若写死 batch（如 Reshape 目标 shape 含 1），较高 batch 编译会失败：
    // 此时打印警告并回退到已编译的最大 batch（至少 batch=1 可用）。
    for (int bs = 1; bs <= max_batch_; ++bs) {
        try {
            auto model = core_.read_model(model_path_xml, model_path_bin);

            // reshape 到固定的 batch size（此时模型还是 NCHW）
            model->reshape({{model->input().get_any_name(),
                             ov::Shape{(size_t)bs, 3, (size_t)height_, (size_t)width_}}});

            ov::preprocess::PrePostProcessor ppp(model);
            ppp.input().tensor()
                .set_element_type(ov::element::u8)
                .set_layout("NHWC")
                .set_color_format(ov::preprocess::ColorFormat::BGR);
            ppp.input().preprocess()
                .convert_element_type(ov::element::f32)
                .convert_color(ov::preprocess::ColorFormat::RGB)
                .scale({255.f, 255.f, 255.f});
            ppp.input().model().set_layout("NCHW");
            ppp.output().tensor().set_element_type(ov::element::f32);
            model = ppp.build();

            auto compiled = core_.compile_model(model, device,
                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
            auto request = compiled.create_infer_request();

            compiled_models_.push_back(std::move(compiled));
            infer_requests_.push_back(std::move(request));

            std::cout << "[InferCore] Compiled batch=" << bs
                      << "  Input: " << compiled_models_.back().input().get_shape()
                      << "  Output: " << compiled_models_.back().output().get_shape() << std::endl;

            // 预热：跑一次全白图像推理
            cv::Mat white(height_, width_, CV_8UC3, cv::Scalar(255, 255, 255));
            size_t single_size = (size_t)height_ * width_ * 3;
            std::vector<uint8_t> blob(bs * single_size);
            for (int i = 0; i < bs; ++i)
                std::memcpy(blob.data() + i * single_size, white.data, single_size);

            ov::Tensor input_tensor(ov::element::u8,
                                    {(size_t)bs, (size_t)height_, (size_t)width_, 3},
                                    blob.data());
            infer_requests_[bs - 1].set_input_tensor(input_tensor);
            infer_requests_[bs - 1].infer();
            std::cout << "[InferCore] Warm-up batch=" << bs << " done." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[InferCore] batch=" << bs << " compile/warmup failed: " << e.what() << std::endl;
            std::cerr << "[InferCore] Falling back to max batch = " << (int)compiled_models_.size() << std::endl;
            break;
        }
    }
    if (compiled_models_.empty()) {
        throw std::runtime_error("InferCore: failed to compile even batch=1");
    }
    std::cout << "[InferCore] " << compiled_models_.size() << " batch models compiled and warmed up." << std::endl;
}

InferEngine::InferEngine(const std::string& model_path_xml,
                         const std::string& model_path_bin,
                         const std::string& device,
                         int width, int height,
                         int max_batch) {
    init(model_path_xml, model_path_bin, device, width, height, max_batch);
}

InferEngine::InferEngine(const std::string& model_path_onnx,
                         const std::string& device,
                         int width, int height,
                         int max_batch) {
    // 检查 IR 文件是否已存在
    std::string base_path = model_path_onnx;
    size_t dot_pos = base_path.rfind(".onnx");
    if (dot_pos != std::string::npos)
        base_path = base_path.substr(0, dot_pos);
    std::string xml_path = base_path + ".xml";
    std::string bin_path = base_path + ".bin";

    bool need_convert = true;
    FILE* f_xml = fopen(xml_path.c_str(), "r");
    FILE* f_bin = fopen(bin_path.c_str(), "r");
    if (f_xml && f_bin) {
        need_convert = false;
        std::cout << "[INFO] IR files already exist, skipping conversion." << std::endl;
    }
    if (f_xml) fclose(f_xml);
    if (f_bin) fclose(f_bin);

    if (need_convert) {
        auto [xml, bin] = convertOnnxToIR(model_path_onnx);
        xml_path = xml;
        bin_path = bin;
    }

    init(xml_path, bin_path, device, width, height, max_batch);
}

std::pair<std::string, std::string> InferEngine::convertOnnxToIR(const std::string& onnx_path) {
    ov::Core core;
    std::cout << "[INFO] Loading ONNX model: " << onnx_path << std::endl;
    auto model = core.read_model(onnx_path);

    std::string base_path = onnx_path;
    size_t dot_pos = base_path.rfind(".onnx");
    if (dot_pos != std::string::npos)
        base_path = base_path.substr(0, dot_pos);
    std::string xml_path = base_path + ".xml";
    std::string bin_path = base_path + ".bin";

    std::cout << "[INFO] Serializing model to IR format..." << std::endl;
    ov::serialize(model, xml_path, bin_path);
    std::cout << "[INFO] IR files generated: " << xml_path << ", " << bin_path << std::endl;

    return {xml_path, bin_path};
}

// 执行一次确定 batch 的推理，返回该 batch 的输出张量
std::shared_ptr<ov::Tensor> InferEngine::inferBatch(
    const std::vector<const cv::Mat*>& preprocessed_imgs,
    int batch_idx) {
    int bs = batch_idx + 1;  // batch_idx 0 → batch=1
    if (bs != static_cast<int>(preprocessed_imgs.size())) {
        std::cerr << "[ERROR] inferBatch: input size mismatch" << std::endl;
        return nullptr;
    }
    if (batch_idx < 0 || batch_idx >= (int)compiled_models_.size()) {
        std::cerr << "[ERROR] inferBatch: batch_idx " << batch_idx
                  << " out of compiled range [0, " << compiled_models_.size() << ")" << std::endl;
        return nullptr;
    }

    size_t single_img_size = (size_t)height_ * width_ * 3;
    std::vector<uint8_t> blob(bs * single_img_size);

    // 直接拼接已预处理的图像（无需 resize）
    for (int i = 0; i < bs; ++i) {
        std::memcpy(blob.data() + i * single_img_size,
                    preprocessed_imgs[i]->data, single_img_size);
    }

    ov::Tensor input_tensor(ov::element::u8,
                            {(size_t)bs, (size_t)height_, (size_t)width_, 3},
                            blob.data());
    auto& request = infer_requests_[batch_idx];
    request.set_input_tensor(input_tensor);
    request.infer();

    return std::make_shared<ov::Tensor>(request.get_output_tensor());
}

std::vector<InferenceOutput> InferEngine::runInference(
    const std::vector<const cv::Mat*>& preprocessed_imgs) {
    std::vector<InferenceOutput> results;
    size_t total = preprocessed_imgs.size();
    if (total == 0)
        return results;

    results.reserve(total);

    // 实际可用的最大 batch（init 时若高 batch 编译失败已回退）
    const size_t max_usable_batch = compiled_models_.size();

    // 贪心拆分为最优 batch 组合：尽量用最大可用 batch，剩余部分取最大可能的 batch
    size_t cursor = 0;
    while (cursor < total) {
        size_t remaining = total - cursor;
        int take = static_cast<int>(std::min(remaining, max_usable_batch));

        std::vector<const cv::Mat*> batch;
        batch.reserve(take);
        for (int i = 0; i < take; ++i)
            batch.push_back(preprocessed_imgs[cursor + i]);

        auto tensor = inferBatch(batch, take - 1);
        for (int i = 0; i < take; ++i)
            results.emplace_back(tensor, i);

        cursor += take;
    }

    return results;
}

} // namespace Infer
