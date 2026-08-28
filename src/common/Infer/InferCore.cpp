// InferCore.cpp — 公共推理核心实现
#include "common/Infer/InferCore.h"
#include "common/Infer/InferShm.h"   // InferShm::alignedOutputBytes（输出区布局规则）
#include "common/PathResolver.h"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <filesystem>

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
                       int max_batch,
                       std::shared_ptr<ov::Core> shared_core,
                       const std::string& cache_dir) {
    width_ = width;
    height_ = height;
    max_batch_ = max_batch;
    cache_dir_ = cache_dir;
    // 缓存目录不存在时自动创建（OpenVINO 编译缓存 + ONNX→IR 转换产物存放处）
    if (!cache_dir_.empty())
        std::filesystem::create_directories(cache_dir_);
    core_ = shared_core ? std::move(shared_core) : std::make_shared<ov::Core>();
    // OpenVINO 编译缓存：编译好的模型（含各 batch 变体）写入 cache_dir，下次启动
    // 直接命中缓存，省去重新编译（各推理进程用各自目录，互不干扰）
    if (!cache_dir_.empty())
        core_->set_property(ov::cache_dir(cache_dir_));
    compiled_models_.reserve(max_batch_);
    infer_requests_.reserve(max_batch_);

    // ── AMD GPU（OpenCL 后端）兼容性：FP32 直编 ──
    // OpenVINO GPU 插件默认按 FP16 精度运行模型，但部分 FP16 内核（如 1x1 卷积的
    // convolution_gpu_bfyx_1x1_hgemm_*）使用 Intel 特有 block-read 指令
    // （intel_sub_group_block_read_us8 等），AMD OpenCL 编译器不支持；混合精度
    // 模型（FP32 输入 + FP16 权重）甚至触发 AMD 驱动 GPU 内存错误直接崩溃。
    // 因此 AMD GPU 上对 FP32 输入模型直接以 FP32 推理精度编译（FP32 内核不依赖
    // Intel 特有指令，实测可正常编译运行）。FP16 输入模型（如 Armor 0526）保持
    // 默认路径（FP16 更快）。Intel GPU / CPU 不受影响。
    ov::AnyMap compile_cfg{
        ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)};
    if (device.find("GPU") != std::string::npos) {
        try {
            std::string arch = core_->get_property(device, ov::device::architecture);
            if (arch.find("vendor=0x1002") != std::string::npos) {  // AMD PCI vendor ID
                auto probe = core_->read_model(model_path_xml, model_path_bin);
                if (probe->input().get_element_type() != ov::element::f16) {
                    compile_cfg[ov::hint::inference_precision.name()] = ov::element::f32;
                    std::cerr << "[InferCore] AMD GPU + FP32 输入模型：直接以 FP32 推理精度编译"
                              << "（规避 AMD OpenCL 对 FP16 内核/驱动的兼容性问题）..." << std::endl;
                }
            }
        } catch (const std::exception&) {
        }
    }

    // 为每种 batch_size (1..max_batch_) 编译一个静态模型。
    // 模型内部若写死 batch（如 Reshape 目标 shape 含 1），较高 batch 编译会失败：
    // 此时打印警告并回退到已编译的最大 batch（至少 batch=1 可用）。
    for (int bs = 1; bs <= max_batch_; ++bs) {
        try {
            auto model = core_->read_model(model_path_xml, model_path_bin);

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

            auto compiled = core_->compile_model(model, device, compile_cfg);
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
                         int max_batch,
                         std::shared_ptr<ov::Core> shared_core,
                         const std::string& cache_dir) {
    init(model_path_xml, model_path_bin, device, width, height, max_batch,
         std::move(shared_core), cache_dir);
}

InferEngine::InferEngine(const std::string& model_path_onnx,
                         const std::string& device,
                         int width, int height,
                         int max_batch,
                         std::shared_ptr<ov::Core> shared_core,
                         const std::string& cache_dir) {
    // 检查 IR 文件是否已存在：优先在 cache_dir 中查找（不存在时自动创建目录），
    // 否则回退到 ONNX 同目录
    std::string base_path = model_path_onnx;
    size_t dot_pos = base_path.rfind(".onnx");
    if (dot_pos != std::string::npos)
        base_path = base_path.substr(0, dot_pos);

    std::string xml_path, bin_path;
    if (!cache_dir.empty()) {
        std::filesystem::create_directories(cache_dir);
        // IR 文件名与 ONNX 同名（去掉扩展名），放在缓存目录下
        std::filesystem::path onnx_fs(model_path_onnx);
        std::string stem = onnx_fs.stem().string();
        xml_path = (std::filesystem::path(cache_dir) / (stem + ".xml")).string();
        bin_path = (std::filesystem::path(cache_dir) / (stem + ".bin")).string();
    } else {
        xml_path = base_path + ".xml";
        bin_path = base_path + ".bin";
    }

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
        auto [xml, bin] = convertOnnxToIR(model_path_onnx, cache_dir);
        xml_path = xml;
        bin_path = bin;
    }

    init(xml_path, bin_path, device, width, height, max_batch,
         std::move(shared_core), cache_dir);
}

std::pair<std::string, std::string> InferEngine::convertOnnxToIR(
    const std::string& onnx_path, const std::string& cache_dir) {
    ov::Core core;
    std::cout << "[INFO] Loading ONNX model: " << onnx_path << std::endl;
    auto model = core.read_model(onnx_path);

    std::string base_path = onnx_path;
    size_t dot_pos = base_path.rfind(".onnx");
    if (dot_pos != std::string::npos)
        base_path = base_path.substr(0, dot_pos);

    std::string xml_path, bin_path;
    if (!cache_dir.empty()) {
        // 转换产物写入缓存目录（不存在时自动创建）
        std::filesystem::create_directories(cache_dir);
        std::filesystem::path onnx_fs(onnx_path);
        std::string stem = onnx_fs.stem().string();
        xml_path = (std::filesystem::path(cache_dir) / (stem + ".xml")).string();
        bin_path = (std::filesystem::path(cache_dir) / (stem + ".bin")).string();
    } else {
        xml_path = base_path + ".xml";
        bin_path = base_path + ".bin";
    }

    std::cout << "[INFO] Serializing model to IR format..." << std::endl;
    ov::serialize(model, xml_path, bin_path);
    std::cout << "[INFO] IR files generated: " << xml_path << ", " << bin_path << std::endl;

    return {xml_path, bin_path};
}

// 执行一次确定 batch 的推理，返回该 batch 的输出张量
std::shared_ptr<ov::Tensor> InferEngine::inferBatch(
    const std::vector<const cv::Mat*>& preprocessed_imgs,
    int batch_idx, char* out_ptr) {
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

    // 直接包装外部输入内存（推理进程侧即共享内存输入区）：请求期间输入区不会被
    // 覆盖（客户端在收到响应后才开始写下一批），免去中间 blob 的分配与 memcpy。
    // 各图在输入区按 img_bytes 步长连续存放（客户端 resize 时写入），故以
    // {bs, h, w, 3} 包装首图地址即可覆盖整批。
    ov::Tensor input_tensor(ov::element::u8,
                            {(size_t)bs, (size_t)height_, (size_t)width_, 3},
                            preprocessed_imgs[0]->data);
    auto& request = infer_requests_[batch_idx];
    request.set_input_tensor(input_tensor);

    // 零拷贝输出：把输出张量直接绑定到调用方内存（推理进程侧即共享内存输出区），
    // 插件在 infer() 时直接把结果写入该内存（CPU 零拷贝直写；GPU 为 D2H 直写），
    // 免去"插件内部缓冲 → memcpy → 输出区"的一整份拷贝。模型输出形状是静态的
    // （每种 batch 一个编译模型），可直接按编译输出形状包装外部内存。
    if (out_ptr != nullptr) {
        const ov::Shape& out_shape = compiled_models_[batch_idx].output().get_shape();
        request.set_output_tensor(ov::Tensor(ov::element::f32, out_shape, out_ptr));
    }
    request.infer();

    // 个别插件/版本可能忽略用户提供的输出张量：校验数据实际落点，不一致则回退
    // 显式拷贝。无论哪种情况，返回的张量都包装 out_ptr，保证调用方的布局假设成立。
    if (out_ptr != nullptr) {
        ov::Tensor actual = request.get_output_tensor();
        if (actual.data() != static_cast<void*>(out_ptr)) {
            std::cerr << "[InferCore] plugin ignored user output tensor, fallback copy"
                      << std::endl;
            std::memcpy(out_ptr, actual.data<float>(), actual.get_byte_size());
        }
        return std::make_shared<ov::Tensor>(
            ov::element::f32, compiled_models_[batch_idx].output().get_shape(), out_ptr);
    }
    return std::make_shared<ov::Tensor>(request.get_output_tensor());
}

std::vector<InferenceOutput> InferEngine::runInference(
    const std::vector<const cv::Mat*>& preprocessed_imgs,
    char* out_area, size_t out_cap) {
    std::vector<InferenceOutput> results;
    size_t total = preprocessed_imgs.size();
    if (total == 0)
        return results;

    results.reserve(total);

    // 调试：UAP_DEBUG_INFER=1 时打印每次推理的 batch / 耗时 / 输出形状
    static const bool dbg = [] {
        const char* e = std::getenv("UAP_DEBUG_INFER");
        return e && std::string(e) == "1";
    }();

    // 实际可用的最大 batch（init 时若高 batch 编译失败已回退）
    const size_t max_usable_batch = compiled_models_.size();

    // 贪心拆分为最优 batch 组合：尽量用最大可用 batch，剩余部分取最大可能的 batch
    size_t cursor = 0;
    size_t offset = 0;   // 输出区游标：与 InferShm 布局一致，每 batch 块 64B 对齐
    while (cursor < total) {
        size_t remaining = total - cursor;
        int take = static_cast<int>(std::min(remaining, max_usable_batch));

        std::vector<const cv::Mat*> batch;
        batch.reserve(take);
        for (int i = 0; i < take; ++i)
            batch.push_back(preprocessed_imgs[cursor + i]);

        // 该 batch 模型的输出形状静态已知，推理前即可算出本块大小与写入位置
        const ov::Shape& out_shape = compiled_models_[take - 1].output().get_shape();
        size_t r = out_shape.size() > 1 ? out_shape[out_shape.size() - 2] : 1;
        size_t c = out_shape.size() > 0 ? out_shape[out_shape.size() - 1] : 1;
        size_t block = InferShm::alignedOutputBytes(take, r, c);

        if (out_area != nullptr && offset + block > out_cap) {
            std::cerr << "[InferCore] output area overflow (" << offset + block
                      << " > " << out_cap << "), dropping remaining batch" << std::endl;
            break;
        }
        char* out_ptr = out_area != nullptr ? out_area + offset : nullptr;

        auto t0 = std::chrono::steady_clock::now();
        auto tensor = inferBatch(batch, take - 1, out_ptr);
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        if (dbg) {
            std::cout << "[InferDebug] batch=" << take << " infer_us=" << us;
            if (tensor) {
                auto shp = tensor->get_shape();
                std::cout << " out_shape=[";
                for (size_t i = 0; i < shp.size(); ++i)
                    std::cout << shp[i] << (i + 1 < shp.size() ? "," : "");
                std::cout << "]";
            }
            std::cout << std::endl;
        }
        for (int i = 0; i < take; ++i)
            results.emplace_back(tensor, i);

        cursor += take;
        offset += block;
    }

    return results;
}

} // namespace Infer
