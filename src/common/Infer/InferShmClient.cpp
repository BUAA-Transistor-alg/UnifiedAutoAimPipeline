// InferShmClient.cpp — 主进程侧推理通信器实现
#include "common/Infer/InferShmClient.h"

#include <sys/ipc.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <time.h>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <stdexcept>

namespace Infer {

InferShmClient::InferShmClient(int shm_key) : shm_key_(shm_key) {
    attachSharedMemory();
    openSemaphores();
    std::cout << "[InferShmClient] attached shm key=" << shm_key_
              << " size=" << InferShm::totalSize() << " bytes" << std::endl;
}

InferShmClient::~InferShmClient() {
    if (shm_) shmdt(shm_);
    if (req_sem_)  sem_close(req_sem_);
    if (resp_sem_) sem_close(resp_sem_);
}

void InferShmClient::attachSharedMemory() {
    // 参考 transistor 项目：shmget(IPC_CREAT|0666) + shmat
    size_t size = InferShm::totalSize();
    shm_id_ = shmget(shm_key_, size, IPC_CREAT | 0666);
    if (shm_id_ == -1) {
        throw std::runtime_error(std::string("InferShmClient: shmget failed: ") +
                                 strerror(errno) +
                                 " (确认推理进程已启动，或 shm key 冲突)");
    }
    shm_ = static_cast<InferShm::ShmLayout*>(shmat(shm_id_, nullptr, 0));
    if (shm_ == reinterpret_cast<void*>(-1)) {
        shm_ = nullptr;
        throw std::runtime_error(std::string("InferShmClient: shmat failed: ") + strerror(errno));
    }
    // 清理旧的响应计数，避免上次请求的残留响应被误读
    shm_->result_batches = 0;
}

void InferShmClient::openSemaphores() {
    // 客户端以 O_CREAT 打开（不存在则创建，初值 0）；服务端负责 unlink 重建
    req_sem_  = sem_open(InferShm::reqSemName(shm_key_).c_str(), O_CREAT, 0666, 0);
    resp_sem_ = sem_open(InferShm::respSemName(shm_key_).c_str(), O_CREAT, 0666, 0);
    if (req_sem_ == SEM_FAILED || resp_sem_ == SEM_FAILED) {
        throw std::runtime_error(std::string("InferShmClient: sem_open failed: ") + strerror(errno));
    }
}

std::vector<InferenceOutput> InferShmClient::runInference(
    const std::vector<const cv::Mat*>& preprocessed_imgs) {
    std::vector<InferenceOutput> results;
    size_t total = preprocessed_imgs.size();
    if (total == 0 || total > InferShm::MAX_IMAGES)
        return results;

    // 1. 写入输入区（所有图等尺寸、连续 u8 BGR）
    const cv::Mat& first = *preprocessed_imgs[0];
    size_t img_bytes = (size_t)first.total() * first.channels();
    if (img_bytes * total > InferShm::MAX_INPUT_BYTES) {
        std::cerr << "[InferShmClient] input too large: " << img_bytes * total << std::endl;
        return results;
    }
    shm_->batch_count = (int)total;
    shm_->in_h = first.rows;
    shm_->in_w = first.cols;
    char* dst = inputArea();
    for (size_t i = 0; i < total; ++i) {
        const cv::Mat& img = *preprocessed_imgs[i];
        std::memcpy(dst + i * img_bytes, img.data, img_bytes);
    }

    // 2. 通知推理进程
    if (sem_post(req_sem_) != 0) {
        std::cerr << "[InferShmClient] sem_post(req) failed: " << strerror(errno) << std::endl;
        return results;
    }

    // 3. 等待响应（推理进程正常时毫秒级返回；用有界超时避免推理进程异常
    //    时本进程永久阻塞——超时则丢弃本批并返回空，由流水线按空结果处理）
    {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 2;   // 2s 上限（正常推理远小于此）
        if (sem_timedwait(resp_sem_, &ts) != 0) {
            std::cerr << "[InferShmClient] sem_timedwait(resp) failed: "
                      << strerror(errno) << "（推理进程未响应，丢弃本批）" << std::endl;
            return results;
        }
    }

    // 4. 重建输出张量：各 batch 一个自持内存的 ov::Tensor，同 batch 图像共享
    results.reserve(total);
    int nb = shm_->result_batches;
    if (nb < 0 || nb > InferShm::MAX_BATCHES) {
        std::cerr << "[InferShmClient] invalid result_batches: " << nb << std::endl;
        return results;
    }
    const char* src = outputArea();
    size_t offset = 0;
    for (int b = 0; b < nb; ++b) {
        int bs = shm_->batch_size[b];
        int r  = shm_->out_rows[b];
        int c  = shm_->out_cols[b];
        if (bs <= 0 || r <= 0 || c <= 0) continue;
        size_t bytes = (size_t)bs * r * c * sizeof(float);
        ov::Tensor tensor(ov::element::f32, {(size_t)bs, (size_t)r, (size_t)c});
        std::memcpy(tensor.data(), src + offset, bytes);
        offset += bytes;
        auto sp = std::make_shared<ov::Tensor>(std::move(tensor));
        for (int k = 0; k < bs; ++k)
            results.emplace_back(sp, k);
    }
    return results;
}

} // namespace Infer
