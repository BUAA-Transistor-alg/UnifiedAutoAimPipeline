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

bool InferShmClient::runInference(
    const std::vector<const cv::Mat*>& preprocessed_imgs,
    std::vector<OutputBuffer*>& outs) {
    // 与 reconnect 互斥：避免关闭/重开信号量时本函数仍在使用旧句柄
    std::lock_guard<std::mutex> lock(io_mtx_);
    size_t total = preprocessed_imgs.size();
    if (total == 0 || total > InferShm::MAX_IMAGES || outs.size() != total)
        return false;

    // 1. 写入输入区（所有图等尺寸、连续 u8 BGR；resize 已由流水线阶段1完成）
    const cv::Mat& first = *preprocessed_imgs[0];
    size_t img_bytes = (size_t)first.total() * first.channels();
    if (img_bytes * total > InferShm::MAX_INPUT_BYTES) {
        std::cerr << "[InferShmClient] input too large: " << img_bytes * total << std::endl;
        return false;
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
        return false;
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
            return false;
        }
    }

    // 4. 结果张量直接 memcpy 进调用方（PipelineData）提供的输出缓冲。
    //    输入区/输出区在同一块 SHM 中，本批数据在下一轮请求写入前保持有效，
    //    同步拷贝完成后即可安全被下一批覆盖（无需双缓冲）。
    int nb = shm_->result_batches;
    if (nb < 0 || nb > InferShm::MAX_BATCHES) {
        std::cerr << "[InferShmClient] invalid result_batches: " << nb << std::endl;
        return false;
    }
    const char* src_out = outputArea();
    size_t offset = 0;
    size_t out_idx = 0;
    for (int b = 0; b < nb; ++b) {
        int bs = shm_->batch_size[b];
        int r  = shm_->out_rows[b];
        int c  = shm_->out_cols[b];
        if (bs <= 0 || r <= 0 || c <= 0) continue;
        size_t per = (size_t)r * c * sizeof(float);
        for (int k = 0; k < bs; ++k) {
            if (out_idx >= outs.size()) break;
            OutputBuffer* out = outs[out_idx++];
            out->rows = r;
            out->cols = c;
            out->data.resize((size_t)r * c);   // 尺寸不变时复用容量，不重新分配
            std::memcpy(out->data.data(), src_out + offset + (size_t)k * per, per);
        }
        offset += (size_t)bs * per;
    }
    return out_idx == outs.size();
}

void InferShmClient::reconnect() {
    // 与 runInference 互斥：等其在途调用结束（最多 2s 响应超时）后再重开信号量
    std::lock_guard<std::mutex> lock(io_mtx_);
    if (req_sem_)  { sem_close(req_sem_);  req_sem_  = nullptr; }
    if (resp_sem_) { sem_close(resp_sem_); resp_sem_ = nullptr; }
    openSemaphores();
    if (shm_) shm_->result_batches = 0;   // 清理残留响应计数
    std::cout << "[InferShmClient] semaphores reconnected, shm key=" << shm_key_
              << std::endl;
}

} // namespace Infer
