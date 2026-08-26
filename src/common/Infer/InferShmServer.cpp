// InferShmServer.cpp — 推理进程侧实现类
#include "common/Infer/InferShmServer.h"

#include <sys/ipc.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <time.h>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <stdexcept>

namespace Infer {

InferShmServer::InferShmServer(int shm_key, InferFn infer_fn)
    : shm_key_(shm_key), infer_fn_(std::move(infer_fn)) {
    createSharedMemory();
    createSemaphores();
    std::cout << "[InferShmServer] ready, shm key=" << shm_key_
              << " size=" << InferShm::totalSize() << " bytes" << std::endl;
}

InferShmServer::~InferShmServer() {
    if (shm_) shmdt(shm_);
    if (req_sem_)  sem_close(req_sem_);
    if (resp_sem_) sem_close(resp_sem_);
    // 信号量文件残留无害（下次启动 sem_unlink 重建）
}

void InferShmServer::createSharedMemory() {
    size_t size = InferShm::totalSize();
    shm_id_ = shmget(shm_key_, size, IPC_CREAT | 0666);
    if (shm_id_ == -1) {
        throw std::runtime_error(std::string("InferShmServer: shmget failed: ") + strerror(errno));
    }
    shm_ = static_cast<InferShm::ShmLayout*>(shmat(shm_id_, nullptr, 0));
    if (shm_ == reinterpret_cast<void*>(-1)) {
        shm_ = nullptr;
        throw std::runtime_error(std::string("InferShmServer: shmat failed: ") + strerror(errno));
    }
    shm_->result_batches = 0;
}

void InferShmServer::createSemaphores() {
    // 服务端重建信号量（清零），保证重启后与客户端协议状态一致
    sem_unlink(InferShm::reqSemName(shm_key_).c_str());
    sem_unlink(InferShm::respSemName(shm_key_).c_str());
    req_sem_  = sem_open(InferShm::reqSemName(shm_key_).c_str(), O_CREAT | O_EXCL, 0666, 0);
    resp_sem_ = sem_open(InferShm::respSemName(shm_key_).c_str(), O_CREAT | O_EXCL, 0666, 0);
    if (req_sem_ == SEM_FAILED || resp_sem_ == SEM_FAILED) {
        throw std::runtime_error(std::string("InferShmServer: sem_open failed: ") + strerror(errno));
    }
}

void InferShmServer::run(const std::function<bool()>& is_running) {
    while (is_running()) {
        // 用 sem_timedwait（100ms 超时）而非 sem_wait：glibc 的 sem_wait 对信号
        // 内部重试不返回 EINTR，会导致 SIGINT 无法唤醒阻塞中的循环；
        // 超时后回到循环顶部检查 is_running()，保证进程可被信号正常退出。
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 100 * 1000 * 1000;
        if (ts.tv_nsec >= 1000000000L) {
            ts.tv_sec += 1;
            ts.tv_nsec -= 1000000000L;
        }
        if (sem_timedwait(req_sem_, &ts) != 0) {
            if (errno == EINTR || errno == ETIMEDOUT)
                continue;   // 回循环顶部检查 is_running
            std::cerr << "[InferShmServer] sem_timedwait(req) failed: "
                      << strerror(errno) << std::endl;
            break;
        }

        if (!is_running()) break;   // 信号处理中：丢弃已唤醒的请求

        int n = shm_->batch_count;
        int h = shm_->in_h;
        int w = shm_->in_w;
        if (n <= 0 || n > InferShm::MAX_IMAGES || h <= 0 || w <= 0) {
            std::cerr << "[InferShmServer] invalid request: n=" << n
                      << " h=" << h << " w=" << w << std::endl;
            shm_->result_batches = 0;
            sem_post(resp_sem_);
            continue;
        }

        // 包装输入区为 cv::Mat（零拷贝）
        std::vector<cv::Mat> mats;
        std::vector<const cv::Mat*> ptrs;
        mats.reserve(n);
        ptrs.reserve(n);
        const char* src = inputArea();
        size_t img_bytes = (size_t)h * w * 3;
        for (int i = 0; i < n; ++i) {
            mats.emplace_back(h, w, CV_8UC3, const_cast<char*>(src + i * img_bytes));
            ptrs.push_back(&mats.back());
        }

        // 推理：输出张量由引擎经 set_output_tensor 直接写入共享内存输出区
        // （零拷贝，免去"插件缓冲 → memcpy → 输出区"），引擎按 64B 对齐块布局
        std::vector<InferenceOutput> outputs =
            infer_fn_(ptrs, outputArea(), InferShm::MAX_OUTPUT_BYTES);

        // 记录输出元数据：各 batch 张量已由引擎写入输出区，这里不再拷贝
        shm_->result_batches = 0;
        char* out = outputArea();
        size_t offset = 0;
        size_t i = 0;
        while (i < outputs.size()) {
            // 同 batch 图像共享同一 tensor 指针（runInference 保证）
            size_t bs = 1;
            while (i + bs < outputs.size() &&
                   outputs[i + bs].first.get() == outputs[i].first.get())
                ++bs;
            const auto& shape = outputs[i].first->get_shape();
            size_t r = shape.size() > 1 ? shape[shape.size() - 2] : 1;
            size_t c = shape.size() > 0 ? shape[shape.size() - 1] : 1;
            size_t bytes = bs * r * c * sizeof(float);
            size_t block = InferShm::alignedOutputBytes(bs, r, c);
            if (shm_->result_batches < InferShm::MAX_BATCHES &&
                offset + block <= InferShm::MAX_OUTPUT_BYTES) {
                // 零拷贝路径下数据应已位于 out+offset；仅作防御性校验，
                // 万一引擎未按约定落位则回退显式拷贝
                if (outputs[i].first->data() != static_cast<void*>(out + offset))
                    std::memcpy(out + offset, outputs[i].first->data<float>(), bytes);
                shm_->batch_size[shm_->result_batches] = (int)bs;
                shm_->out_rows[shm_->result_batches]   = (int)r;
                shm_->out_cols[shm_->result_batches]   = (int)c;
                shm_->result_batches++;
                offset += block;
            } else {
                std::cerr << "[InferShmServer] output overflow, dropping batch" << std::endl;
            }
            i += bs;
        }

        sem_post(resp_sem_);
    }
}

} // namespace Infer
