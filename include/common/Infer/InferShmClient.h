#ifndef INFER_SHM_CLIENT_H
#define INFER_SHM_CLIENT_H

// InferShmClient.h — 主进程侧推理通信器
//
// 接口与原有推理器（YoloPoseInfer / OutpostInfer）的 runInference 完全一致，
// 流水线推理阶段可直接替换：把预处理后的图像写入共享内存 → 通知推理进程 →
// 等待结果 → 重建输出张量返回。推理器本体在独立的推理进程中（InferShmServer）。

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <semaphore.h>

#include "common/Infer/InferShm.h"
#include "common/Infer/InferCore.h"   // InferenceOutput

namespace Infer {

class InferShmClient {
public:
    /// @param shm_key 共享内存 Key（config 中配置；两进程用同一 Key）
    explicit InferShmClient(int shm_key);
    ~InferShmClient();

    InferShmClient(const InferShmClient&) = delete;
    InferShmClient& operator=(const InferShmClient&) = delete;

    /// 主推理接口：与推理器同签名。接收已预处理（resize 到模型输入尺寸）的
    /// u8 BGR 图像指针向量，经共享内存发送到推理进程，返回每个输入图对应的
    /// InferenceOutput（同 batch 共享同一张量，张量由本客户端持有内存）。
    std::vector<InferenceOutput> runInference(
        const std::vector<const cv::Mat*>& preprocessed_imgs);

private:
    int  shm_key_;
    int  shm_id_ = -1;
    InferShm::ShmLayout* shm_ = nullptr;   // 附加后的共享内存基址
    sem_t* req_sem_  = nullptr;
    sem_t* resp_sem_ = nullptr;

    void attachSharedMemory();
    void openSemaphores();
    char* inputArea()  const { return reinterpret_cast<char*>(shm_) + InferShm::inputOffset(); }
    char* outputArea() const { return reinterpret_cast<char*>(shm_) + InferShm::outputOffset(); }
};

} // namespace Infer

#endif // INFER_SHM_CLIENT_H
