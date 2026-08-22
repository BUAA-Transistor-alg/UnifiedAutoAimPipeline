#ifndef INFER_SHM_CLIENT_H
#define INFER_SHM_CLIENT_H

// InferShmClient.h — 主进程侧推理通信器
//
// 接口与原有推理器（YoloPoseInfer / OutpostInfer）的 runInference 完全一致，
// 流水线推理阶段可直接替换：把预处理后的图像写入共享内存 → 通知推理进程 →
// 等待结果 → 重建输出张量返回。推理器本体在独立的推理进程中（InferShmServer）。

#include <vector>
#include <memory>
#include <mutex>
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

    /// 推理进程（重新）启动后重新打开信号量。
    /// lazy 模式下推理进程按需启停，服务端启动时会 sem_unlink 重建信号量，
    /// 本客户端若先于服务端附加（构造时创建了同名信号量），旧句柄即与新的
    /// 服务端信号量断开，需调用本方法重连（在服务端就绪后调用）。
    void reconnect();

private:
    int  shm_key_;
    int  shm_id_ = -1;
    InferShm::ShmLayout* shm_ = nullptr;   // 附加后的共享内存基址
    sem_t* req_sem_  = nullptr;
    sem_t* resp_sem_ = nullptr;
    // 串行化 runInference 与 reconnect：reconnect 关闭/重开信号量时，
    // 等待在途 runInference 结束（最多 2s 响应超时），避免关闭被阻塞中的信号量。
    std::mutex io_mtx_;

    void attachSharedMemory();
    void openSemaphores();
    char* inputArea()  const { return reinterpret_cast<char*>(shm_) + InferShm::inputOffset(); }
    char* outputArea() const { return reinterpret_cast<char*>(shm_) + InferShm::outputOffset(); }
};

} // namespace Infer

#endif // INFER_SHM_CLIENT_H
