#ifndef INFER_SHM_CLIENT_H
#define INFER_SHM_CLIENT_H

// InferShmClient.h — 主进程侧推理通信器
//
// 接口与原有推理器（PowerRuneInfer / ArmorInfer）的 runInference 一致，流水线推理
// 阶段可直接替换：把预处理后的图像 memcpy 进共享内存输入区 → 通知推理进程 →
// 等待结果 → 把响应张量 memcpy 进调用方（PipelineData）提供的输出缓冲。
// 推理器本体在独立的推理进程中（InferShmServer）。

#include <vector>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <semaphore.h>

#include "common/Infer/InferShm.h"

namespace Infer {

/// 单图推理输出缓冲：由调用方（流水线数据）持有，客户端把响应张量直接
/// memcpy 进 data（行优先 f32），并写入 rows/cols（张量 shape[1]/shape[2]）。
struct OutputBuffer {
    std::vector<float> data;
    int rows = 0;
    int cols = 0;
};

class InferShmClient {
public:
    /// @param shm_key 共享内存 Key（config 中配置；两进程用同一 Key）
    explicit InferShmClient(int shm_key);
    ~InferShmClient();

    InferShmClient(const InferShmClient&) = delete;
    InferShmClient& operator=(const InferShmClient&) = delete;

    /// 主推理接口：接收已预处理（resize 到模型输入尺寸）的 u8 BGR 图像指针向量，
    /// memcpy 进共享内存输入区；响应张量 memcpy 进 outs（与 imgs 一一对应，
    /// 内容被覆盖）。
    /// @return true 成功；false 通信失败（本批丢弃，outs 内容无效）
    bool runInference(const std::vector<const cv::Mat*>& preprocessed_imgs,
                      std::vector<OutputBuffer*>& outs);

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
