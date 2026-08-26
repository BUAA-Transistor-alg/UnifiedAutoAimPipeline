#ifndef INFER_SHM_SERVER_H
#define INFER_SHM_SERVER_H

// InferShmServer.h — 推理进程侧实现类
//
// 持有推理器（由外部传入推理函数，推理器本体生命周期由进程 main 管理），
// 循环：等待请求 → 把共享内存中的图像包装为 cv::Mat → 调用推理函数 → 把各
// batch 输出张量拷入共享内存 → 通知客户端。两个推理进程的 main 仅参数不同
// （模型路径/设备/分辨率/max_batch + shm key），共用本类。

#include <vector>
#include <functional>
#include <memory>
#include <opencv2/opencv.hpp>
#include <semaphore.h>

#include "common/Infer/InferShm.h"
#include "common/Infer/InferCore.h"   // InferenceOutput

namespace Infer {

class InferShmServer {
public:
    /// 推理函数：输入已预处理（模型输入尺寸）的 u8 BGR 图像指针，把各 batch 输出
    /// 张量直接写入 out_area（共享内存输出区，按 InferShm::alignedOutputBytes 对齐
    /// 连续存放，容量 out_cap），返回每个图对应的 InferenceOutput（同 batch 共享
    /// 张量）——即推理器 runInference 的零拷贝输出路径。
    using InferFn = std::function<std::vector<InferenceOutput>(
        const std::vector<const cv::Mat*>&, char* out_area, size_t out_cap)>;

    /// @param shm_key  共享内存 Key（与客户端同 Key）
    /// @param infer_fn 推理函数（闭包捕获推理器）
    InferShmServer(int shm_key, InferFn infer_fn);
    ~InferShmServer();

    InferShmServer(const InferShmServer&) = delete;
    InferShmServer& operator=(const InferShmServer&) = delete;

    /// 阻塞运行处理循环，直到 is_running() 返回 false（配合信号处理）
    void run(const std::function<bool()>& is_running);

private:
    int shm_key_;
    int shm_id_ = -1;
    InferShm::ShmLayout* shm_ = nullptr;
    sem_t* req_sem_  = nullptr;
    sem_t* resp_sem_ = nullptr;
    InferFn infer_fn_;

    void createSharedMemory();
    void createSemaphores();
    char* inputArea()  const { return reinterpret_cast<char*>(shm_) + InferShm::inputOffset(); }
    char* outputArea() const { return reinterpret_cast<char*>(shm_) + InferShm::outputOffset(); }
};

} // namespace Infer

#endif // INFER_SHM_SERVER_H
