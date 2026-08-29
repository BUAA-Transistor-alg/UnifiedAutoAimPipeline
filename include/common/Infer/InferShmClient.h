#ifndef INFER_SHM_CLIENT_H
#define INFER_SHM_CLIENT_H

// InferShmClient.h — 主进程侧推理通信器
//
// 接口与原有推理器（PowerRuneInfer / ArmorInfer）的 runInference 一致，流水线推理
// 阶段可直接替换：把预处理后的图像 memcpy 进共享内存输入区 → 通知推理进程 →
// 等待结果 → 把响应张量 memcpy 进调用方（PipelineData）提供的输出缓冲。
// 推理器本体在独立的推理进程中（InferShmServer）。
//
// 推理挂死自愈（config common.infer_force_restart_timeout_sec）：
//   - 每次推理正常返回时记录时间戳 last_ok_（推理正常返回不检测挂死）；
//   - 每次推理响应超时（2s sem_timedwait）返回时，检测 now - last_ok_ 是否超过
//     配置时限：超过则调用强制重启回调（主程序挂钩到 InferProcessManager::
//     forceRestart，重启本客户端对应的推理进程），并把 last_ok_ 视为已失效，
//     直至推理进程（重新）就绪（reconnect）或再次正常返回后才重新计时；
//   - 同一客户端两次强制重启至少间隔配置时限（节流，配合管理器的去重，避免
//     重启期间反复触发）。

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <vector>
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
    /// @param force_restart_timeout_sec 推理挂死强制重启时限（秒，config
    ///        common.infer_force_restart_timeout_sec）：推理响应超时返回时，若距
    ///        上一次推理正常返回的间隔超过该时限，则触发强制重启回调。
    ///        <= 0 表示关闭该功能。
    explicit InferShmClient(int shm_key, double force_restart_timeout_sec);
    ~InferShmClient();

    InferShmClient(const InferShmClient&) = delete;
    InferShmClient& operator=(const InferShmClient&) = delete;

    /// 主推理接口：接收已预处理（resize 到模型输入尺寸）的 u8 BGR 图像指针向量，
    /// memcpy 进共享内存输入区；响应张量 memcpy 进 outs（与 imgs 一一对应，
    /// 内容被覆盖）。
    /// @return true 成功；false 通信失败（本批丢弃，outs 内容无效）
    bool runInference(const std::vector<const cv::Mat*>& preprocessed_imgs,
                      std::vector<OutputBuffer*>& outs);

    /// 设置推理挂死强制重启回调（主程序把本客户端对应流水线的推理进程强制重启
    /// 挂钩到 InferProcessManager::forceRestart）。可在任意线程调用；满足挂死条件
    /// 时从推理阶段线程调用该回调。
    void setForceRestartHandler(std::function<void()> handler);

    /// 推理进程（重新）启动后重新打开信号量。
    /// lazy 模式下推理进程按需启停，服务端启动时会 sem_unlink 重建信号量，
    /// 本客户端若先于服务端附加（构造时创建了同名信号量），旧句柄即与新的
    /// 服务端信号量断开，需调用本方法重连（在服务端就绪后调用）。
    /// 重连同时把"上次推理正常返回"重置为当前时刻（推理进程已（重新）就绪，
    /// 挂死检测从此刻起重新计时）。
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

    // ── 推理挂死检测（见文件头注释）──
    double force_restart_timeout_sec_ = 0.0;   // 时限（秒；<= 0 关闭）
    std::mutex handler_mtx_;                    // 保护 restart_handler_
    std::function<void()> restart_handler_;
    // 以下原子量由 runInference（推理阶段线程）与 reconnect（后台线程）并发访问
    std::atomic<std::chrono::steady_clock::time_point> last_ok_{};    // 上次推理正常返回时间
    std::atomic<bool> last_ok_valid_{false};                          // last_ok_ 是否已建立
    std::atomic<std::chrono::steady_clock::time_point> last_restart_{};  // 上次触发强制重启时间

    void attachSharedMemory();
    void openSemaphores();
    /// 推理响应超时返回时调用：距上次正常返回超过配置时限则触发强制重启回调
    /// （正常返回不进入本函数，不计算间隔）
    void maybeTriggerForceRestart();
    char* inputArea()  const { return reinterpret_cast<char*>(shm_) + InferShm::inputOffset(); }
    char* outputArea() const { return reinterpret_cast<char*>(shm_) + InferShm::outputOffset(); }
};

} // namespace Infer

#endif // INFER_SHM_CLIENT_H
