#ifndef INFER_SHM_H
#define INFER_SHM_H

// InferShm.h — 推理进程间通信（共享内存 + 具名信号量）
//
// 背景：进程内"两套模型都编译在 GPU 上"会导致推理退化（2026.3 GPU 插件实测），
// 而跨进程驻留零影响（实测 255fps 全程不变）。因此把推理（仅推理一步）提取到
// 两个独立进程（outpost / power_rune 各一个），主进程只做预处理/后处理，通过
// 本模块通信。共享内存布局与通信协议参考
// transistor_rm2026_algorithm_visual_ws 的 SharedMemoryYOLOPose（System V
// shmget/shmat + 状态标志），这里用具名信号量（sem_open/sem_wait/sem_post）
// 替代忙轮询，阻塞式等待、不占 CPU。
//
// 协议（一次请求 = 一批图像）：
//   client: 写入 batch_count / in_h / in_w + 输入图像区 → sem_post(req)
//   server: sem_wait(req) → 包装图像 → runInference → 写结果区（各 batch 的
//           f32 张量按 64B 对齐块连续存放，batch_size/out_rows/out_cols 写入头）
//           → sem_post(resp)
//   client: sem_wait(resp) → 按 batch 对齐块定位并重建 ov::Tensor → 返回
//
// 输出区布局（服务端与客户端必须一致）：每个 batch 块 = alignedOutputBytes(bs,r,c)
// 字节（64B 对齐），块内 bs 张图按 r*c*4 步长连续存放。64B 对齐是 CPU 插件对用户
// 输出内存走 SIMD 快速路径的常规要求，对 GPU D2H 直写亦无副作用。
//
// 共享内存 Key 由 config 提供（outpost.inference.shm_key / power_rune.inference.shm_key）；
// 信号量名由 Key 派生（/uap_infer_req_<key> / /uap_infer_resp_<key>）。

#include <cstddef>
#include <string>

namespace InferShm {

// 每批最多图像数（共享内存容量上限；config 中各流水线的
// pipeline.inference_batch 须 ≤ 该值，且 ≤ 其 inference.max_batch）
constexpr int    MAX_IMAGES    = 4;
// 响应中最多 batch 数（贪心拆分，最多 MAX_IMAGES 个 batch=1）
constexpr int    MAX_BATCHES   = MAX_IMAGES;
// 输入区上限：4 张 640×640×3（outpost 最大输入）
constexpr size_t MAX_INPUT_BYTES  = (size_t)4 * 640 * 640 * 3;
// 输出区上限：4 个 [1,25200,22] f32（outpost 最大输出，覆盖 power_rune 的 [4,300,102]）
constexpr size_t MAX_OUTPUT_BYTES = (size_t)4 * 25200 * 22 * sizeof(float);

// 共享内存头（packed，其后依次是输入图像区、输出张量区）
#pragma pack(push, 1)
struct ShmLayout {
    // ── 请求（client → server）──
    int batch_count;        // 本次请求图像数（1..MAX_IMAGES）
    int in_h;               // 预处理后图像高度（模型输入 H）
    int in_w;               // 预处理后图像宽度（模型输入 W）
    // ── 响应（server → client）──
    int result_batches;     // 输出 batch 数（贪心拆分后）
    int batch_size[MAX_BATCHES];   // 各 batch 的图像数
    int out_rows[MAX_BATCHES];     // 各 batch 输出张量 shape[1]
    int out_cols[MAX_BATCHES];     // 各 batch 输出张量 shape[2]
};
#pragma pack(pop)

inline size_t inputOffset()  { return sizeof(ShmLayout); }
inline size_t outputOffset() { return inputOffset() + MAX_INPUT_BYTES; }
inline size_t totalSize()    { return outputOffset() + MAX_OUTPUT_BYTES; }

/// 单个 batch 输出块在输出区占用的字节数（64B 对齐，见文件头注释）。
/// 引擎写入、服务端记录、客户端定位三处必须使用同一规则。
inline size_t alignedOutputBytes(size_t bs, size_t r, size_t c) {
    size_t raw = bs * r * c * sizeof(float);
    return (raw + 63) & ~static_cast<size_t>(63);
}

inline std::string reqSemName(int key)  { return "/uap_infer_req_"  + std::to_string(key); }
inline std::string respSemName(int key) { return "/uap_infer_resp_" + std::to_string(key); }

} // namespace InferShm

#endif // INFER_SHM_H
