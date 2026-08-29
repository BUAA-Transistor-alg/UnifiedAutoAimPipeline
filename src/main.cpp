// main.cpp — 统一自瞄流水线主程序（Armor / PowerRune 双流水线 + 输入/输出模式）
//
// 用法：
//   unified_auto_aim [--pipeline armor|power_rune]
//                    [--input camera|video|interactive]
//                    [--output none|visualize|gimbal]
//                    [-v <video>] [-e <extra_info.txt>] [--record] [--skip-unaccepted] [-i] [-h]
//
// 设计要点：
//  - 两个流水线本体在启动时全部构造、且始终保持构造（阶段线程/调度器常驻）；推理器
//    （模型编译产物，占用 GPU 显存）提取到两个独立进程（armor_infer_process /
//    power_rune_infer_process），各进程只编译一个模型——2026.3 GPU 插件实测跨进程
//    驻留零影响，而进程内两套模型共存会互相拖慢（详见 InferShm.h 背景说明）；
//  - 流水线推理阶段经共享内存 + 具名信号量与推理进程通信（InferShmClient），预处理/
//    后处理仍在主进程；推理进程未启动时推理阶段阻塞等待（协议要求先启动推理进程）；
//  - 推理进程启动策略由 config common.infer_process_lazy 决定：
//      false（默认）：由 launch_all.py 启动全部推理进程并常驻，无任务时后台闲置；
//      true：由本进程按需启动/停止（InferProcessManager）——仅启动当前流水线所需
//            推理进程，切换流水线时立即关闭不再需要的进程（释放 GPU 显存）；
//  - 任意时刻只有一个是"激活"状态（接收帧），运行时可通过 API（switchPipeline）
//    或热键 '1'/'2' 切换；切换只交换指针（推理进程按上述策略常驻或按需切换）；
//  - 输入模式（相机/视频/交互）与输出模式（无/可视化/云台控制）在启动时
//    指定，运行时也可通过 API（switchPipeline / toggleOutput）或热键 '1'/'2'、'v'/'g' 切换；
//  - RobotController（串口 + MPC）仅在需要时构造（相机输入或云台输出）。
#include "common/Input/IInputMode.h"
#include "common/Input/CameraInputMode.h"
#include "common/Input/VideoInputMode.h"
#include "common/Input/InteractiveInputMode.h"
#include "common/BacklogAdaptiveDelay.h"
#include "common/IPipeline.h"
#include "Armor/ArmorPipeline.h"
#include "PowerRune/PowerRunePipeline.h"
#include "common/Output/IOutputMode.h"
#include "common/Output/VisualizeOutput.h"
#include "common/Output/GimbalOutput.h"
#include "common/Ballistic/SequencePredictor.h"
#include "common/LatestSlot.h"
#include "common/RobotConfig.h"
#include "common/FrameRateCounter.h"
#include "common/Record/FrameRecorder.h"
#include "common/Infer/InferProcessManager.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <functional>
#include <memory>
#include <thread>
#include <atomic>
#include <csignal>

// ── backward-cpp 崩溃堆栈回溯 ──
#include "backward.hpp"
namespace {
backward::SignalHandling sh;  // 自动捕获 SIGSEGV/SIGABRT 等信号并打印堆栈
}

// ==================== 命令行选项 ====================

enum class InputKind { CAMERA, VIDEO, INTERACTIVE };

/// 输出模式组合（visualize 与 gimbal 独立开关，可同时启用）
struct OutputConfig {
    bool visualize = false;
    bool gimbal = false;
};

struct Options {
    PipelineMode pipeline = PipelineMode::ARMOR;
    InputKind    input    = InputKind::INTERACTIVE;
    OutputConfig output;    // 默认 visualize
    std::string  video_path;
    std::string  extra_info_path;
    bool         record = false;            // 开启录制（帧 + 时间戳 + extra_info）
    bool         skip_unaccepted = false;   // 回放时跳过未成功加入流水线的帧（v2 extra info）

    Options() { output.visualize = true; }
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --pipeline armor|power_rune  流水线模式 (默认 armor)\n"
              << "  --input camera|video|interactive  输入模式 (默认 interactive)\n"
              << "  --output <mode>[+<mode>...]     输出模式组合 (默认 visualize)\n"
              << "            none | visualize | gimbal | visualize+gimbal\n"
              << "  -v, --video <file>              视频路径 (input=video 时必需)\n"
              << "  -e, --extra-info <file>         视频额外信息 txt (相机位姿 / 完整 ExtraInputInfo)\n"
              << "  -i, --interactive               交互式图片输入 (等价 --input interactive)\n"
              << "  -r, --record                    开启录制：每帧保存为 MKV 视频 + 时间戳 + 完整\n"
              << "                                   extra_info txt（输出目录与剩余空间阈值见\n"
              << "                                   机器配置文件（config/robots/*.yaml）的 common.recording）\n"
              << "      --skip-unaccepted           回放视频时跳过录制中未成功加入流水线的帧\n"
              << "                                   （需 -e 指定 v2 格式的 extra info 文件）\n"
              << "  -h, --help                      显示帮助\n"
              << "无参数运行默认显示本帮助。\n"
              << "窗口行为:\n"
              << "  - 启动时未指定 visualize：不显示窗口、不绘制任何可视化，纯后台运行\n"
              << "  - 启动时指定 visualize：可视化开启时绘制检测/位姿/滤波 + 统一覆盖层\n"
              << "    (串口信息 / 帧数统计 / 热键提醒，两流水线一致)；按 'v' 关闭可视化后\n"
              << "    窗口仅显示原始画面，热键仍可用，'v' 可随时恢复\n"
              << "  - Armor 模式且可视化开启时额外打开 'Armor XY Plane' 顶视图窗口\n"
              << "    (车体中心 / 四块装甲板 t+0 预测 / 瞄准目标 / 自身底盘 + 连线)，\n"
              << "    切换出 Armor 模式或关闭可视化时自动关闭\n"
              << "热键 (窗口内):\n"
              << "  '1'/'2' 切换流水线  'v' 开关可视化  'g' 开关云台输出  'n' 关闭全部输出  'q'/ESC 退出\n";
}

static PipelineMode parsePipeline(const std::string& s) {
    if (s == "armor") return PipelineMode::ARMOR;
    if (s == "power_rune") return PipelineMode::POWER_RUNE;
    throw std::runtime_error("未知流水线模式: " + s + " (可选 armor / power_rune)");
}

static InputKind parseInput(const std::string& s) {
    if (s == "camera") return InputKind::CAMERA;
    if (s == "video") return InputKind::VIDEO;
    if (s == "interactive") return InputKind::INTERACTIVE;
    throw std::runtime_error("未知输入模式: " + s + " (可选 camera / video / interactive)");
}

/// 解析输出模式组合："none" 全关；支持 "+" 或 "," 分隔多个模式
static OutputConfig parseOutput(const std::string& s) {
    OutputConfig cfg;
    auto flush = [&](const std::string& tok) {
        if (tok.empty()) return;
        if (tok == "visualize") cfg.visualize = true;
        else if (tok == "gimbal") cfg.gimbal = true;
        else if (tok == "none") { cfg.visualize = false; cfg.gimbal = false; }
        else throw std::runtime_error("未知输出模式: " + tok +
                                      " (可选 none / visualize / gimbal / visualize+gimbal)");
    };
    std::string cur;
    for (char c : s) {
        if (c == '+' || c == ',') { flush(cur); cur.clear(); }
        else cur += c;
    }
    flush(cur);
    return cfg;
}

static Options parseArgs(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("参数 " + arg + " 缺少值");
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            exit(0);
        } else if (arg == "--pipeline") {
            opt.pipeline = parsePipeline(next());
        } else if (arg == "--input") {
            opt.input = parseInput(next());
        } else if (arg == "--output") {
            opt.output = parseOutput(next());
        } else if (arg == "-v" || arg == "--video") {
            opt.video_path = next();
        } else if (arg == "-e" || arg == "--extra-info") {
            opt.extra_info_path = next();
        } else if (arg == "-r" || arg == "--record") {
            opt.record = true;
        } else if (arg == "--skip-unaccepted") {
            opt.skip_unaccepted = true;
        } else if (arg == "-i" || arg == "--interactive") {
            opt.input = InputKind::INTERACTIVE;
        } else {
            throw std::runtime_error("未知参数: " + arg);
        }
    }
    if (opt.input == InputKind::VIDEO && opt.video_path.empty()) {
        throw std::runtime_error("input=video 时必须指定 -v <视频路径>");
    }
    return opt;
}

// ==================== 可视化统一覆盖层 ====================
// 可视化开启时，无论 Armor 还是 PowerRune 流水线，都在画面上叠加：
//   1. 热键提醒  — 顶部 (10,20)，0.6 号粗 2 绿
//   2. 队列积压  — 原 main info 位置 (10,60)，Q[in i0 i1 i2 i3 out] 格式
//   3. 帧数统计  — 原 drawFps 样式：右上角
//      "Pipeline: xx  Ballistic: xx  Gimbal: xx  Visual: xx  MPC: xx" 0.7 粗 2 黄
//      （无统计/不可用时对应项显示 N/A）+ "TS: .." 0.45 粗 1 黄
//      Pipeline=流水线输出提取帧率；Ballistic=弹道解算循环线程帧率；
//      Gimbal=云台输出循环线程帧率（未开启时 N/A）；
//      Visual=可视化绘制循环线程帧率（未开启时 N/A）；
//      MPC=McuMpcController 后台循环帧率（无 RobotController 时 N/A）
//   4. 串口输入信息 — 原 drawCommInfo 样式：(8,80) 起 0.45 粗 1 绿，按来源分块显示
//      （--- MCU --- / --- IMU --- / --- FUSED --- / --- STRICT --- / --- MPC ---），
//      MCU 块含温度行（按温度区间变色）

// 帧率显示：无统计（fps <= 0，尚无帧或不可用）时显示 N/A
static std::string fpsText(double fps) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    if (fps > 0.0) {
        oss << fps;
    } else {
        oss << "N/A";
    }
    return oss.str();
}

static void drawOverlay(cv::Mat& img,
                        const std::string& pipeline_name,
                        const std::string& output_names,
                        double pipeline_fps,
                        double ballistic_fps,
                        double gimbal_fps,
                        double visualize_fps,
                        RobotController* rc,
                        const std::chrono::steady_clock::time_point& frame_ts,
                        const PipelineResult::QueueSizes& queue_sizes,
                        double extra_delay_s) {
    // 提前获取 RobotController 状态：帧率行需显示 MPC 后台循环帧率（无统计时为 N/A），
    // 第 4 块串口信息区同样使用该状态（此处统一获取一次，避免重复加锁）
    const RobotController::State st =
        (rc != nullptr) ? rc->getState() : RobotController::State{};

    // 1. 热键提醒（顶部）
    cv::putText(img, "Keys: 1/2 pipeline | v visualize | g gimbal | n none | q quit",
                cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

    // 2. 当前流水线各缓存积压长度（与原来 main info 的 Q[...] 格式一致）；
    //    行尾追加队列积压自适应额外延迟 extra_delay_s（BacklogAdaptiveDelay，
    //    功能关闭时为 0，单位秒）
    char qbuf[192];
    std::snprintf(qbuf, sizeof(qbuf), "Q[in:%d i0:%d i1:%d i2:%d i3:%d out:%d] extra:%6.3fs",
                  queue_sizes.input, queue_sizes.inter0, queue_sizes.inter1,
                  queue_sizes.inter2, queue_sizes.inter3, queue_sizes.output,
                  extra_delay_s);
    cv::putText(img, qbuf, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2);

    // 3. 帧数统计（原 drawFps 样式，右上角）
    using namespace std::chrono;
    std::ostringstream oss;
    // Pipeline=流水线输出提取帧率（原 FPS 统计，已重命名）；
    // Ballistic=弹道解算循环线程帧率；Gimbal=云台输出循环线程帧率；
    // Visual=可视化绘制循环线程帧率；MPC=McuMpcController 后台循环帧率。
    // 各帧率无统计/不可用时显示 N/A。
    oss << "Pipeline: " << fpsText(pipeline_fps)
        << "  Ballistic: " << fpsText(ballistic_fps)
        << "  Gimbal: " << fpsText(gimbal_fps)
        << "  Visual: " << fpsText(visualize_fps)
        << "  MPC: " << fpsText(st.mpc.loop_fps);
    int baseline = 0;
    cv::Size sz = cv::getTextSize(oss.str(), cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
    cv::putText(img, oss.str(), cv::Point(img.cols - sz.width - 10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    // 时间戳 + 流水线名 + 输出模式（TS 行附加信息）
    auto epoch = duration_cast<duration<double>>(frame_ts.time_since_epoch()).count();
    oss.str("");
    oss << std::fixed << std::setprecision(3) << "TS: " << epoch
        << "  |  " << pipeline_name << "  |  out: " << output_names;
    sz = cv::getTextSize(oss.str(), cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
    cv::putText(img, oss.str(), cv::Point(img.cols - sz.width - 10, 52),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);

    // 4. 串口输入信息（原 drawCommInfo 样式）
    int y = 80;
    const int lineH = 18;
    auto put = [&](const std::string& t, cv::Scalar color = cv::Scalar(0, 255, 0)) {
        cv::putText(img, t, cv::Point(8, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1);
        y += lineH;
    };
    if (rc == nullptr) {
        put("Serial: N/A (RobotController not constructed)");
        return;
    }

    // ── MCU ──
    put("--- MCU ---");
    if (st.mcu.valid) {
        oss.str("");
        oss << std::fixed << std::setprecision(3);
        oss << "bullet: " << st.mcu.bullet_velocity
            << "  pitch: " << st.mcu.pitch_angle
            << "  yaw: " << st.mcu.yaw_angle;
        put(oss.str());
        oss.str(""); oss << "yaw_omega: " << st.mcu.yaw_omega
                         << "  imu_yaw: " << st.mcu.chassis_imu_yaw
                         << "  imu_omega: " << st.mcu.chassis_imu_omega;
        put(oss.str());
        oss.str(""); oss << "mark: " << (int)st.mcu.mark
                         << "  color: " << (int)st.mcu.color
                         << "  aim: " << (int)st.mcu.auto_aim_switch;
        put(oss.str());
        // 温度行（按温度区间变色：>=70 红 / >=50 黄 / 其余绿）
        cv::Scalar temp_color(0, 255, 0);
        if (st.mcu.yaw_temperature >= 70)      temp_color = cv::Scalar(0, 0, 255);
        else if (st.mcu.yaw_temperature >= 50) temp_color = cv::Scalar(0, 255, 255);
        oss.str(""); oss << "temp: " << (int)st.mcu.yaw_temperature;
        put(oss.str(), temp_color);
    } else {
        put("(no data)");
    }

    // ── IMU ──
    put("--- IMU ---");
    if (st.imu.valid) {
        oss.str(""); oss << std::fixed << std::setprecision(4);
        oss << "gyro: " << st.imu.gx << " " << st.imu.gy << " " << st.imu.gz;
        put(oss.str());
        oss.str(""); oss << "acc: " << st.imu.ax << " " << st.imu.ay << " " << st.imu.az;
        put(oss.str());
        oss.str(""); oss << "euler: " << st.imu.euler_yaw
                         << " " << st.imu.euler_pitch
                         << " " << st.imu.euler_roll;
        put(oss.str());
        oss.str(""); oss << "dt: " << st.imu.dt_one_tenth_ms;
        put(oss.str());
    } else {
        put("(no data)");
    }

    // ── FUSED ──
    put("--- FUSED ---");
    if (st.fused.valid) {
        oss.str(""); oss << std::fixed << std::setprecision(4);
        oss << "yaw_pos: " << st.fused.yaw_pos
            << "  yaw_rate: " << st.fused.yaw_rate;
        put(oss.str());
        oss.str(""); oss << "chassis_yaw: " << st.fused.chassis_yaw
                         << "  pitch: " << st.fused.chassis_pitch
                         << "  roll: " << st.fused.chassis_roll;
        put(oss.str());
    } else {
        put("(no data)");
    }

    // ── STRICT ──
    put("--- STRICT ---");
    {
        oss.str(""); oss << std::fixed << std::setprecision(4);
        oss << "imu_euler: " << st.strict.imu_euler_yaw
            << " " << st.strict.imu_euler_pitch
            << " " << st.strict.imu_euler_roll;
        put(oss.str());
        oss.str(""); oss << "yaw_pos: " << st.strict.yaw_pos
                         << "  pitch: " << st.strict.pitch_angle;
        put(oss.str());
        oss.str(""); oss << "chassis: " << st.strict.chassis_yaw
                         << " " << st.strict.chassis_pitch
                         << " " << st.strict.chassis_roll;
        put(oss.str());
    }

    // ── MPC ──
    put("--- MPC ---");
    {
        oss.str(""); oss << std::fixed << std::setprecision(4);
        oss << "target_yaw: " << st.mpc.yaw_target_angle
            << "  target_vel: " << st.mpc.yaw_target_velocity
            << "  torque: " << st.mpc.yaw_torque;
        put(oss.str());
        oss.str(""); oss << "delayed: " << st.mpc.delayed_target
                         << "  integral: " << st.mpc.integral;
        put(oss.str());
    }
}

// ==================== 主程序 ====================

static volatile bool g_running = true;

void signalHandler(int) {
    g_running = false;
}

int main(int argc, char** argv) {
    signal(SIGINT, signalHandler);

    // ── 无参数运行：显示帮助 ──
    if (argc <= 1) {
        printUsage(argv[0]);
        return 0;
    }

    Options opt;
    try {
        opt = parseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Unified Auto-Aim Pipeline" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Pipeline: " << (opt.pipeline == PipelineMode::ARMOR ? "Armor" : "PowerRune") << std::endl;
    std::cout << "  Input:    " << (opt.input == InputKind::CAMERA ? "Camera"
                                   : opt.input == InputKind::VIDEO ? "Video" : "Interactive") << std::endl;
    std::cout << "  Output:   "
              << (opt.output.visualize ? "Visualize " : "")
              << (opt.output.gimbal ? "Gimbal " : "")
              << (!opt.output.visualize && !opt.output.gimbal ? "None" : "") << std::endl;
    std::cout << "  Record:   " << (opt.record ? "ON" : "OFF")
              << (opt.skip_unaccepted ? "  (skip-unaccepted: ON)" : "") << std::endl;
    std::cout << "========================================" << std::endl;

    const RobotConfig& cfg = RobotConfig::instance();

    // ── 队列积压自适应额外延迟（可选功能，config common.backlog_adaptive_delay）──
    // 开启后：处理线程每次 tryPopFrame() 后统计除输出缓冲队列外各缓冲队列积压
    // 总数，以 target_backlog 为目标做 PI 直接输出（比例 + 积分、抗饱和，
    // 无 EMA、无速率上限；v6，见 BacklogAdaptiveDelay，稳态无极限环震荡）。
    // 全部可调参数均来自 config，此处仅组装并调用其接口。
    const RobotConfig::CommonParams::BacklogAdaptiveDelayParams& bad_cfg =
        cfg.common.backlogAdaptiveDelay;
    BacklogAdaptiveDelay backlog_delay(BacklogAdaptiveDelay::Config{
        bad_cfg.enabled, bad_cfg.targetBacklog, bad_cfg.maxExtraDelaySeconds,
        bad_cfg.gainP, bad_cfg.gainI});
    if (backlog_delay.enabled()) {
        std::ostringstream oss_bad;
        oss_bad << std::fixed << std::setprecision(6)
                << "[main] Backlog adaptive delay: ON (target "
                << bad_cfg.targetBacklog << ", max "
                << bad_cfg.maxExtraDelaySeconds << " s, gain_p "
                << bad_cfg.gainP << " s/unit, gain_i "
                << bad_cfg.gainI << " s/unit/s)";
        std::cout << oss_bad.str() << std::endl;
    }

    // 相机参数：按输入模式自动选择（camera → input_mode.camera_mode；video/interactive → input_mode.video_mode）
    const RobotConfig::CameraParams& camera_params =
        (opt.input == InputKind::CAMERA) ? cfg.common.inputMode.cameraMode : cfg.common.inputMode.videoMode;

    // ── RobotController（仅在需要时构造：相机输入需要 strict 数据，云台输出需要控制）──
    // 相机输入模式下 CameraInputMode 后台线程持续采样 rc_.getState()（延迟队列），
    // 因此相机输入需要 RobotController（无硬件时串口线程静默失败）。
    // 线程化后构造可能发生在窗口线程（运行时开启云台），读取发生在处理/弹道/
    // 窗口线程，用 rc_mtx 保护指针的按需构造与读取。
    std::mutex rc_mtx;
    std::unique_ptr<RobotController> robot_controller;
    auto ensureRobotController = [&]() {
        std::lock_guard<std::mutex> lock(rc_mtx);
        if (robot_controller) return;
        const auto& rp = cfg.common.robotController;
        robot_controller = std::make_unique<RobotController>(
            rp.dtControl, rp.mpcPredN, rp.J, rp.tauC, rp.b, rp.tauD,
            rp.maxTorque, rp.maxTorqueRate, rp.Q, rp.R, rp.Rd, rp.maxIter,
            rp.integralGain,
            /*mpc_loop_period=*/rp.dtControl,   // MPC 后台循环周期与 dt_control 相同
            McuDataPreprocessor::LinearParams{
                rp.sendPitchScale, rp.sendPitchOffset,
                rp.recvPitchScale, rp.recvPitchOffset},
            rp.sequenceMode);
        std::cout << "[main] RobotController constructed (serial threads may fail silently without hardware)." << std::endl;
    };
    // 线程安全读取当前 RobotController 指针（未构造时为 nullptr）
    auto robotControllerPtr = [&]() -> RobotController* {
        std::lock_guard<std::mutex> lock(rc_mtx);
        return robot_controller.get();
    };
    if (opt.input == InputKind::CAMERA || opt.output.gimbal) {
        ensureRobotController();
    }

    // ── 输入模式 ──
    std::unique_ptr<IInputMode> input_mode;
    std::unique_ptr<Camera> camera;
    switch (opt.input) {
        case InputKind::CAMERA: {
            camera = std::make_unique<Camera>(camera_params.deviceIp, camera_params.netIp);
            camera->setExposureTime(camera_params.exposure);
            camera->setGain(camera_params.gain);
            if (!camera->start()) {
                std::cerr << "Camera start failed!" << std::endl;
                return -1;
            }
            input_mode = std::make_unique<CameraInputMode>(*camera, *robot_controller);
            break;
        }
        case InputKind::VIDEO:
            input_mode = std::make_unique<VideoInputMode>(opt.video_path, opt.extra_info_path,
                                                          opt.skip_unaccepted,
                                                          cfg.common.inputMode.videoMode.testMaxFps);
            break;
        case InputKind::INTERACTIVE:
        default:
            input_mode = std::make_unique<InteractiveInputMode>();
            break;
    }

    // ── 录制器（--record 开启）：帧 + 时间戳 + extra_info 完整保存 ──
    // 输出目录与剩余空间阈值来自 config/common.recording（路径已加入 .gitignore）
    std::unique_ptr<FrameRecorder> recorder;
    if (opt.record) {
        FrameRecorder::Config rcfg;
        rcfg.outputDir = cfg.common.recording.outputDir;
        rcfg.minFreeSpaceBytes = cfg.common.recording.minFreeSpaceBytes;
        recorder = std::make_unique<FrameRecorder>(rcfg);
        if (!recorder->start()) {
            std::cerr << "[main] 录制启动失败，录制已禁用。" << std::endl;
            recorder.reset();
        }
    }

    // ── 流水线生命周期 ──
    // 两条流水线始终构造、始终持有各自的推理通信器（InferShmClient）。推理器本体
    // （模型编译产物，占 GPU）提取到两个独立进程（armor_infer_process /
    // power_rune_infer_process，各自只编译一个模型——2026.3 GPU 插件实测跨进程驻留
    // 零影响，而进程内两套模型共存会互相拖慢）。因此本进程不做 GPU 推理，切换模式
    // 只交换指针；推理进程的启停策略见 config common.infer_process_lazy：
    //   false（默认）：由 launch_all.py 启动全部推理进程并常驻（先于主程序启动）；
    //   true：由下方 InferProcessManager 按需启停（仅启动当前流水线所需进程）。
    // min_delay_seconds 共用 config common.min_delay_seconds；相机参数按输入模式选择；
    // 各流水线缓冲队列长度取自 config armor.pipeline / power_rune.pipeline
    ArmorPipeline   armor_pipeline(cfg.armor.pipeline.queueMaxSizes,
                                       cfg.common.minDelaySeconds, camera_params);
    PowerRunePipeline power_rune_pipeline(cfg.powerRune.pipeline.queueMaxSizes,
                                          cfg.common.minDelaySeconds, camera_params);
    IPipeline* active_pipeline =
        (opt.pipeline == PipelineMode::ARMOR) ? static_cast<IPipeline*>(&armor_pipeline)
                                                : static_cast<IPipeline*>(&power_rune_pipeline);

    // ── 推理进程管理器（始终构造）──
    // 主程序始终持有 InferProcessManager：
    //   - lazy 模式（common.infer_process_lazy=true）：按需启停推理进程——仅启动
    //     当前流水线所需进程，切换流水线时立即关闭不需要的进程（释放 GPU 显存），
    //     再启动新流水线所需进程；推理进程就绪后需重连该流水线的 InferShmClient
    //     （客户端先于服务端附加，服务端 sem_unlink 重建信号量后旧句柄失效）。
    //   - 非 lazy 模式（默认）：推理进程由 launch_all.py 启动并常驻，本管理器不
    //     主动启停；仅当推理客户端检测到推理进程挂死时（响应超时距上次正常返回
    //     超过 common.infer_force_restart_timeout_sec）经 forceRestart 强制重启。
    // 用 shared_ptr 持有，并让两条流水线的强制重启回调捕获其副本：管理器因此存活
    // 于任何可能调用它的推理阶段线程之后（回调随客户端析构才释放引用）。
    auto reconnectFor = [&](Infer::InferProcessManager::Kind k) {
        if (k == Infer::InferProcessManager::Kind::ARMOR)
            armor_pipeline.reconnectInferClient();
        else
            power_rune_pipeline.reconnectInferClient();
    };
    auto infer_manager = std::make_shared<Infer::InferProcessManager>(reconnectFor);

    // 推理客户端 → 管理器：推理进程疑似挂死时强制重启对应进程（两条流水线各一个
    // 客户端；强制重启对非 lazy 模式同样生效——按进程名终止外部进程后自启）
    armor_pipeline.setInferRestartHandler([mgr = infer_manager]() {
        mgr->forceRestart(Infer::InferProcessManager::Kind::ARMOR);
    });
    power_rune_pipeline.setInferRestartHandler([mgr = infer_manager]() {
        mgr->forceRestart(Infer::InferProcessManager::Kind::POWER_RUNE);
    });

    // lazy 模式：初始启动当前流水线所需推理进程（仅 lazy 模式使用管理器管理进程）
    if (cfg.common.inferProcessLazy) {
        const Infer::InferProcessManager::Kind initial_kind =
            (opt.pipeline == PipelineMode::ARMOR)
                ? Infer::InferProcessManager::Kind::ARMOR
                : Infer::InferProcessManager::Kind::POWER_RUNE;
        std::cout << "[main] lazy 模式（infer_process_lazy=true）：仅启动当前流水线所需推理进程"
                  << "，切换时立即关闭不再需要的进程。" << std::endl;
        if (!infer_manager->startAndWait(initial_kind)) {
            std::cerr << "[main] 启动初始推理进程失败（lazy 模式），退出。"
                      << "请检查模型路径/config 后重试。" << std::endl;
            return 1;
        }
    }

    // ── 共享状态与输出模式注册表 ──
    // 处理阶段线程化后，以下共享状态由多个线程访问，用互斥锁保护：
    //   pipeline_mtx —— active_pipeline（输入/处理线程读取，窗口线程切换时写）
    //   rc_mtx       —— robot_controller 的按需构造与读取
    //   output_mtx   —— output_modes（弹道线程读快照，窗口线程切换时写）
    std::mutex pipeline_mtx;
    std::mutex output_mtx;
    std::vector<std::shared_ptr<IOutputMode>> output_modes;   // shared_ptr：输出请求跨线程持有对象
    // 预测瞄准点通用类：任何情况下（无论输出模式）每帧调用，生成预测云台控制
    // 序列 + 瞄准点序列；GimbalOutput 消费控制序列，VisualizeOutput 消费首个瞄准点
    SequencePredictor sequence_predictor;
    // 相机投影（与输入模式绑定，两个流水线/可视化共用）
    auto camera_proj = std::make_shared<CameraProjection>(
        camera_params.cameraMatrix, camera_params.distCoeffs,
        ImageResolution{camera_params.width, camera_params.height});

    // ── 各处理阶段：缓冲位（LatestSlot，单槽最新覆盖）+ 循环线程（数据流顺序见下）──
    // 数据流：input_thread → 流水线 → process_thread（只填充弹道缓冲位）
    //   → ballistic_thread（弹道解算；给每一个输出模式的缓冲位填入）
    //   → gimbal_thread / visualize_thread（各自的输出循环线程，
    //      模式首次开启时创建，随后不销毁）
    // 各处理阶段所需时间戳直接取最新 shared_frame_timestamp（见各线程体）。
    struct BallisticRequest {
        RobotController::State st;      // 弹道解算所需云台/串口状态快照
        RobotController* rc = nullptr;  // 转发给可视化线程
        std::unique_ptr<PipelineResult> result;  // 流水线结果（移动转发，含 predictor 快照）
    };
    struct GimbalRequest {
        std::shared_ptr<GimbalOutput> gimbal;  // 当前云台输出（模式关闭时为空 → 线程跳过）
    };
    struct VisualizeRequest {
        std::shared_ptr<VisualizeOutput> vis;  // 当前可视化输出（模式关闭时为空 → 仅更新原始帧）
        std::unique_ptr<PipelineResult> result;
        RobotController* rc = nullptr;
    };
    // 输出模式阶段：输入缓冲 + 循环线程（首次开启时创建，随后不销毁）+ 循环帧率
    struct GimbalStage {
        LatestSlot<GimbalRequest> slot;
        std::unique_ptr<std::thread> thread;
        std::atomic<double> fps{0.0};
    };
    struct VisualizeStage {
        LatestSlot<VisualizeRequest> slot;
        std::unique_ptr<std::thread> thread;
        std::atomic<double> fps{0.0};
    };
    LatestSlot<BallisticRequest> ballistic_slot;
    GimbalStage     gimbal_stage;
    VisualizeStage  visualize_stage;
    std::atomic<double> pipeline_fps{0.0};   // 流水线输出提取帧率（原 FPS 统计，已重命名）
    std::atomic<double> ballistic_fps{0.0};  // 弹道解算循环线程帧率

    // 阶段回调（std::function：线程体与切换回调互相引用，先声明后赋值）
    std::function<void(PipelineMode)> switchPipeline;
    std::function<void(OutputMode)>  toggleOutput;
    std::function<void()>            ensureGimbalThread;
    std::function<void()>            ensureVisualizeThread;

    // 输出模式查询（线程安全：加锁读 output_modes 快照）
    auto findVisualize = [&]() -> std::shared_ptr<VisualizeOutput> {
        std::lock_guard<std::mutex> lock(output_mtx);
        for (auto& m : output_modes)
            if (m->type() == OutputMode::VISUALIZE)
                return std::static_pointer_cast<VisualizeOutput>(m);
        return nullptr;
    };
    auto findGimbal = [&]() -> std::shared_ptr<GimbalOutput> {
        std::lock_guard<std::mutex> lock(output_mtx);
        for (auto& m : output_modes)
            if (m->type() == OutputMode::GIMBAL)
                return std::static_pointer_cast<GimbalOutput>(m);
        return nullptr;
    };
    // 当前输出模式组合名（覆盖层显示用，如 "Visualize+Gimbal" / "None"）
    auto outputNames = [&]() -> std::string {
        std::lock_guard<std::mutex> lock(output_mtx);
        std::string s;
        for (auto& m : output_modes) {
            if (!s.empty()) s += "+";
            s += m->getName();
        }
        return s.empty() ? "None" : s;
    };

    // 初始输出模式（在 ensureGimbalThread/ensureVisualizeThread 赋值后调用，
    // 见下方"启动"处；首次开启某模式时会创建对应循环线程）
    auto makeOutputs = [&](const OutputConfig& oc) {
        {
            std::lock_guard<std::mutex> lock(output_mtx);
            output_modes.clear();
        }
        if (oc.visualize) {
            ensureVisualizeThread();   // 首次开启时创建可视化循环线程（随后不销毁）
            auto vis = std::make_shared<VisualizeOutput>(camera_proj, sequence_predictor);
            vis->setMode(active_pipeline->mode());
            // 初始流水线为 Armor 且可视化开启：打开 XY 平面窗口
            if (active_pipeline->mode() == PipelineMode::ARMOR) {
                vis->openArmorXYWindow();
            }
            std::lock_guard<std::mutex> lock(output_mtx);
            output_modes.push_back(vis);
        }
        if (oc.gimbal) {
            ensureRobotController();
            ensureGimbalThread();      // 首次开启时创建云台循环线程（随后不销毁）
            auto gimbal = std::make_shared<GimbalOutput>(sequence_predictor, *robot_controller);
            std::lock_guard<std::mutex> lock(output_mtx);
            output_modes.push_back(gimbal);
        }
        std::cout << "[main] Output modes: " << outputNames() << std::endl;
    };

    // ── 运行时切换接口（API；热键在 visualize 线程的窗口内触发）──
    // 两条流水线始终构造，切换只交换指针 + 清空状态；推理进程按启动策略处理：
    // 非 lazy 模式推理进程始终存在；lazy 模式（infer_process_lazy=true）由
    // InferProcessManager 按需切换（立即停止不需要的进程，后台启动需要的进程）。
    switchPipeline = [&](PipelineMode m) {
        std::string new_name;
        {
            std::lock_guard<std::mutex> lock(pipeline_mtx);
            if (active_pipeline->mode() == m) return;
            active_pipeline->clear();   // 清空旧流水线（队列 + 滤波状态）
            active_pipeline = (m == PipelineMode::ARMOR)
                ? static_cast<IPipeline*>(&armor_pipeline)
                : static_cast<IPipeline*>(&power_rune_pipeline);
            active_pipeline->clear();
            new_name = active_pipeline->name();
        }
        if (std::shared_ptr<VisualizeOutput> vis = findVisualize()) {
            vis->setMode(m);
            // XY 平面窗口：进入 Armor 模式时开启、退出 Armor 模式时自动关闭
            // （仅在可视化开启时执行——findVisualize() 非空即可视化开启）
            if (m == PipelineMode::ARMOR) {
                vis->openArmorXYWindow();
            } else {
                vis->closeArmorXYWindow();
            }
        }
        // 仅 lazy 模式由 InferProcessManager 管理推理进程（切换时按需启停）；
        // 非 lazy 模式推理进程常驻，切换只交换指针，不触碰推理进程。
        if (cfg.common.inferProcessLazy) {
            infer_manager->switchTo(m == PipelineMode::ARMOR
                ? Infer::InferProcessManager::Kind::ARMOR
                : Infer::InferProcessManager::Kind::POWER_RUNE);
        }
        std::cout << "[main] Pipeline -> " << new_name << std::endl;
    };
    // 'v'：切换可视化开关（不影响云台）；'g'：切换云台开关（不影响可视化）；'n'：全部关闭
    toggleOutput = [&](OutputMode m) {
        std::shared_ptr<VisualizeOutput> removed_vis;   // 被移除的可视化输出（用于关闭 XY 窗口）
        bool add = true;
        {
            std::lock_guard<std::mutex> lock(output_mtx);
            for (auto it = output_modes.begin(); it != output_modes.end(); ++it) {
                if ((*it)->type() == m) {
                    if (m == OutputMode::VISUALIZE) {
                        removed_vis = std::static_pointer_cast<VisualizeOutput>(*it);
                    }
                    output_modes.erase(it);
                    add = false;
                    break;
                }
            }
        }
        // 可视化关闭：同步关闭 Armor XY 平面窗口（窗口操作须在窗口线程内）
        if (removed_vis) {
            removed_vis->closeArmorXYWindow();
        }
        if (add) {
            if (m == OutputMode::VISUALIZE) {
                ensureVisualizeThread();   // 首次开启时创建可视化循环线程（随后不销毁）
                auto vis = std::make_shared<VisualizeOutput>(camera_proj, sequence_predictor);
                vis->setMode(active_pipeline->mode());
                // 可视化开启且当前为 Armor 模式：打开 XY 平面窗口
                if (active_pipeline->mode() == PipelineMode::ARMOR) {
                    vis->openArmorXYWindow();
                }
                std::lock_guard<std::mutex> lock(output_mtx);
                output_modes.push_back(vis);
            } else if (m == OutputMode::GIMBAL) {
                ensureRobotController();
                ensureGimbalThread();      // 首次开启时创建云台循环线程（随后不销毁）
                auto gimbal = std::make_shared<GimbalOutput>(sequence_predictor, *robot_controller);
                std::lock_guard<std::mutex> lock(output_mtx);
                output_modes.push_back(gimbal);
            }
        }
        std::cout << "[main] Output modes: " << outputNames() << std::endl;
    };

    // ── 输入线程：从 input_mode 取帧，送入激活流水线 ──
    using TimePoint = std::chrono::steady_clock::time_point;
    static_assert(std::atomic<TimePoint>::is_always_lock_free,
                  "std::atomic<steady_clock::time_point> must be lock-free");
    std::atomic<TimePoint> shared_frame_timestamp{TimePoint{}};
    std::atomic<bool> t1_done{false};
    std::atomic<bool> t2_done{false};

    std::thread input_thread([&]() {
        while (!t2_done.load(std::memory_order_acquire) && g_running) {
            TimePoint time_for_delay = std::chrono::steady_clock::now();

            cv::Mat frame;
            TimePoint frame_timestamp;
            ExtraInputInfo extra_info;
            if (!input_mode->getNextFrame(frame, frame_timestamp, extra_info)) break;

            // 无论输入源是否产生新帧，都把 frame_timestamp 发布到共享变量：
            // 无新帧时各输入模式也会更新 frame_timestamp 与 extra_info（约定见
            // IInputMode），保证处理线程的时钟（tryPopFrame 的 min_delay 判定）
            // 在输入源停顿期间持续推进，已入队结果仍能按时提取。
            shared_frame_timestamp.store(frame_timestamp, std::memory_order_release);

            if (frame.empty()) {
                // 输入源无新帧：时间戳已发布，短暂等待后继续轮询
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // 录制：addFrame 会移动原帧，故先克隆一份（仅在录制开启时）
            cv::Mat frame_for_record;
            if (recorder && recorder->active()) {
                frame_for_record = frame.clone();
            }

            // 无论 addFrame 成败都记录该帧；accepted 标记该帧是否成功加入流水线
            // （active_pipeline 由窗口线程切换，加锁读取）
            bool accepted;
            {
                std::lock_guard<std::mutex> lock(pipeline_mtx);
                accepted = active_pipeline->addFrame(std::move(frame), frame_timestamp, extra_info);
            }
            if (recorder && !frame_for_record.empty()) {
                recorder->recordFrame(std::move(frame_for_record), frame_timestamp, extra_info, accepted);
            }

            // 测试最大帧率：test_max_fps 的作用已移入 VideoInputMode（开启时其
            // getFrameDelay() 返回 0，不做按视频帧率的节流）。
            float base_delay = input_mode->getFrameDelay();
            // 队列积压自适应额外延迟（可选功能，BacklogAdaptiveDelay；关闭时为 0）：
            // 叠加在基础帧间隔之上，流水线积压严重时放慢取帧节奏、给流水线消化时间
            float extra_delay_s = backlog_delay.extraDelaySeconds();
            float delay_s = base_delay + extra_delay_s -
                std::chrono::duration<float>(std::chrono::steady_clock::now() - time_for_delay).count();
            int delay_us = static_cast<int>(delay_s * 1e6);
            if (delay_us > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
            }
        }
        t1_done.store(true, std::memory_order_release);
    });

    // ── 处理线程：提取到时帧 → 填充弹道缓冲位即截止 ──
    // 本线程不再做任何输出工作（弹道/云台/可视化/窗口均由各自循环线程处理），
    // 有效帧时只组装弹道请求并发布到 ballistic_slot。
    std::thread process_thread([&]() {
        FrameRateCounter fps(60);
        while (!t1_done.load(std::memory_order_acquire) && g_running) {
            const TimePoint timestamp = shared_frame_timestamp.load(std::memory_order_acquire);
            if (timestamp == TimePoint{}) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            PipelineResult result;
            {
                std::lock_guard<std::mutex> lock(pipeline_mtx);   // active_pipeline 由窗口线程切换
                result = active_pipeline->tryPopFrame(timestamp);
            }

            // ── 队列积压自适应额外延迟：tryPopFrame 返回各缓冲队列积压个数，
            // 除输出缓冲队列外的积压总数（input + inter0..inter3）交给 // 注：若新增Pipeline阶段数不为5，需调整这里
            // BacklogAdaptiveDelay 调整取帧线程的额外延迟（功能关闭时跳过）──
            if (backlog_delay.enabled()) {
                const int non_output_backlog =
                    result.queue_sizes.input + result.queue_sizes.inter0 + result.queue_sizes.inter1 +
                    result.queue_sizes.inter2 + result.queue_sizes.inter3;
                backlog_delay.update(non_output_backlog);
            }

            // ── 有效帧：填充弹道解算缓冲位（本线程工作至此截止）──
            // 缓冲位未被取走时直接覆盖（latest-wins），流水线提取不因下游耗时阻塞。
            if (result.valid) {
                RobotController::State st;
                RobotController* rc = nullptr;
                {
                    std::lock_guard<std::mutex> lock(rc_mtx);
                    rc = robot_controller.get();
                    if (rc) st = rc->getState();
                }
                BallisticRequest req;
                req.st = st;
                req.rc = rc;
                req.result = std::make_unique<PipelineResult>(std::move(result));
                ballistic_slot.publish(std::move(req));
                fps.tick();
                pipeline_fps.store(fps.fps(), std::memory_order_relaxed);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        t2_done.store(true, std::memory_order_release);
    });

    // ── 弹道解算循环线程 ──
    // 循环取弹道缓冲位中的请求处理（弹道解算），并给每一个输出模式的缓冲位填入；
    // 缓冲位空则等待，有数据即处理（不设帧率上限，最新请求覆盖未消费的旧请求）。
    // 所需时间戳直接取最新 shared_frame_timestamp。
    std::thread ballistic_thread([&]() {
        FrameRateCounter fps(60);
        BallisticRequest req;
        while (ballistic_slot.take(req)) {
            if (req.result) {
                // 弹道解算：按流水线模式选择目标策略（与原 process_thread 内逻辑一致）
                if (req.result->armor.target_valid && req.result->armor.target_predictor) {
                    sequence_predictor.setTargetSelection(PredictedBallisticSolver::TargetSelection::NEAREST);
                    sequence_predictor.predict(req.st, *req.result->armor.target_predictor,
                                          shared_frame_timestamp.load(std::memory_order_acquire),
                                          req.result->armor.target_predictor_timestamp);
                } else if (req.result->power_rune.target_predictor) {
                    sequence_predictor.setTargetSelection(PredictedBallisticSolver::TargetSelection::LOWEST_Z);
                    sequence_predictor.predict(req.st, *req.result->power_rune.target_predictor,
                                          shared_frame_timestamp.load(std::memory_order_acquire),
                                          req.result->power_rune.predictor_timestamp);
                } else {
                    sequence_predictor.invalidate();
                }

                // 给每一个输出模式的输入缓冲填入（模式关闭时对象为空，对应线程跳过）
                VisualizeRequest vreq;
                vreq.vis = findVisualize();
                vreq.result = std::move(req.result);
                vreq.rc = req.rc;
                visualize_stage.slot.publish(std::move(vreq));

                GimbalRequest greq;
                greq.gimbal = findGimbal();
                gimbal_stage.slot.publish(std::move(greq));
            }
            fps.tick();
            ballistic_fps.store(fps.fps(), std::memory_order_relaxed);
        }
    });

    // ── 云台输出循环线程（模式首次开启时创建，随后不销毁）──
    // 消费云台缓冲位中的请求，调用 GimbalOutput::update（发送控制序列）。
    ensureGimbalThread = [&]() {
        if (gimbal_stage.thread) return;
        gimbal_stage.thread = std::make_unique<std::thread>([&]() {
            FrameRateCounter fps(60);
            GimbalRequest req;
            while (gimbal_stage.slot.take(req)) {
                if (req.gimbal) {
                    // GimbalOutput 仅读取 result.valid（云台状态/瞄准点自行读取），
                    // 用轻量 stub 传入（完整流水线结果已转给可视化线程）
                    PipelineResult stub;
                    stub.valid = true;
                    req.gimbal->update(stub, nullptr);
                    fps.tick();
                    gimbal_stage.fps.store(fps.fps(), std::memory_order_relaxed);
                }
            }
        });
    };

    // ── 可视化输出循环线程（模式首次开启时创建，随后不销毁）──
    // drawOverlay、imshow 与按键读取等窗口相关操作均在本线程；
    // 无请求时仍持续泵窗口（tryTake 非阻塞），保证按键响应。
    ensureVisualizeThread = [&]() {
        if (visualize_stage.thread) return;
        visualize_stage.thread = std::make_unique<std::thread>([&]() {
            FrameRateCounter fps(60);
            cv::Mat last_display;      // 最近渲染画面（可视化开启时显示 + 覆盖层）
            cv::Mat last_raw_frame;    // 最近原始帧（可视化关闭时窗口显示原始画面）
            PipelineResult::QueueSizes last_qs;
            while (!t1_done.load(std::memory_order_acquire) && g_running) {
                // 取可视化缓冲位（非阻塞；窗口泵不因无请求而停顿）
                VisualizeRequest req;
                const bool got = visualize_stage.slot.tryTake(req);
                if (got && req.result) {
                    last_qs = req.result->queue_sizes;
                    if (req.vis) {
                        req.vis->update(*req.result, req.rc);
                        last_display = req.vis->display();
                        fps.tick();
                        visualize_stage.fps.store(fps.fps(), std::memory_order_relaxed);
                    }
                    last_raw_frame = req.result->frame;   // 浅拷贝（引用计数保活）
                }

                // ── 窗口泵：drawOverlay + imshow + 按键读取（均在本线程）──
                const std::shared_ptr<VisualizeOutput> vis = findVisualize();
                const bool vis_active = (vis != nullptr && !last_display.empty());
                cv::Mat to_show = vis_active ? last_display : last_raw_frame;
                if (!to_show.empty()) {
                    if (vis_active) {
                        drawOverlay(to_show, active_pipeline->name(), outputNames(),
                                    pipeline_fps.load(std::memory_order_relaxed),
                                    ballistic_fps.load(std::memory_order_relaxed),
                                    gimbal_stage.fps.load(std::memory_order_relaxed),
                                    visualize_stage.fps.load(std::memory_order_relaxed),
                                    robotControllerPtr(),
                                    shared_frame_timestamp.load(std::memory_order_acquire),
                                    last_qs, backlog_delay.extraDelaySeconds());
                    }
                    cv::imshow("Unified Auto-Aim", to_show);
                }
                int key = cv::waitKey(1) & 0xFF;
                if (key == 'q' || key == 'Q' || key == 27) {
                    t1_done.store(true, std::memory_order_release);
                    break;
                } else if (key == '1') {
                    switchPipeline(PipelineMode::ARMOR);
                } else if (key == '2') {
                    switchPipeline(PipelineMode::POWER_RUNE);
                } else if (key == 'n') {
                    // 全部关闭（窗口保留，仅显示原始画面）
                    std::shared_ptr<VisualizeOutput> vis_to_close;
                    {
                        std::lock_guard<std::mutex> lock(output_mtx);
                        for (auto& m : output_modes) {
                            if (m->type() == OutputMode::VISUALIZE) {
                                vis_to_close = std::static_pointer_cast<VisualizeOutput>(m);
                            }
                        }
                        output_modes.clear();
                    }
                    // 可视化全部关闭：同步关闭 Armor XY 平面窗口
                    if (vis_to_close) {
                        vis_to_close->closeArmorXYWindow();
                    }
                    std::cout << "[main] Output modes: " << outputNames() << std::endl;
                } else if (key == 'v') {
                    // 开关可视化：关闭后窗口仅显示原始画面，热键始终可用，可随时恢复
                    toggleOutput(OutputMode::VISUALIZE);
                } else if (key == 'g') {
                    toggleOutput(OutputMode::GIMBAL);
                }
                if (!got) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        });
    };

    // ── 启动：创建初始输出模式（首次开启某模式时创建对应循环线程）──
    makeOutputs(opt.output);

    input_thread.join();
    process_thread.join();

    // 停止各阶段循环线程：先停弹道（其消费结束后不再向输出缓冲位发布），
    // 再停输出线程；stop() 唤醒阻塞的 take() 使线程退出，随后 join。
    // 注意：云台线程可能在可视化线程内被创建（'g' 热键），故先 join 可视化
    // 线程（提供 happens-before）再读取/停止云台线程，避免对 thread 指针的竞态。
    ballistic_slot.stop();
    ballistic_thread.join();
    if (visualize_stage.thread) {
        visualize_stage.slot.stop();
        visualize_stage.thread->join();
    }
    if (gimbal_stage.thread) {
        gimbal_stage.slot.stop();
        gimbal_stage.thread->join();
    }

    std::cout << "\nExiting." << std::endl;
    if (recorder) {
        recorder->close();   // 写视频尾并关闭信息文件（幂等）
        if (recorder->spaceExhausted()) {
            std::cout << "[main] 注意：录制因剩余空间不足而提前停止。" << std::endl;
        }
    }
    if (camera) camera->stop();
    // 停止推理进程管理器（幂等）：lazy 模式停止全部由管理器启动的推理进程并等待
    // 后台切换线程结束；非 lazy 模式仅回收本管理器强制重启后持有的进程。
    infer_manager->shutdown();
    // robot_controller 析构时自动停止串口线程与 MPC 后台发送线程
    cv::destroyAllWindows();
    return 0;
}
