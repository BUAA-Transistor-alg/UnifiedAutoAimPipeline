// main.cpp — 统一自瞄流水线主程序（Outpost / PowerRune 双流水线 + 输入/输出模式）
//
// 用法：
//   unified_auto_aim [--pipeline outpost|power_rune]
//                    [--input camera|video|interactive]
//                    [--output none|visualize|gimbal]
//                    [-v <video>] [-e <extra_info.txt>] [--record] [--skip-unaccepted] [-i] [-h]
//
// 设计要点：
//  - 两个流水线在启动时全部构造（模型编译一次），之后不重建；
//    任意时刻只有一个是"激活"状态（接收帧），运行时可通过 API（switchPipeline）
//    或热键 '1'/'2' 切换；切换时调用 clear() 清空其队列与滤波状态；
//  - 输入模式（相机/视频/交互）与输出模式（无/可视化/云台控制）在启动时
//    指定，运行时也可通过 API（switchPipeline / toggleOutput）或热键 '1'/'2'、'v'/'g' 切换；
//  - RobotController（串口 + MPC）仅在需要时构造（相机输入或云台输出）。
#include "common/Input/IInputMode.h"
#include "common/Input/CameraInputMode.h"
#include "common/Input/VideoInputMode.h"
#include "common/Input/InteractiveInputMode.h"
#include "common/IPipeline.h"
#include "Outpost/OutpostPipeline.h"
#include "PowerRune/PowerRunePipeline.h"
#include "common/Output/IOutputMode.h"
#include "common/Output/VisualizeOutput.h"
#include "common/Output/GimbalOutput.h"
#include "common/Ballistic/AimPredictor.h"
#include "common/RobotConfig.h"
#include "common/FrameRateCounter.h"
#include "common/Record/FrameRecorder.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <sstream>
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
    PipelineMode pipeline = PipelineMode::OUTPOST;
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
              << "  --pipeline outpost|power_rune  流水线模式 (默认 outpost)\n"
              << "  --input camera|video|interactive  输入模式 (默认 interactive)\n"
              << "  --output <mode>[+<mode>...]     输出模式组合 (默认 visualize)\n"
              << "            none | visualize | gimbal | visualize+gimbal\n"
              << "  -v, --video <file>              视频路径 (input=video 时必需)\n"
              << "  -e, --extra-info <file>         视频额外信息 txt (相机位姿 / 完整 ExtraInputInfo)\n"
              << "  -i, --interactive               交互式图片输入 (等价 --input interactive)\n"
              << "  -r, --record                    开启录制：每帧保存为 MKV 视频 + 时间戳 + 完整\n"
              << "                                   extra_info txt（输出目录与剩余空间阈值见\n"
              << "                                   config/config.yaml 的 common.recording）\n"
              << "      --skip-unaccepted           回放视频时跳过录制中未成功加入流水线的帧\n"
              << "                                   （需 -e 指定 v2 格式的 extra info 文件）\n"
              << "  -h, --help                      显示帮助\n"
              << "无参数运行默认显示本帮助。\n"
              << "窗口行为:\n"
              << "  - 启动时未指定 visualize：不显示窗口、不绘制任何可视化，纯后台运行\n"
              << "  - 启动时指定 visualize：可视化开启时绘制检测/位姿/滤波 + 统一覆盖层\n"
              << "    (串口信息 / 帧数统计 / 热键提醒，两流水线一致)；按 'v' 关闭可视化后\n"
              << "    窗口仅显示原始画面，热键仍可用，'v' 可随时恢复\n"
              << "热键 (窗口内):\n"
              << "  '1'/'2' 切换流水线  'v' 开关可视化  'g' 开关云台输出  'n' 关闭全部输出  'q'/ESC 退出\n";
}

static PipelineMode parsePipeline(const std::string& s) {
    if (s == "outpost") return PipelineMode::OUTPOST;
    if (s == "power_rune") return PipelineMode::POWER_RUNE;
    throw std::runtime_error("未知流水线模式: " + s + " (可选 outpost / power_rune)");
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
// 可视化开启时，无论 Outpost 还是 PowerRune 流水线，都在画面上叠加：
//   1. 热键提醒  — 顶部 (10,20)，0.6 号粗 2 绿
//   2. 队列积压  — 原 main info 位置 (10,60)，Q[in i0 i1 i2 i3 out] 格式
//   3. 帧数统计  — 原 drawFps 样式：右上角 "FPS: xx" 0.7 粗 2 黄 + "TS: .." 0.45 粗 1 黄
//   4. 串口输入信息 — 原 drawCommInfo 样式：(8,80) 起 0.45 粗 1 绿，按来源分块显示
//      （--- MCU --- / --- IMU --- / --- FUSED --- / --- STRICT --- / --- MPC ---），
//      MCU 块含温度行（按温度区间变色）
static void drawOverlay(cv::Mat& img,
                        const std::string& pipeline_name,
                        const std::string& output_names,
                        double fps,
                        RobotController* rc,
                        const std::chrono::steady_clock::time_point& frame_ts,
                        const PipelineResult::QueueSizes& queue_sizes) {
    // 1. 热键提醒（顶部）
    cv::putText(img, "Keys: 1/2 pipeline | v visualize | g gimbal | n none | q quit",
                cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

    // 2. 当前流水线各缓存积压长度（与原来 main info 的 Q[...] 格式一致）
    char qbuf[160];
    std::snprintf(qbuf, sizeof(qbuf), "Q[in:%d i0:%d i1:%d i2:%d i3:%d out:%d]",
                  queue_sizes.input, queue_sizes.inter0, queue_sizes.inter1,
                  queue_sizes.inter2, queue_sizes.inter3, queue_sizes.output);
    cv::putText(img, qbuf, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2);

    // 3. 帧数统计（原 drawFps 样式，右上角）
    using namespace std::chrono;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << "FPS: " << fps;
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
    const RobotController::State st = rc->getState();

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
    std::cout << "  Pipeline: " << (opt.pipeline == PipelineMode::OUTPOST ? "Outpost" : "PowerRune") << std::endl;
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

    // 相机参数：按输入模式自动选择（camera → input_mode.camera_mode；video/interactive → input_mode.video_mode）
    const RobotConfig::CameraParams& camera_params =
        (opt.input == InputKind::CAMERA) ? cfg.common.inputMode.cameraMode : cfg.common.inputMode.videoMode;

    // ── RobotController（仅在需要时构造：相机输入需要 strict 数据，云台输出需要控制）──
    // 相机输入模式下 CameraInputMode 后台线程持续采样 rc_.getState()（延迟队列），
    // 因此相机输入需要 RobotController（无硬件时串口线程静默失败）。
    std::unique_ptr<RobotController> robot_controller;
    auto ensureRobotController = [&]() {
        if (robot_controller) return;
        const auto& rp = cfg.common.robotController;
        robot_controller = std::make_unique<RobotController>(
            rp.dtControl, rp.mpcPredN, rp.J, rp.tauC, rp.b, rp.tauD,
            rp.maxTorque, rp.maxTorqueRate, rp.Q, rp.R, rp.Rd, rp.maxIter,
            rp.integralGain,
            McuDataPreprocessor::LinearParams{
                rp.sendPitchScale, rp.sendPitchOffset,
                rp.recvPitchScale, rp.recvPitchOffset},
            rp.sequenceMode);
        std::cout << "[main] RobotController constructed (serial threads may fail silently without hardware)." << std::endl;
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
                                                          opt.skip_unaccepted);
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

    // ── 启动时构造两套流水线（模型编译一次，之后不重建）──
    // max_delay_seconds 共用 config common.max_delay_seconds；相机参数按输入模式选择
    const std::array<int, OutpostPipeline::NUM_QUEUES> queue_sizes = {10, 10, 10, 10, 10, 10};
    OutpostPipeline   outpost_pipeline(queue_sizes, cfg.common.maxDelaySeconds, camera_params);
    PowerRunePipeline power_rune_pipeline(queue_sizes, cfg.common.maxDelaySeconds, camera_params);
    IPipeline* active_pipeline =
        (opt.pipeline == PipelineMode::OUTPOST) ? static_cast<IPipeline*>(&outpost_pipeline)
                                                : static_cast<IPipeline*>(&power_rune_pipeline);

    // ── 输出模式组合（visualize 与 gimbal 独立开关，可同时启用；随时可切换）──
    std::vector<std::unique_ptr<IOutputMode>> output_modes;
    // 预测瞄准点通用类：任何情况下（无论输出模式）每帧调用，生成预测云台控制
    // 序列 + 瞄准点序列；GimbalOutput 消费控制序列，VisualizeOutput 消费首个瞄准点
    AimPredictor aim_predictor;
    // 相机投影（与输入模式绑定，两个流水线/可视化共用）
    auto camera_proj = std::make_shared<CameraProjection>(
        camera_params.cameraMatrix, camera_params.distCoeffs,
        ImageResolution{camera_params.width, camera_params.height});
    auto makeOutputs = [&](const OutputConfig& oc) {
        output_modes.clear();
        if (oc.visualize) {
            auto vis = std::make_unique<VisualizeOutput>(camera_proj, aim_predictor);
            vis->setMode(active_pipeline->mode());
            output_modes.push_back(std::move(vis));
        }
        if (oc.gimbal) {
            ensureRobotController();
            output_modes.push_back(std::make_unique<GimbalOutput>(aim_predictor, *robot_controller));
        }
        std::cout << "[main] Output modes:";
        if (output_modes.empty()) std::cout << " None";
        for (auto& m : output_modes) std::cout << " " << m->getName();
        std::cout << std::endl;
    };
    makeOutputs(opt.output);

    // 查找当前 visualize 输出（若存在）
    auto findVisualize = [&]() -> VisualizeOutput* {
        for (auto& m : output_modes)
            if (m->type() == OutputMode::VISUALIZE)
                return static_cast<VisualizeOutput*>(m.get());
        return nullptr;
    };

    // 当前输出模式组合名（覆盖层显示用，如 "Visualize+Gimbal" / "None"）
    auto outputNames = [&]() -> std::string {
        std::string s;
        for (auto& m : output_modes) {
            if (!s.empty()) s += "+";
            s += m->getName();
        }
        return s.empty() ? "None" : s;
    };

    // ── 运行时切换接口（API；热键在 visualize 窗口内触发）──
    auto switchPipeline = [&](PipelineMode m) {
        if (active_pipeline->mode() == m) return;
        // 清空旧流水线（队列 + 滤波状态），切换到新流水线开始填充
        active_pipeline->clear();
        active_pipeline = (m == PipelineMode::OUTPOST)
            ? static_cast<IPipeline*>(&outpost_pipeline)
            : static_cast<IPipeline*>(&power_rune_pipeline);
        active_pipeline->clear();
        if (VisualizeOutput* vis = findVisualize()) {
            vis->setMode(m);
        }
        std::cout << "[main] Pipeline -> " << active_pipeline->name() << std::endl;
    };
    // 'v'：切换可视化开关（不影响云台）；'g'：切换云台开关（不影响可视化）；'n'：全部关闭
    auto toggleOutput = [&](OutputMode m) {
        bool add = true;
        for (auto it = output_modes.begin(); it != output_modes.end(); ++it) {
            if ((*it)->type() == m) {
                output_modes.erase(it);
                add = false;
                break;
            }
        }
        if (add) {
            if (m == OutputMode::VISUALIZE) {
                auto vis = std::make_unique<VisualizeOutput>(camera_proj, aim_predictor);
                vis->setMode(active_pipeline->mode());
                output_modes.push_back(std::move(vis));
            } else if (m == OutputMode::GIMBAL) {
                ensureRobotController();
                output_modes.push_back(std::make_unique<GimbalOutput>(aim_predictor, *robot_controller));
            }
        }
        std::cout << "[main] Output modes:";
        if (output_modes.empty()) std::cout << " None";
        for (auto& om : output_modes) std::cout << " " << om->getName();
        std::cout << std::endl;
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
            if (frame.empty()) {
                // 输入源无新帧：短暂等待后继续轮询
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            shared_frame_timestamp.store(frame_timestamp, std::memory_order_release);

            // 录制：addFrame 会移动原帧，故先克隆一份（仅在录制开启时）
            cv::Mat frame_for_record;
            if (recorder && recorder->active()) {
                frame_for_record = frame.clone();
            }

            // 无论 addFrame 成败都记录该帧；accepted 标记该帧是否成功加入流水线
            bool accepted = active_pipeline->addFrame(std::move(frame), frame_timestamp, extra_info);
            if (recorder && !frame_for_record.empty()) {
                recorder->recordFrame(frame_for_record, frame_timestamp, extra_info, accepted);
            }

            float delay_s = input_mode->getFrameDelay() -
                std::chrono::duration<float>(std::chrono::steady_clock::now() - time_for_delay).count();
            int delay_us = static_cast<int>(delay_s * 1e6);
            if (delay_us > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
            }
        }
        t1_done.store(true, std::memory_order_release);
    });

    // ── 处理线程：提取到时帧 → 输出模式；窗口显示与热键 ──
    // 窗口仅在启动时指定了 visualize 时创建；否则纯后台运行（不显示窗口、不绘制任何可视化）。
    const bool show_window = opt.output.visualize;
    std::thread process_thread([&]() {
        FrameRateCounter overlay_fps(60);
        cv::Mat last_frame;   // 最近一帧（可视化关闭时窗口仅显示原始画面）
        TimePoint last_valid_ts{};   // 最近一次有效帧时间戳（覆盖层 TS 显示用；
                                    // tryPopFrame 返回无效帧时 frame_timestamp 为默认 0）
        while (!t1_done.load(std::memory_order_acquire) && g_running) {
            TimePoint timestamp = shared_frame_timestamp.load(std::memory_order_acquire);
            if (timestamp == TimePoint{}) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            PipelineResult result = active_pipeline->tryPopFrame(timestamp);

            // ── 统一预测瞄准点（任何情况下都调用，与输出模式无关）──
            // 生成预测云台控制序列 + 瞄准点序列；GimbalOutput 消费控制序列，
            // VisualizeOutput 消费瞄准点序列第一个值。
            if (result.valid) {
                const RobotController::State st =
                    robot_controller ? robot_controller->getState() : RobotController::State{};
                if (result.outpost.esekf_initialized && result.outpost.predictor) {
                    aim_predictor.setTargetSelection(PredictedBallisticSolver::TargetSelection::NEAREST);
                    aim_predictor.predict(st, *result.outpost.predictor);
                } else if (result.power_rune.target_predictor) {
                    // PowerRune：靶点预测函数已在流水线内部封装为统一签名
                    // std::vector<cv::Point3f>(double)（TargetPositionCalculator::compose），
                    // 直接传入 AimPredictor，无需再包装。
                    aim_predictor.setTargetSelection(PredictedBallisticSolver::TargetSelection::LOWEST_Z);
                    aim_predictor.predict(st, *result.power_rune.target_predictor);
                } else {
                    aim_predictor.invalidate();
                }
            }

            // 更新全部输出模式（visualize 与 gimbal 可同时启用）
            for (auto& m : output_modes) {
                m->update(result, robot_controller.get());
            }

            if (!show_window) {
                // 纯后台运行：不显示窗口、不绘制任何可视化信息
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // 可视化开启：显示各 visualizer 的完整渲染画面并叠加统一覆盖层
            // （串口信息 / 帧数统计 / 热键提醒，两个流水线一致）；
            // 可视化关闭：窗口仅显示原始画面，不绘制任何覆盖层。
            VisualizeOutput* vis = findVisualize();
            bool vis_active = (vis != nullptr && !vis->display().empty());
            cv::Mat to_show;
            if (vis_active) {
                to_show = vis->display().clone();
                if (result.valid) overlay_fps.tick();
            } else if (!last_frame.empty()) {
                to_show = last_frame.clone();
            }
            if (result.valid) {
                last_valid_ts = result.frame_timestamp;
                last_frame = result.frame.clone();
            }

            if (to_show.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            if (vis_active) {
                drawOverlay(to_show, active_pipeline->name(), outputNames(),
                            overlay_fps.fps(), robot_controller.get(), last_valid_ts,
                            result.queue_sizes);
            }

            cv::imshow("Unified Auto-Aim", to_show);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 'Q' || key == 27) {
                t1_done.store(true, std::memory_order_release);
                break;
            } else if (key == '1') {
                switchPipeline(PipelineMode::OUTPOST);
            } else if (key == '2') {
                switchPipeline(PipelineMode::POWER_RUNE);
            } else if (key == 'n') {
                // 全部关闭（窗口保留，仅显示原始画面）
                output_modes.clear();
                std::cout << "[main] Output modes: None" << std::endl;
            } else if (key == 'v') {
                // 开关可视化：关闭后窗口仅显示原始画面，热键始终可用，可随时恢复
                toggleOutput(OutputMode::VISUALIZE);
            } else if (key == 'g') {
                toggleOutput(OutputMode::GIMBAL);
            }
        }
        t2_done.store(true, std::memory_order_release);
    });

    input_thread.join();
    process_thread.join();

    std::cout << "\nExiting." << std::endl;
    if (recorder) {
        recorder->close();   // 写视频尾并关闭信息文件（幂等）
        if (recorder->spaceExhausted()) {
            std::cout << "[main] 注意：录制因剩余空间不足而提前停止。" << std::endl;
        }
    }
    if (camera) camera->stop();
    // robot_controller 析构时自动停止串口线程与 MPC 后台发送线程
    cv::destroyAllWindows();
    return 0;
}
