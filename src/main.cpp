// main.cpp — 统一自瞄流水线主程序（Outpost / PowerRune 双流水线 + 输入/输出模式）
//
// 用法：
//   unified_auto_aim [--pipeline outpost|power_rune]
//                    [--input camera|video|interactive]
//                    [--output none|visualize|gimbal]
//                    [-v <video>] [-e <extra_info.txt>] [-i] [-h]
//
// 设计要点：
//  - 两个流水线在启动时全部构造（模型编译一次），之后不重建；
//    任意时刻只有一个是"激活"状态（接收帧），运行时可通过 API（switchPipeline）
//    或热键 '1'/'2' 切换；切换时调用 clear() 清空其队列与滤波状态；
//  - 输入模式（相机/视频/交互）与输出模式（无/可视化/云台控制）在启动时
//    指定，运行时也可通过 API（switchPipeline / toggleOutput）或热键 '1'/'2'、'v'/'g' 切换；
//  - RobotController（串口 + MPC）仅在需要时构造（相机输入或云台输出）。
#include "IInputMode.h"
#include "CameraInputMode.h"
#include "VideoInputMode.h"
#include "InteractiveInputMode.h"
#include "IPipeline.h"
#include "OutpostPipeline.h"
#include "PowerRunePipeline.h"
#include "Output/IOutputMode.h"
#include "Output/VisualizeOutput.h"
#include "Output/GimbalOutput.h"
#include "RobotConfig.h"
#include "FrameRateCounter.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
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

    Options() { output.visualize = true; }
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --pipeline outpost|power_rune  流水线模式 (默认 outpost)\n"
              << "  --input camera|video|interactive  输入模式 (默认 interactive)\n"
              << "  --output <mode>[+<mode>...]     输出模式组合 (默认 visualize)\n"
              << "            none | visualize | gimbal | visualize+gimbal\n"
              << "  -v, --video <file>              视频路径 (input=video 时必需)\n"
              << "  -e, --extra-info <file>         视频额外信息 txt (相机位姿)\n"
              << "  -i, --interactive               交互式图片输入 (等价 --input interactive)\n"
              << "  -h, --help                      显示帮助\n"
              << "无参数运行默认显示本帮助。\n"
              << "热键 (窗口内，窗口常驻不消失):\n"
              << "  '1'/'2' 切换流水线  'v' 开关可视化  'g' 开关云台输出  'n' 关闭全部输出  'q'/ESC 退出\n"
              << "  (窗口常驻：关闭可视化后仍显示最近一帧，热键始终可用，'v' 可随时恢复)\n";
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
    std::cout << "========================================" << std::endl;

    const RobotConfig& cfg = RobotConfig::instance();

    // ── RobotController（仅在需要时构造：相机输入需要 strict 数据，云台输出需要控制）──
    std::unique_ptr<RobotController> robot_controller;
    auto ensureRobotController = [&]() {
        if (robot_controller) return;
        const auto& rp = cfg.robotController;
        robot_controller = std::make_unique<RobotController>(
            rp.dtControl, rp.mpcPredN, rp.J, rp.tauC, rp.b, rp.tauD,
            rp.maxTorque, rp.maxTorqueRate, rp.Q, rp.R, rp.Rd, rp.maxIter,
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
            camera = std::make_unique<Camera>(cfg.camera.deviceIp, cfg.camera.netIp);
            camera->setExposureTime(cfg.camera.exposure);
            camera->setGain(cfg.camera.gain);
            if (!camera->start()) {
                std::cerr << "Camera start failed!" << std::endl;
                return -1;
            }
            input_mode = std::make_unique<CameraInputMode>(*camera, *robot_controller);
            break;
        }
        case InputKind::VIDEO:
            input_mode = std::make_unique<VideoInputMode>(opt.video_path, opt.extra_info_path);
            break;
        case InputKind::INTERACTIVE:
        default:
            input_mode = std::make_unique<InteractiveInputMode>();
            break;
    }

    // ── 启动时构造两套流水线（模型编译一次，之后不重建）──
    OutpostPipeline   outpost_pipeline;
    PowerRunePipeline power_rune_pipeline;
    IPipeline* active_pipeline =
        (opt.pipeline == PipelineMode::OUTPOST) ? static_cast<IPipeline*>(&outpost_pipeline)
                                                : static_cast<IPipeline*>(&power_rune_pipeline);

    // ── 输出模式组合（visualize 与 gimbal 独立开关，可同时启用；随时可切换）──
    std::vector<std::unique_ptr<IOutputMode>> output_modes;
    auto makeOutputs = [&](const OutputConfig& oc) {
        output_modes.clear();
        if (oc.visualize) {
            auto vis = std::make_unique<VisualizeOutput>();
            vis->setMode(active_pipeline->mode());
            output_modes.push_back(std::move(vis));
        }
        if (oc.gimbal) {
            ensureRobotController();
            output_modes.push_back(std::make_unique<GimbalOutput>(*robot_controller));
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
                auto vis = std::make_unique<VisualizeOutput>();
                vis->setMode(active_pipeline->mode());
                output_modes.push_back(std::move(vis));
            } else if (m == OutputMode::GIMBAL) {
                ensureRobotController();
                output_modes.push_back(std::make_unique<GimbalOutput>(*robot_controller));
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
    cv::Mat last_frame;   // 最近一帧（无 visualize 输出时窗口兜底显示，保证窗口/热键/退出路径常驻）

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
            active_pipeline->addFrame(std::move(frame), frame_timestamp, extra_info);

            float delay_s = input_mode->getFrameDelay() -
                std::chrono::duration<float>(std::chrono::steady_clock::now() - time_for_delay).count();
            int delay_us = static_cast<int>(delay_s * 1e6);
            if (delay_us > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
            }
        }
        t1_done.store(true, std::memory_order_release);
    });

    // ── 处理线程：提取到时帧 → 输出模式；visualize 窗口内处理热键 ──
    std::thread process_thread([&]() {
        while (!t1_done.load(std::memory_order_acquire) && g_running) {
            TimePoint timestamp = shared_frame_timestamp.load(std::memory_order_acquire);
            if (timestamp == TimePoint{}) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            PipelineResult result = active_pipeline->tryPopFrame(timestamp);
            // 更新全部输出模式（visualize 与 gimbal 可同时启用）
            for (auto& m : output_modes) {
                m->update(result, robot_controller.get());
            }

            // 窗口常驻：有 visualize 输出时显示渲染画面；否则显示最近一帧原图
            // （附提示文字），保证任意输出状态下都有窗口、热键与退出路径，不会卡死。
            VisualizeOutput* vis = findVisualize();
            cv::Mat to_show;
            if (vis != nullptr && !vis->display().empty()) {
                to_show = vis->display();
            } else if (!last_frame.empty()) {
                to_show = last_frame.clone();
                cv::putText(to_show, "No visual output - 'g' gimbal | '1'/'2' pipeline | 'q' quit",
                            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(0, 255, 0), 2);
            }
            if (result.valid) {
                last_frame = result.frame.clone();   // 保留最近一帧（无 visualize 时兜底显示）
            }

            if (!to_show.empty()) {
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
                    // 全部关闭（窗口保留，显示最近一帧）
                    output_modes.clear();
                    std::cout << "[main] Output modes: None" << std::endl;
                } else if (key == 'v') {
                    // 开关可视化：关闭后窗口仍显示最近一帧（常驻），热键始终可用，可随时恢复
                    toggleOutput(OutputMode::VISUALIZE);
                } else if (key == 'g') {
                    toggleOutput(OutputMode::GIMBAL);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        t2_done.store(true, std::memory_order_release);
    });

    input_thread.join();
    process_thread.join();

    std::cout << "\nExiting." << std::endl;
    if (camera) camera->stop();
    // robot_controller 析构时自动停止串口线程与 MPC 后台发送线程
    cv::destroyAllWindows();
    return 0;
}
