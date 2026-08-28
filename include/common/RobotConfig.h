// RobotConfig.h — 全局参数配置（从项目根目录 config/config.yaml 读取）
//
// ⚠ 重要约定（给后续修改者）：config.yaml 中的所有参数均为必填，本文件及
//   RobotConfig.cpp 中不设任何默认值/回退值——缺段或缺字段时 RobotConfig::load
//   直接抛异常退出，绝不静默采用默认值。新增配置项时必须同步：
//   1) 在 config/config.yaml 对应段中添加字段并写明含义；
//   2) 在 RobotConfig.h 对应结构体中添加成员（无默认初始化）；
//   3) 在 src/common/RobotConfig.cpp 中通过 requireScalar 等读取。
//
// config.yaml 顶层分为三个大类：
//   - common      ：两个流水线共用的参数（tf 偏移 / 相机内参 / 弹道 / MPC / 输入控制器 /
//                    min_delay_seconds 等）
//   - armor     ：Armor 流水线独占参数（推理模型 / 批量 / 观测丢失超时）
//   - power_rune  ：PowerRune 流水线独占参数（推理模型 / NMS / 阈值 / 批量）
//
// 相机内参与畸变系数两流水线共用，但分为两套按输入模式自动选择
// （common.input_mode 下）：
//   - input_mode.camera_mode ：--input camera（实机相机，含 IP/曝光/增益/extra_info_delay）
//   - input_mode.video_mode  ：--input video / interactive（录制视频 / 交互图片）
#ifndef ROBOT_CONFIG_H
#define ROBOT_CONFIG_H

#include <cstdint>
#include <string>
#include <array>

#include <opencv2/opencv.hpp>

class RobotConfig {
public:
    // 变换树各节点相对其父节点的固定偏移（单位：米）
    struct TfOffsets {
        float yawJointZOffset;  // yaw 关节沿 z 轴偏移（相对 chassis）
        float pitchJointYOffset;  // pitch 关节沿 y 轴偏移（相对 yaw 旋转中心）
        float imuOffsetX, imuOffsetY, imuOffsetZ;          // imu 相对 head
        float cameraOffsetX, cameraOffsetY, cameraOffsetZ;  // camera 相对 head
        float muzzleOffsetX, muzzleOffsetY, muzzleOffsetZ;  // muzzle 相对 head
    };

    // 相机参数（分辨率 + 内参 + 畸变；相机模式额外含 IP/曝光/增益/extra_info_delay）
    // ⚠ 给后续修改者：本结构体所有字段均无默认值，必须由 config.yaml 提供：
    //   - 相机模式（common.input_mode.camera_mode）必填 device_ip / net_ip /
    //     exposure / gain / extra_info_delay；
    //   - 视频/交互模式（common.input_mode.video_mode）必填 test_max_fps；
    //   - 两模式均必填 resolution / camera_matrix / dist_coeffs。
    //   缺任一字段 RobotConfig::load 直接抛异常退出，解析见 src/common/RobotConfig.cpp。
    struct CameraParams {
        std::string deviceIp;   // 相机设备 IP（相机模式必填）
        std::string netIp;      // 本机网口 IP（相机模式必填）
        float exposure;         // 曝光时间（微秒，相机模式必填）
        float gain;             // 增益（相机模式必填）
        int width, height;      // 图像分辨率（像素）
        cv::Mat cameraMatrix;       // 3x3 CV_64F 内参矩阵
        cv::Mat distCoeffs;         // Nx1 CV_64F 畸变系数
        // 相机输入模式（CameraInputMode）extra_info 延迟（秒）：
        // 后台线程持续采样 RobotController::getState()，返回给流水线的 extra_info
        // 为相对当前时刻 extra_info_delay 前的队头数据（0.0 = 最新状态）。
        // 相机模式必填（config: extra_info_delay）。
        double extraInfoDelay;
        // 测试最大帧率（视频/交互模式必填，config: test_max_fps）：开启后
        // VideoInputMode 的 getFrameDelay() 返回 0（不做按视频帧率的节流），
        // 用于测量视频输入 + 流水线的最大帧数/FPS。
        bool testMaxFps;
    };

    // 云台角度解算参数
    struct GimbalParams {
        double bulletDiameter;      // 弹丸直径（米）
        double bulletMass;          // 弹丸质量（kg）
        double bulletVelocity;      // 默认弹丸初速（m/s）
        double integrationStep;     // 弹道积分步长（秒）
        double distanceThreshold;   // 弹道最近点距离阈值（米）
        double distanceIterateThreshold;  // 迭代触发阈值（米）
        double stopZ;               // 弹道计算截止高度（world 系，米）
        float  pitchMin;            // pitch 搜索下界（弧度）
        float  pitchMax;            // pitch 搜索上界（弧度）
        float  pitchSearchStep;     // pitch 粗搜索步长（弧度）
    };

    // 预测弹道解算参数
    struct PredictedBallisticParams {
        double extraPredictTime;      // 额外预测时间（秒）
        int    maxIterations;         // 飞行时间迭代上限
        double timeErrorTolerance;    // 迭代提前停止的时间误差容差（秒）
    };

    // RobotController（TorqueController 子模组）构造参数
    struct RobotControllerParams {
        bool   sequenceMode;     // 序列输入模式
        double dtControl;        // 控制周期（秒）
        int    mpcPredN;         // MPC 预测步数
        double J;                // yaw 轴转动惯量
        double tauC;             // 库仑摩擦
        double b;                // 粘滞摩擦系数
        double tauD;             // 常数扰动
        double maxTorque;        // 最大力矩（N·m）
        double maxTorqueRate;    // 最大力矩变化率（N·m/s）
        double Q;                // MPC 状态代价
        double R;                // MPC 控制代价
        double Rd;               // MPC 控制变化率代价
        int    maxIter;          // MPC 迭代上限
        double integralGain;     // yaw 力矩积分补偿比例系数
        // ── MCU 数据线性映射标定参数（McuDataPreprocessor::LinearParams，当前标定默认值）──
        double sendPitchScale;   // imu_euler_pitch → pitch_target_angle（发送）
        double sendPitchOffset;  // 发送偏移
        double recvPitchScale;   // mcu_pitch_angle → imu_euler_pitch（接收）
        double recvPitchOffset;  // 接收偏移
    };

    // 预测序列参数（序列输入模式：预测序列生成与消费）
    struct PredictSequenceParams {
        int    predictionPoints;    // 预测点数：实际精确弹道解算的点数（M）
        int    interpolationRefine; // 插值细化倍数：相邻实际计算点之间细分的返回点间隔数（K）
                                    // 原划分序列点数 = (M-1)*K + 1，间隔恰为 dt_control
        int    exactLeadPoints;     // 前导精确点数（n）：在序列最前面拼接 n 个逐点精确弹道
                                    // 解算的前导点（不使用插值；0 表示关闭，行为与原版一致）。
                                    // 返回序列 = [前导精确点] + [原划分序列]，
                                    // 总返回点数 = (M-1)*K+1+n，精确解算点共 M+n 个。
                                    // 原划分首段（紧邻窗口 n+1..n+K-1）需要外推时参考点改用
                                    // items[n-1]（须与段左端点同目标，否则复制左端点），
                                    // 其余段仍用原规则。取值范围 n >= 0
        int    pitchSeqLead;        // m：pitch 序列提前数（必须小于总返回点数 (M-1)*K+1+n）
        int    fireSeqLead;         // o：fire 序列提前数
        double pitchBias;           // pitch 轴偏置（弧度）
        double yawBias;             // yaw 轴偏置（弧度）
        double fireAngleLowerLimit; // fire 判定角度阈值下限（弧度）
        double fireAngleLength;     // fire 判定弧长（米）
    };

    // 流水线缓冲队列与批量参数（config 各流水线段的 pipeline 子段）
    // 两条流水线结构相同（5 阶段 / 6 队列），各用各的一份配置。
    struct PipelineParams {
        // 6 个缓冲队列最大长度（[输入, 阶段间×4, 输出]）；队列满时新帧直接丢弃。
        // 须与流水线 NUM_QUEUES（=6）一致。
        std::array<int, 6> queueMaxSizes;
        int preprocessBatch;    // 阶段1 预处理最大批量
        int inferenceBatch;     // 阶段2 推理最大批量（须 ≤ inference.max_batch 与共享内存容量）
        int postprocessBatch;   // 阶段3 后处理最大批量
    };

    // Armor 流水线独占参数
    struct ArmorParams {
        std::string modelPath;              // 推理模型路径
        std::string device;                 // 推理设备
        int inputWidth;                     // YOLO 推理输入宽度（像素，须与模型输入一致）
        int inputHeight;                    // YOLO 推理输入高度（像素，须与模型输入一致）
        int maxBatch;                       // 推理最大批量（动态 batch 1..max_batch）
        int shmKey;                         // 共享内存 Key（推理进程通信，见 InferShm.h）
        double observationLostTimeoutSec;   // 连续观测丢失多久后重置滤波器（秒）
        PipelineParams pipeline;            // 缓冲队列长度 + 可批处理阶段批量

        // OutpostESEKF 误差状态扩展卡尔曼滤波参数
        struct EsekfParams {
            double positionNoise;           // 位置过程噪声（Q 位置块）
            double rotationNoise;           // 姿态过程噪声（Q 姿态块）
            double measurementNoise;        // 观测噪声（重投影误差 R）
            double orientationZRegNoise;    // 姿态 z 轴正则化观测噪声
            double dzNoise;                 // dz 偏移过程噪声（Q(7,7)/Q(8,8)）
            double dzSearchRange;           // dz 黄金分割搜索范围（米）
            double dzLimit;                 // dz 偏移限幅（米，|dz| 上限）
            double initPositionNoise;       // 初始化位置噪声系数（P 位置块）
            double initOrientationNoise;    // 初始化姿态噪声系数（P 姿态块）
            double initYawRateNoise;        // 初始化旋转速度不确定性（P(6,6)）
            double initDz2Noise;            // 初始化 dz2 不确定性（P(7,7)）
            double initDz3Noise;            // 初始化 dz3 不确定性（P(8,8)）
        };
        EsekfParams esekf;
    };

    // PowerRune 流水线独占参数
    struct PowerRuneParams {
        std::string modelPath;      // 推理模型路径
        std::string device;         // 推理设备
        int         inputWidth;     // YOLO 推理输入宽度（像素，须与模型输入一致）
        int         inputHeight;    // YOLO 推理输入高度（像素，须与模型输入一致）
        bool        manualNms;      // true: 无 NMS 原始输出，需手动 NMS
        float       confThreshold;  // 置信度阈值
        int         maxBatch;       // 推理最大批量
        int         shmKey;         // 共享内存 Key（推理进程通信，见 InferShm.h）
        PipelineParams pipeline;    // 缓冲队列长度 + 可批处理阶段批量
    };

    // 共用参数（两个流水线共享）
    struct CommonParams {
        TfOffsets    tf;                        // 变换树偏移

        // 输入模式相机参数（common.input_mode）：两流水线共用，按输入模式自动选择
        struct InputModeParams {
            CameraParams cameraMode;    // 相机输入模式（--input camera，含 IP/曝光/增益/extra_info_delay）
            CameraParams videoMode;     // 视频/互动输入模式（--input video/interactive）
        };
        InputModeParams inputMode;

        // 推理进程启动策略（必填，config: common.infer_process_lazy）：
        // true: 仅启动当前流水线所需推理进程（按需启停，见 InferProcessManager）；
        // false: 启动时启动全部推理进程并后台闲置（launch_all.py 预启动）
        bool inferProcessLazy;

        // 队列积压自适应额外延迟（可选功能，见 BacklogAdaptiveDelay，v6 PID 式 PI）：
        // 开启后处理线程每次 tryPopFrame() 后统计除输出缓冲队列外各缓冲队列
        // 积压总数，以 targetBacklog 为目标做 PI 直接输出
        // （extra_delay = gainP*e + ∫gainI*e·dt，抗饱和，无 EMA、无速率上限），
        // 得到取帧线程的额外延迟。小量级增益下稳态无极限环震荡。
        // ⚠ 给后续修改者：本结构体所有字段均无默认值，必须由 config.yaml 的
        //   common.backlog_adaptive_delay 段提供（缺段/缺字段时 RobotConfig::load
        //   直接抛异常退出），解析见 src/common/RobotConfig.cpp，取值校验也在那里。
        struct BacklogAdaptiveDelayParams {
            bool   enabled;             // 功能总开关（config: enabled）
            double targetBacklog;       // 目标积压（设定点，>= 1，config: target_backlog）
            double maxExtraDelaySeconds; // 额外延迟上限（秒，config: max_extra_delay_seconds）
            double gainP;               // 比例增益，秒/单位积压 >= 0（config: gain_p）
            double gainI;               // 积分增益，秒/单位积压/秒 >= 0（config: gain_i）
        };
        BacklogAdaptiveDelayParams backlogAdaptiveDelay;

        GimbalParams gimbal;                    // 云台解算参数
        PredictedBallisticParams predictedBallistic;  // 预测弹道解算参数
        RobotControllerParams robotController;         // RobotController 构造参数
        PredictSequenceParams predictSequence;         // 预测序列参数
        double       minDelaySeconds;           // 两个流水线共用的提取帧最小延迟（秒）

        // 录制参数（必填段，config: common.recording；--record 开启录制时生效）
        // ⚠ 无默认值：output_dir / min_free_space_mb 必须由 config.yaml 提供。
        struct RecordingParams {
            std::string outputDir;      // 录制输出目录（相对项目根目录或以 / 开头为绝对路径）
            int64_t minFreeSpaceBytes;  // 剩余空间阈值（字节，低于该值停止写入；
                                        // config 中为 MB，加载时换算）
        };
        RecordingParams recording;
    };

    CommonParams    common;      // 共用参数
    ArmorParams   armor;     // Armor 独占参数
    PowerRuneParams powerRune;   // PowerRune 独占参数

    // 从指定 yaml 文件加载配置；文件缺失或格式错误抛出 std::runtime_error。
    static RobotConfig load(const std::string& yamlPath);

    // 懒加载单例：首次调用时读取 PathResolver::resolvePath("config/config.yaml")。
    static RobotConfig& instance();
};

#endif // ROBOT_CONFIG_H
