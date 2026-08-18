// RobotConfig.h — 全局参数配置（从项目根目录 config/config.yaml 读取）
//
// config.yaml 顶层分为三个大类：
//   - common      ：两个流水线共用的参数（tf 偏移 / 相机内参 / 弹道 / MPC / 输入控制器 /
//                    max_delay_seconds 等）
//   - outpost     ：Outpost 流水线独占参数（推理模型 / 观测丢失超时）
//   - power_rune  ：PowerRune 流水线独占参数（推理模型 / NMS / 阈值 / 批量）
//
// 相机内参与畸变系数两流水线共用，但分为两套按输入模式自动选择：
//   - camera_mode ：--input camera（实机相机，含 IP/曝光/增益）
//   - video_mode  ：--input video / interactive（录制视频 / 交互图片）
#ifndef ROBOT_CONFIG_H
#define ROBOT_CONFIG_H

#include <cstdint>
#include <string>

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

    // 相机参数（分辨率 + 内参 + 畸变；相机模式额外含 IP/曝光/增益）
    struct CameraParams {
        std::string deviceIp;   // 相机设备 IP（相机模式）
        std::string netIp;      // 本机网口 IP（相机模式）
        float exposure = 5000.0f;   // 曝光时间（微秒，相机模式）
        float gain = 16.0f;         // 增益（相机模式）
        int width = 0, height = 0;  // 图像分辨率（像素）
        cv::Mat cameraMatrix;       // 3x3 CV_64F 内参矩阵
        cv::Mat distCoeffs;         // Nx1 CV_64F 畸变系数
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
        // ── MCU 数据线性映射标定参数（McuDataPreprocessor::LinearParams，当前标定默认值）──
        double sendPitchScale;   // imu_euler_pitch → pitch_target_angle（发送）
        double sendPitchOffset;  // 发送偏移
        double recvPitchScale;   // mcu_pitch_angle → imu_euler_pitch（接收）
        double recvPitchOffset;  // 接收偏移
    };

    // 云台输入控制器参数（序列输入模式）
    struct InputControllerParams {
        int    predictionPoints;    // 预测点数：实际精确弹道解算的点数（M）
        int    interpolationRefine; // 插值细化倍数：相邻实际计算点之间细分的返回点间隔数（K）
                                    // 总返回点数 = (M-1)*K + 1，间隔恰为 dt_control
        int    pitchSeqLead;        // m：pitch 序列提前数（必须小于总返回点数）
        int    fireSeqLead;         // o：fire 序列提前数
        double pitchBias;           // pitch 轴偏置（弧度）
        double fireAngleLowerLimit; // fire 判定角度阈值下限（弧度）
        double fireAngleLength;     // fire 判定弧长（米）
    };

    // Outpost 流水线独占参数
    struct OutpostParams {
        std::string modelPath;              // 推理模型路径
        std::string device;                 // 推理设备
        double observationLostTimeoutSec;   // 连续观测丢失多久后重置滤波器（秒）
    };

    // PowerRune 流水线独占参数
    struct PowerRuneParams {
        std::string modelPath;      // 推理模型路径
        std::string device;         // 推理设备
        bool        manualNms;      // true: 无 NMS 原始输出，需手动 NMS
        float       confThreshold;  // 置信度阈值
        int         maxBatch;       // 推理最大批量
    };

    // 共用参数（两个流水线共享）
    struct CommonParams {
        TfOffsets    tf;                        // 变换树偏移
        CameraParams cameraMode;                // 相机输入模式（--input camera）
        CameraParams videoMode;                 // 视频/互动输入模式（--input video/interactive）
        GimbalParams gimbal;                    // 云台解算参数
        PredictedBallisticParams predictedBallistic;  // 预测弹道解算参数
        RobotControllerParams robotController;         // RobotController 构造参数
        InputControllerParams inputController;         // 云台输入控制器参数
        double       maxDelaySeconds;           // 两个流水线共用的提取帧最大延迟（秒）

        // 录制参数（--record 开启录制时生效）
        struct RecordingParams {
            std::string outputDir = "recordings";           // 录制输出目录（相对项目根目录或以 / 开头为绝对路径）
            int64_t minFreeSpaceBytes = 1024LL * 1024 * 1024; // 剩余空间阈值（字节），低于该值停止写入
        };
        RecordingParams recording;
    };

    CommonParams    common;      // 共用参数
    OutpostParams   outpost;     // Outpost 独占参数
    PowerRuneParams powerRune;   // PowerRune 独占参数

    // 从指定 yaml 文件加载配置；文件缺失或格式错误抛出 std::runtime_error。
    static RobotConfig load(const std::string& yamlPath);

    // 懒加载单例：首次调用时读取 PathResolver::resolvePath("config/config.yaml")。
    static RobotConfig& instance();
};

#endif // ROBOT_CONFIG_H
