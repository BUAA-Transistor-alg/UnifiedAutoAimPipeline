// RobotConfig.h — 全局参数配置（从项目根目录 config/config.yaml 读取）
#ifndef ROBOT_CONFIG_H
#define ROBOT_CONFIG_H

#include <string>

#include <opencv2/opencv.hpp>

// 集中管理所有可调参数（变换树偏移、相机、推理等）。
// 数据来源：项目根目录 config/config.yaml（相对路径经 PathResolver 解析为绝对路径）。
// 各对象构造时通过 RobotConfig::instance() 自动读取，修改配置文件即可调整参数，无需重新编译。
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

    // 相机参数
    struct CameraParams {
        std::string deviceIp;   // 相机设备 IP
        std::string netIp;      // 本机网口 IP
        float exposure;         // 曝光时间（微秒）
        float gain;             // 增益
        int width, height;      // 图像分辨率（像素）
        cv::Mat cameraMatrix;   // 3x3 CV_64F 内参矩阵
        cv::Mat distCoeffs;     // Nx1 CV_64F 畸变系数
    };

    // 云台角度解算参数
    struct GimbalParams {
        double bulletDiameter;      // 弹丸直径（米）
        double bulletMass;          // 弹丸质量（kg）
        double bulletVelocity;      // 默认弹丸初速（m/s）
        double integrationStep;     // 弹道积分步长（秒）
        double distanceThreshold;   // 弹道最近点距离阈值（米）
        double distanceIterateThreshold;  // 迭代触发阈值（米）：首轮距离大于该值则触发 yaw-pitch 迭代
        double stopZ;               // 弹道计算截止高度（world 系，米），低于该高度停止积分
        float  pitchMin;            // pitch 搜索下界（弧度）
        float  pitchMax;            // pitch 搜索上界（弧度）
        float  pitchSearchStep;     // pitch 粗搜索步长（弧度）
    };

    // 预测弹道解算参数
    struct PredictedBallisticParams {
        double extraPredictTime;      // 额外预测时间（秒），叠加在弹道飞行时间之上
        int    maxIterations;         // 飞行时间迭代上限
        double timeErrorTolerance;    // 迭代提前停止的时间误差容差（秒）
    };

    // RobotController（TorqueController 子模组）构造参数
    struct RobotControllerParams {
        bool   sequenceMode;     // 序列输入模式（true=序列 set / false=单目标 set）
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
    };

    // 云台输入控制器参数（序列输入模式）
    struct InputControllerParams {
        int    predictionSeqLen;   // n：预测序列长度（i=1..n 预测 extra+i*dt+飞行时间 的目标）
        int    pitchSeqLead;       // m：pitch 序列提前数（必须小于 n），pitch 序列截取第 m 个之后
        int    fireSeqLead;        // o：fire 序列提前数，fire 序列截取第 o 个之后
        double pitchBias;          // pitch 轴偏置（弧度），输出时叠加
        double fireAngleLowerLimit;// fire 判定角度阈值下限（弧度）
        double fireAngleLength;    // fire 判定弧长（米）：动态阈值 = length / 目标 xy 距离
    };

    // PowerRune（能量机关）推理与相机参数
    struct PowerRuneParams {
        std::string modelPath;      // 模型路径（相对项目根目录，或绝对路径）
        std::string device;         // 推理设备（CPU/GPU/...）
        bool        manualNms;      // true: 无 NMS 原始输出，需手动 NMS
        float       confThreshold;  // 置信度阈值
        int         maxBatch;       // 推理最大批量
        int         width, height;  // 图像分辨率（像素）
        cv::Mat     cameraMatrix;   // 3x3 CV_64F 内参矩阵
        cv::Mat     distCoeffs;     // Nx1 CV_64F 畸变系数
    };

    TfOffsets    tf;                        // 变换树偏移
    CameraParams camera;                    // 相机参数
    GimbalParams gimbal;                    // 云台解算参数
    PredictedBallisticParams predictedBallistic;  // 预测弹道解算参数
    RobotControllerParams robotController;         // RobotController 构造参数
    InputControllerParams inputController;         // 云台输入控制器参数
    PowerRuneParams powerRune;                     // PowerRune（能量机关）推理与相机参数
    std::string  modelPath;                 // Outpost 推理模型路径（相对项目根目录，或绝对路径）
    std::string  inferenceDevice;           // Outpost 推理设备（CPU/GPU/...）
    double       observationLostTimeoutSec;  // 连续观测丢失多久后重置滤波器（秒）

    // 从指定 yaml 文件加载配置；文件缺失或格式错误抛出 std::runtime_error。
    static RobotConfig load(const std::string& yamlPath);

    // 懒加载单例：首次调用时读取 PathResolver::resolvePath("config/config.yaml")。
    // RobotTfTree 等对象在构造时自动调用本方法。
    static RobotConfig& instance();
};

#endif // ROBOT_CONFIG_H
