// ESEKF.h — 误差状态扩展卡尔曼滤波 (Error-State Extended Kalman Filter)
// 名义状态：三维坐标 + 方向四元数 + 绕世界系 z 轴的旋转速度 ω_z + 两个 z 偏移 dz2/dz3
// 误差状态 (9 维)：[δp(3); δθ(3); δω_z(1); δdz2(1); δdz3(1)]
#ifndef ESEKF_HPP
#define ESEKF_HPP

#include <chrono>
#include <functional>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "TransformTree/RobotTfTree.h"
#include "CameraProjection.h"

class ESEKF {
public:
    using TimePoint = std::chrono::steady_clock::time_point;

    // 滤波可调参数（默认值与历史硬编码值一致；运行时由 config/outpost.esekf 覆盖）
    struct Params {
        double positionNoise;        // 位置过程噪声（Q 位置块）
        double rotationNoise;        // 姿态过程噪声（Q 姿态块）
        double measurementNoise;     // 观测噪声（重投影误差 R）
        double orientationZRegNoise; // 姿态 z 轴正则化观测噪声
        double dzNoise;              // dz 偏移过程噪声（Q(7,7)/Q(8,8)）
        double dzSearchRange;        // dz 黄金分割搜索范围（米）
        double dzLimit;              // dz 偏移限幅（米，|dz| 上限）
        double initPositionNoise;    // 初始化位置噪声系数（P 位置块）
        double initOrientationNoise; // 初始化姿态噪声系数（P 姿态块）
        double initYawRateNoise;     // 初始化旋转速度不确定性（P(6,6)）
        double initDz2Noise;         // 初始化 dz2 不确定性（P(7,7)）
        double initDz3Noise;         // 初始化 dz3 不确定性（P(8,8)）

        Params()
            : positionNoise(100.0), rotationNoise(10.0), measurementNoise(400.0),
              orientationZRegNoise(1e-4), dzNoise(0.1), dzSearchRange(0.5), dzLimit(0.3),
              initPositionNoise(1.0), initOrientationNoise(0.1), initYawRateNoise(1.0),
              initDz2Noise(0.01), initDz3Noise(0.01) {}
    };

    ESEKF(std::shared_ptr<RobotTfTree> tf_tree,
          std::shared_ptr<CameraProjection> camera_proj,
          const std::vector<std::vector<cv::Point3f>>& points_3d_list,
          const std::vector<cv::Point3f>& target_centers_3d_list,
          const Params& params = Params{});

    void init(const cv::Vec3d& position, const cv::Mat& rotation_matrix, const TimePoint& t);
    void reset();
    void update(const std::vector<std::vector<cv::Point2f>>& points_2d_list, const TimePoint& t);
    void predict(const TimePoint& t);

    std::vector<cv::Point3f> getWorldPoints() const;

    /**
     * @brief 捕获当前滤波器状态的快照，返回一个独立于后续状态变化的自身位姿预测函数。
     *
     * 仿照 RollPredictor::capturePredictor：返回的 function 内部复制了调用时刻的
     * 所有状态（位置、姿态四元数、绕世界系 z 轴的旋转速度 ω_z、dz2/dz3 偏移以及
     * 目标中心关键点），不随 ESEKF 后续 update/predict 而改变。
     * 预测运动模型与 predict(dt) 一致：位置不变，仅绕世界系 z 轴以 ω_z 恒速旋转。
     *
     * @return 签名为 std::vector<cv::Point3f>(double dt) 的函数：传入预测时间 dt（秒，
     *         可为任意实数，0 表示当前时刻），返回 OUTPOST_TARGET_CENTER_3D_LIST
     *         关键点（含 dz 偏移）在预测后的位姿下转换到世界坐标系的关键点列表。
     */
    std::unique_ptr<std::function<std::vector<cv::Point3f>(double)>> capturePosePredictor() const;

    std::vector<double> computeError(const cv::Vec3d& position, const cv::Mat& rotation_matrix,
                                     const std::vector<std::vector<cv::Point2f>>& points_2d_list,
                                     const std::vector<size_t>& assignment,
                                     double dz2, double dz3) const;
    cv::Mat computeJacobian(const std::vector<std::vector<cv::Point2f>>& points_2d_list,
                            const std::vector<size_t>& assignment) const;

    const cv::Vec3d& getPosition() const { return position_; }
    const Eigen::Quaterniond& getOrientation() const { return orientation_; }
    double getYawRate() const { return yaw_rate_; }
    double getDz2() const { return dz2_; }
    double getDz3() const { return dz3_; }
    bool isDz2Initialized() const { return dz2_initialized_; }
    bool isDz3Initialized() const { return dz3_initialized_; }
    const TimePoint& getLastTime() const { return last_time_; }
    cv::Mat getRotationMatrix() const;

private:
    void predict(double dt);
    void applyCorrection(const Eigen::VectorXd& dx);
    void applyOrientationZAxisRegularization();

    void enumerateAssignments(const std::vector<std::vector<cv::Point2f>>& p2_list,
                              const cv::Vec3d& position, const cv::Mat& R,
                              size_t depth,
                              std::vector<size_t>& assignment, std::vector<bool>& used,
                              std::vector<size_t>& best_assignment, double& best_cost) const;

    std::vector<cv::Point3f> effectiveLocalPoints(size_t obj_idx, double dz) const;
    std::vector<cv::Point2f> buildConcat2D(const std::vector<std::vector<cv::Point2f>>& points_2d_list) const;
    std::vector<cv::Point3f> buildConcat3D(const std::vector<size_t>& assignment, double dz2, double dz3) const;

    double costForDzPair(const std::vector<std::vector<cv::Point2f>>& p2_list,
                         const std::vector<size_t>& assignment,
                         const cv::Vec3d& position, const cv::Mat& R,
                         double dz2, double dz3) const;
    double optimizeDz(const std::vector<std::vector<cv::Point2f>>& p2_list,
                      const std::vector<size_t>& assignment,
                      const cv::Vec3d& position, const cv::Mat& R,
                      size_t obj_idx) const;
    double assignmentCost(const std::vector<std::vector<cv::Point2f>>& p2_list,
                          const std::vector<size_t>& assignment,
                          const cv::Vec3d& position, const cv::Mat& R) const;

    std::vector<double> computeErrorCore(const cv::Vec3d& position, const cv::Mat& rotation_matrix,
                                         const std::vector<cv::Point2f>& c2,
                                         const std::vector<cv::Point3f>& c3) const;

    static Eigen::Quaterniond rotationMatrixToQuaternion(const cv::Mat& R);
    static cv::Vec3d quaternionToRotationVector(const Eigen::Quaterniond& q);
    static Eigen::Quaterniond rotationVectorToQuaternion(const cv::Vec3d& rotvec);
    static cv::Mat quaternionToRotationMatrix(const Eigen::Quaterniond& q);
    static cv::Mat rotationVectorToRotationMatrix(const cv::Vec3d& rotvec);
    static std::vector<cv::Point3f> localToWorld(const cv::Vec3d& position, const cv::Mat& R,
                                                 const std::vector<cv::Point3f>& local_points);

    std::shared_ptr<RobotTfTree>       tf_tree_;
    std::shared_ptr<CameraProjection>    camera_proj_;
    std::vector<std::vector<cv::Point3f>> points_3d_list_;  // 3 个物体的局部 3D 关键点
    std::vector<cv::Point3f> target_centers_3d_list_;       // 目标中心关键点（局部系，3 个面中心）

    cv::Vec3d          position_;
    Eigen::Quaterniond orientation_;
    double             yaw_rate_;   // 绕世界系 z 轴的旋转速度 (rad/s)
    double             dz2_;
    double             dz3_;
    bool               dz2_initialized_;
    bool               dz3_initialized_;
    TimePoint          last_time_;

    Eigen::Matrix<double, 9, 9> P_;

    double position_noise_;
    double rotation_noise_;
    double measurement_noise_;
    double orientation_z_reg_noise_;
    double dz_noise_;
    double dz_search_range_;
    double dz_limit_;
    double init_position_noise_;
    double init_orientation_noise_;
    double init_yaw_rate_noise_;
    double init_dz2_noise_;
    double init_dz3_noise_;
};

#endif // ESEKF_HPP
