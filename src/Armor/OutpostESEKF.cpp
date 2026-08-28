#include "Armor/OutpostESEKF.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "common/TransformTree/CoordinateTransform.h"

OutpostESEKF::OutpostESEKF(std::shared_ptr<RobotTfTree> tf_tree,
             std::shared_ptr<CameraProjection> camera_proj,
             const std::vector<std::vector<cv::Point3f>>& points_3d_list,
             const std::vector<cv::Point3f>& target_centers_3d_list,
             const Params& params)
    : tf_tree_(tf_tree),
      camera_proj_(camera_proj),
      points_3d_list_(points_3d_list),
      target_centers_3d_list_(target_centers_3d_list),
      position_(0.0, 0.0, 0.0),
      orientation_(Eigen::Quaterniond::Identity()),
      yaw_rate_(0.0),
      dz2_(0.0), dz3_(0.0),
      dz2_initialized_(false), dz3_initialized_(false),
      initialized_(false),
      has_observation_time_(false),
      observation_lost_timeout_(params.observationLostTimeoutSec),
      position_noise_(params.positionNoise),
      rotation_noise_(params.rotationNoise),
      measurement_noise_(params.measurementNoise),
      orientation_z_reg_noise_(params.orientationZRegNoise),
      dz_noise_(params.dzNoise),
      dz_search_range_(params.dzSearchRange),
      dz_limit_(params.dzLimit),
      init_position_noise_(params.initPositionNoise),
      init_orientation_noise_(params.initOrientationNoise),
      init_yaw_rate_noise_(params.initYawRateNoise),
      init_dz2_noise_(params.initDz2Noise),
      init_dz3_noise_(params.initDz3Noise) {
    CV_Assert(!points_3d_list_.empty());
    const size_t np = points_3d_list_[0].size();
    for (const auto& p : points_3d_list_) CV_Assert(p.size() == np);
    P_.setIdentity();
}

bool OutpostESEKF::init(const std::vector<cv::Point2f>& points_2d,
                 const TimePoint& t) {
    CV_Assert(points_2d.size() == points_3d_list_[0].size());

    // 所需位姿信息在内部计算（与原有 stage4 初始化 PnP 一致）：
    // 1. 对面 0 模型（points_3d_list_[0]）做 PnP，得到相机系位置/欧拉角
    cv::Vec3f position_cam, euler_cam;
    bool pnp_ok = camera_proj_->solvePnP_Cam(
        points_3d_list_[0], points_2d,
        {cv::SOLVEPNP_IPPE, cv::SOLVEPNP_ITERATIVE},
        position_cam, euler_cam);
    if (!pnp_ok)
        return false;

    // 2. 相机系 → 世界系
    cv::Vec3f pos_center = tf_tree_->transformPoint(RobotTfTree::CAMERA, RobotTfTree::WORLD, position_cam);
    cv::Vec3f euler_center = tf_tree_->transformEuler(RobotTfTree::CAMERA, RobotTfTree::WORLD, euler_cam);
    cv::Mat R_center = CoordinateTransform::eulerToRotationMatrix(euler_center);

    // 3. 名义状态 + 初始协方差
    position_ = cv::Vec3d(pos_center[0], pos_center[1], pos_center[2]);
    orientation_ = rotationMatrixToQuaternion(R_center);
    yaw_rate_ = 0.0;
    last_time_ = t;

    P_ = Eigen::Matrix<double, 9, 9>::Identity();
    P_.topLeftCorner<3, 3>() *= init_position_noise_;
    P_.block<3, 3>(3, 3)   *= init_orientation_noise_;
    P_(6, 6) = init_yaw_rate_noise_;   // 绕 z 轴旋转速度的初始不确定性
    P_(7, 7) = init_dz2_noise_;
    P_(8, 8) = init_dz3_noise_;
    return true;
}

void OutpostESEKF::reset() {
    position_ = cv::Vec3d(0.0, 0.0, 0.0);
    orientation_.setIdentity();
    yaw_rate_ = 0.0;
    dz2_ = 0.0;
    dz3_ = 0.0;
    dz2_initialized_ = false;
    dz3_initialized_ = false;
    initialized_ = false;
    last_observation_time_ = TimePoint();
    has_observation_time_ = false;
    last_time_ = TimePoint();
    P_.setIdentity();
}

bool OutpostESEKF::processFrame(const std::vector<std::vector<cv::Point2f>>& all_image_points,
                                const TimePoint& t) {
    const bool has_obs = !all_image_points.empty();

    // ── 观测丢失计时：连续超过阈值未观测到目标则重置 ──
    if (has_obs) {
        last_observation_time_ = t;
        has_observation_time_ = true;
    }
    if (has_observation_time_ &&
        std::chrono::duration<double>(t - last_observation_time_).count() > observation_lost_timeout_) {
        reset();   // 复位后初始化标志 / 观测计时均为 false，下方按未初始化处理
    }

    // ── 初始化 / 更新 / 仅预测 ──
    if (!initialized_) {
        // 初始化：传入首个观测的图像关键点，位姿信息（PnP + 世界系转换）在 init 内部计算
        if (has_obs)
            initialized_ = init(all_image_points[0], t);
    } else {
        if (has_obs) {
            // 更新时观测截断到 EKF 支持的最大观测数（3D 模型个数，即 3 个装甲面）
            const size_t n = std::min(all_image_points.size(), points_3d_list_.size());
            std::vector<std::vector<cv::Point2f>> obs_points(
                all_image_points.begin(), all_image_points.begin() + n);
            update(obs_points, t);
        } else {
            predict(t);   // 无观测，仅推进运动模型
        }
    }
    return initialized_;
}

void OutpostESEKF::update(const std::vector<std::vector<cv::Point2f>>& points_2d_list,
                   const TimePoint& t) {
    const size_t M = points_2d_list.size();
    CV_Assert(M > 0);
    const size_t num_points = points_2d_list[0].size();
    for (const auto& p : points_2d_list) CV_Assert(p.size() == num_points);
    CV_Assert(num_points == points_3d_list_[0].size());

    // 1. 恒定旋转速率运动模型预测（仅绕世界系 z 轴）
    double dt = std::chrono::duration<double>(t - last_time_).count();
    last_time_ = t;
    if (dt > 0.0) predict(dt);

    const cv::Mat rotation_matrix = quaternionToRotationMatrix(orientation_);

    // 2. 枚举 2D→3D 对应关系；每个候选配对都搜索使总误差最小的 dz
    std::vector<size_t> assignment(M, 0);
    std::vector<size_t> best_assignment(M, 0);
    std::vector<bool> used(points_3d_list_.size(), false);
    double best_cost = std::numeric_limits<double>::infinity();
    enumerateAssignments(points_2d_list, position_, rotation_matrix, 0,
                         assignment, used, best_assignment, best_cost);

    // 3. 更新标志位（dz 状态由 EKF 正常更新，不再直接设为误差最小的值）
    for (size_t i = 0; i < M; ++i) {
        if (best_assignment[i] == 1) dz2_initialized_ = true;
        else if (best_assignment[i] == 2) dz3_initialized_ = true;
    }

    // 4. 误差与雅可比（使用状态 dz2_/dz3_）
    std::vector<double> error = computeError(position_, rotation_matrix, points_2d_list,
                                             best_assignment, dz2_, dz3_);
    cv::Mat H = computeJacobian(points_2d_list, best_assignment);
    const size_t error_dim = error.size();

    Eigen::MatrixXd R = Eigen::MatrixXd::Identity((int)error_dim, (int)error_dim)
                        * measurement_noise_;
    Eigen::MatrixXd He((int)error_dim, 9);
    for (size_t r = 0; r < error_dim; ++r)
        for (int c = 0; c < 9; ++c)
            He((int)r, c) = H.at<double>((int)r, c);

    Eigen::MatrixXd Ht = He.transpose();
    Eigen::MatrixXd S = He * P_ * Ht + R;
    Eigen::MatrixXd K = P_ * Ht * S.inverse();

    Eigen::VectorXd e((int)error_dim);
    for (size_t i = 0; i < error_dim; ++i) e((int)i) = error[i];

    Eigen::VectorXd dx = K * (-e);
    applyCorrection(dx);

    Eigen::Matrix<double, 9, 9> KH = Eigen::Matrix<double, 9, 9>(K * He);
    P_ = (Eigen::Matrix<double, 9, 9>::Identity() - KH) * P_;

    // 5. 正则化（保持姿态 z 轴与世界系 z 轴对齐）
    applyOrientationZAxisRegularization();
}

void OutpostESEKF::predict(const TimePoint& t) {
    double dt = std::chrono::duration<double>(t - last_time_).count();
    last_time_ = t;
    if (dt > 0.0) predict(dt);

    applyOrientationZAxisRegularization();
}

std::vector<cv::Point3f> OutpostESEKF::getWorldPoints() const {
    cv::Mat R = quaternionToRotationMatrix(orientation_);
    std::vector<cv::Point3f> all_local;
    for (size_t i = 0; i < points_3d_list_.size(); ++i) {
        double dz = 0.0;
        if (i == 1) dz = dz2_;
        else if (i == 2) dz = dz3_;
        std::vector<cv::Point3f> pts = effectiveLocalPoints(i, dz);
        all_local.insert(all_local.end(), pts.begin(), pts.end());
    }
    return localToWorld(position_, R, all_local);
}

std::unique_ptr<std::function<std::vector<cv::Point3f>(double)>>
OutpostESEKF::capturePosePredictor() const {
    // 捕获调用时刻的所有状态（值拷贝），返回的函数不随 OutpostESEKF 后续状态变化而改变
    const cv::Vec3d pos = position_;
    const Eigen::Quaterniond q = orientation_;
    const double yaw_rate = yaw_rate_;
    const double dz2 = dz2_;
    const double dz3 = dz3_;
    const std::vector<cv::Point3f> centers = target_centers_3d_list_;

    auto func = [pos, q, yaw_rate, dz2, dz3, centers](double dt) -> std::vector<cv::Point3f> {
        // 与 predict(dt) 一致的运动模型：位置不变，仅绕世界系 z 轴以 ω_z 恒速旋转
        Eigen::Quaterniond q_pred = q;
        if (dt != 0.0) {
            Eigen::Quaterniond dq = rotationVectorToQuaternion(cv::Vec3d(0.0, 0.0, yaw_rate * dt));
            q_pred = dq * q;
            q_pred.normalize();
        }
        const cv::Mat R_pred = quaternionToRotationMatrix(q_pred);

        // 目标中心关键点（面 1/2 施加对应的 dz 偏移，与 effectiveLocalPoints 一致）
        std::vector<cv::Point3f> local = centers;
        for (size_t i = 0; i < local.size(); ++i) {
            double dz = 0.0;
            if (i == 1) dz = dz2;
            else if (i == 2) dz = dz3;
            local[i].z += (float)dz;
        }
        return localToWorld(pos, R_pred, local);
    };

    return std::make_unique<std::function<std::vector<cv::Point3f>(double)>>(std::move(func));
}

std::vector<double> OutpostESEKF::computeError(const cv::Vec3d& position,
                                        const cv::Mat& rotation_matrix,
                                        const std::vector<std::vector<cv::Point2f>>& points_2d_list,
                                        const std::vector<size_t>& assignment,
                                        double dz2, double dz3) const {
    return computeErrorCore(position, rotation_matrix,
                            buildConcat2D(points_2d_list),
                            buildConcat3D(assignment, dz2, dz3));
}

cv::Mat OutpostESEKF::computeJacobian(const std::vector<std::vector<cv::Point2f>>& points_2d_list,
                               const std::vector<size_t>& assignment) const {
    const std::vector<cv::Point2f> c2 = buildConcat2D(points_2d_list);
    const size_t n = c2.size();
    const size_t error_dim = 2 * n;
    cv::Mat J((int)error_dim, 9, CV_64F, cv::Scalar(0.0));

    const cv::Vec3d rotvec = quaternionToRotationVector(orientation_);
    const cv::Mat base_R = quaternionToRotationMatrix(orientation_);
    const double delta = 1e-5;

    // 位置 (0-2)
    for (int c = 0; c < 3; ++c) {
        cv::Vec3d p_plus = position_, p_minus = position_;
        p_plus[c] += delta; p_minus[c] -= delta;
        std::vector<double> e_plus = computeErrorCore(p_plus, base_R, c2, buildConcat3D(assignment, dz2_, dz3_));
        std::vector<double> e_minus = computeErrorCore(p_minus, base_R, c2, buildConcat3D(assignment, dz2_, dz3_));
        for (size_t r = 0; r < error_dim; ++r)
            J.at<double>((int)r, c) = (e_plus[r] - e_minus[r]) / (2.0 * delta);
    }

    // 旋转向量 (3-5)
    for (int c = 0; c < 3; ++c) {
        cv::Vec3d r_plus = rotvec, r_minus = rotvec;
        r_plus[c] += delta; r_minus[c] -= delta;
        cv::Mat R_plus = rotationVectorToRotationMatrix(r_plus);
        cv::Mat R_minus = rotationVectorToRotationMatrix(r_minus);
        std::vector<double> e_plus = computeErrorCore(position_, R_plus, c2, buildConcat3D(assignment, dz2_, dz3_));
        std::vector<double> e_minus = computeErrorCore(position_, R_minus, c2, buildConcat3D(assignment, dz2_, dz3_));
        for (size_t r = 0; r < error_dim; ++r)
            J.at<double>((int)r, 3 + c) = (e_plus[r] - e_minus[r]) / (2.0 * delta);
    }

    // δω_z (6)：误差不直接依赖旋转速度，该列保持为零
    // dz2 (7)、dz3 (8)：仅当对应物体被匹配时参与
    bool has1 = false, has2 = false;
    for (size_t i = 0; i < assignment.size(); ++i) {
        if (assignment[i] == 1) has1 = true;
        else if (assignment[i] == 2) has2 = true;
    }
    if (has1) {
        std::vector<double> e_plus = computeErrorCore(position_, base_R, c2, buildConcat3D(assignment, dz2_ + delta, dz3_));
        std::vector<double> e_minus = computeErrorCore(position_, base_R, c2, buildConcat3D(assignment, dz2_ - delta, dz3_));
        for (size_t r = 0; r < error_dim; ++r)
            J.at<double>((int)r, 7) = (e_plus[r] - e_minus[r]) / (2.0 * delta);
    }
    if (has2) {
        std::vector<double> e_plus = computeErrorCore(position_, base_R, c2, buildConcat3D(assignment, dz2_, dz3_ + delta));
        std::vector<double> e_minus = computeErrorCore(position_, base_R, c2, buildConcat3D(assignment, dz2_, dz3_ - delta));
        for (size_t r = 0; r < error_dim; ++r)
            J.at<double>((int)r, 8) = (e_plus[r] - e_minus[r]) / (2.0 * delta);
    }

    return J;
}

cv::Mat OutpostESEKF::getRotationMatrix() const {
    return quaternionToRotationMatrix(orientation_);
}

void OutpostESEKF::predict(double dt) {
    if (dt <= 0.0) return;

    // 名义状态：位置不变，仅绕世界系 z 轴以 ω_z 恒速旋转
    Eigen::Quaterniond dq = rotationVectorToQuaternion(cv::Vec3d(0.0, 0.0, yaw_rate_ * dt));
    orientation_ = dq * orientation_;
    orientation_.normalize();

    Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
    const double theta = yaw_rate_ * dt;
    Eigen::Matrix3d Rz;
    Rz << std::cos(theta), -std::sin(theta), 0.0,
          std::sin(theta),  std::cos(theta), 0.0,
          0.0, 0.0, 1.0;
    // δθ_{k+1} = Rz·δθ_k + z·δω_z·dt  （z = 世界系 z 轴）
    F.block<3, 3>(3, 3) = Rz;
    F.block<3, 1>(3, 6) = Eigen::Vector3d(0.0, 0.0, dt);

    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;
    Eigen::Matrix<double, 9, 9> Q = Eigen::Matrix<double, 9, 9>::Zero();
    Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (position_noise_ * dt4 / 4.0);
    Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (rotation_noise_ * dt4 / 4.0);
    Q.block<3, 1>(3, 6) = Eigen::Vector3d(0.0, 0.0, 1.0) * (rotation_noise_ * dt3 / 2.0);
    Q.block<1, 3>(6, 3) = Q.block<3, 1>(3, 6).transpose();
    Q(6, 6) = rotation_noise_ * dt2;
    Q(7, 7) = dz_noise_ * dt;
    Q(8, 8) = dz_noise_ * dt;

    P_ = F * P_ * F.transpose() + Q;
}

void OutpostESEKF::applyCorrection(const Eigen::VectorXd& dx) {
    position_[0] += dx(0);
    position_[1] += dx(1);
    position_[2] += dx(2);

    cv::Vec3d dtheta(dx(3), dx(4), dx(5));
    Eigen::Quaterniond dq = rotationVectorToQuaternion(dtheta);
    orientation_ = orientation_ * dq;
    orientation_.normalize();

    yaw_rate_ += dx(6);

    dz2_ += dx(7);
    dz3_ += dx(8);

    dz2_ = std::clamp(dz2_, -dz_limit_, dz_limit_);
    dz3_ = std::clamp(dz3_, -dz_limit_, dz_limit_);
}

void OutpostESEKF::applyOrientationZAxisRegularization() {
    cv::Mat Rmat = quaternionToRotationMatrix(orientation_);
    const double R00 = Rmat.at<double>(0, 0);
    const double R01 = Rmat.at<double>(0, 1);
    const double R02 = Rmat.at<double>(0, 2);
    const double R10 = Rmat.at<double>(1, 0);
    const double R11 = Rmat.at<double>(1, 1);
    const double R12 = Rmat.at<double>(1, 2);
    const double R20 = Rmat.at<double>(2, 0);
    const double R21 = Rmat.at<double>(2, 1);
    const double R22 = Rmat.at<double>(2, 2);

    Eigen::VectorXd e(3);
    e(0) = -R02;
    e(1) = -R12;
    e(2) = 1.0 - R22;

    Eigen::MatrixXd He = Eigen::MatrixXd::Zero(3, 9);
    He(0, 3) = R01; He(0, 4) = -R00;
    He(1, 3) = R11; He(1, 4) = -R10;
    He(2, 3) = R21; He(2, 4) = -R20;

    Eigen::MatrixXd Rn = Eigen::MatrixXd::Identity(3, 3) * orientation_z_reg_noise_;
    Eigen::MatrixXd Ht = He.transpose();
    Eigen::MatrixXd S = He * P_ * Ht + Rn;
    Eigen::MatrixXd K = P_ * Ht * S.inverse();
    Eigen::VectorXd dx = K * (-e);
    applyCorrection(dx);
    Eigen::Matrix<double, 9, 9> KH = Eigen::Matrix<double, 9, 9>(K * He);
    P_ = (Eigen::Matrix<double, 9, 9>::Identity() - KH) * P_;
}

void OutpostESEKF::enumerateAssignments(const std::vector<std::vector<cv::Point2f>>& p2_list,
                                 const cv::Vec3d& position, const cv::Mat& R,
                                 size_t depth,
                                 std::vector<size_t>& assignment, std::vector<bool>& used,
                                 std::vector<size_t>& best_assignment, double& best_cost) const {
    const size_t M = p2_list.size();
    const size_t N = points_3d_list_.size();
    if (depth == M) {
        double cost = assignmentCost(p2_list, assignment, position, R);
        if (cost < best_cost) {
            best_cost = cost;
            best_assignment = assignment;
        }
        return;
    }
    for (size_t j = 0; j < N; ++j) {
        if (used[j]) continue;
        used[j] = true;
        assignment[depth] = j;
        enumerateAssignments(p2_list, position, R, depth + 1,
                             assignment, used, best_assignment, best_cost);
        used[j] = false;
    }
}

std::vector<cv::Point3f> OutpostESEKF::effectiveLocalPoints(size_t obj_idx, double dz) const {
    std::vector<cv::Point3f> pts = points_3d_list_[obj_idx];
    if (dz != 0.0) {
        for (auto& p : pts) p.z += (float)dz;
    }
    return pts;
}

std::vector<cv::Point2f> OutpostESEKF::buildConcat2D(const std::vector<std::vector<cv::Point2f>>& points_2d_list) const {
    std::vector<cv::Point2f> c2;
    for (const auto& p : points_2d_list) c2.insert(c2.end(), p.begin(), p.end());
    return c2;
}

std::vector<cv::Point3f> OutpostESEKF::buildConcat3D(const std::vector<size_t>& assignment,
                                              double dz2, double dz3) const {
    std::vector<cv::Point3f> c3;
    for (size_t i = 0; i < assignment.size(); ++i) {
        size_t obj = assignment[i];
        double dz = 0.0;
        if (obj == 1) dz = dz2;
        else if (obj == 2) dz = dz3;
        std::vector<cv::Point3f> pts = effectiveLocalPoints(obj, dz);
        c3.insert(c3.end(), pts.begin(), pts.end());
    }
    return c3;
}

double OutpostESEKF::costForDzPair(const std::vector<std::vector<cv::Point2f>>& p2_list,
                            const std::vector<size_t>& assignment,
                            const cv::Vec3d& position, const cv::Mat& R,
                            double dz2, double dz3) const {
    std::vector<double> r = computeError(position, R, p2_list, assignment, dz2, dz3);
    double cost = 0.0;
    for (double v : r) cost += v * v;
    return cost;
}

double OutpostESEKF::optimizeDz(const std::vector<std::vector<cv::Point2f>>& p2_list,
                         const std::vector<size_t>& assignment,
                         const cv::Vec3d& position, const cv::Mat& R,
                         size_t obj_idx) const {
    double lo = -dz_search_range_;
    double hi = dz_search_range_;
    const double gr = (std::sqrt(5.0) - 1.0) / 2.0;
    double c = hi - gr * (hi - lo);
    double d = lo + gr * (hi - lo);
    auto eval = [&](double dz) -> double {
        double dz2 = (obj_idx == 1) ? dz : 0.0;
        double dz3 = (obj_idx == 2) ? dz : 0.0;
        return costForDzPair(p2_list, assignment, position, R, dz2, dz3);
    };
    double fc = eval(c);
    double fd = eval(d);
    while (std::fabs(hi - lo) > 1e-4) {
        if (fc < fd) {
            hi = d; d = c; fd = fc;
            c = hi - gr * (hi - lo);
            fc = eval(c);
        } else {
            lo = c; c = d; fc = fd;
            d = lo + gr * (hi - lo);
            fd = eval(d);
        }
    }
    return std::clamp(0.5 * (lo + hi), -dz_limit_, dz_limit_);
}

double OutpostESEKF::assignmentCost(const std::vector<std::vector<cv::Point2f>>& p2_list,
                             const std::vector<size_t>& assignment,
                             const cv::Vec3d& position, const cv::Mat& R) const {
    bool has1 = false, has2 = false;
    for (size_t i = 0; i < assignment.size(); ++i) {
        if (assignment[i] == 1) has1 = true;
        else if (assignment[i] == 2) has2 = true;
    }
    double dz2 = 0.0, dz3 = 0.0;
    if (has1) dz2 = optimizeDz(p2_list, assignment, position, R, 1);
    if (has2) dz3 = optimizeDz(p2_list, assignment, position, R, 2);
    return costForDzPair(p2_list, assignment, position, R, dz2, dz3);
}

std::vector<double> OutpostESEKF::computeErrorCore(const cv::Vec3d& position,
                                            const cv::Mat& rotation_matrix,
                                            const std::vector<cv::Point2f>& c2,
                                            const std::vector<cv::Point3f>& c3) const {
    const size_t n_point = c2.size();
    CV_Assert(n_point == c3.size());
    cv::Mat R;
    if (rotation_matrix.type() == CV_64F) {
        R = rotation_matrix;
    } else {
        rotation_matrix.convertTo(R, CV_64F);
    }
    std::vector<cv::Point3f> world_points = localToWorld(position, R, c3);
    std::vector<cv::Point3f> cam_points;
    cam_points.reserve(n_point);
    for (const auto& wp : world_points) {
        cv::Vec3f p = tf_tree_->transformPoint(RobotTfTree::WORLD, RobotTfTree::CAMERA, cv::Vec3f(wp.x, wp.y, wp.z));
        cam_points.emplace_back(p[0], p[1], p[2]);
    }
    std::vector<cv::Point2f> projections;
    camera_proj_->projectPoints_Cam(cam_points, projections);
    std::vector<double> residuals;
    residuals.reserve(n_point * 2);
    for (size_t i = 0; i < n_point; ++i) {
        residuals.push_back((double)c2[i].x - (double)projections[i].x);
        residuals.push_back((double)c2[i].y - (double)projections[i].y);
    }
    return residuals;
}

Eigen::Quaterniond OutpostESEKF::rotationMatrixToQuaternion(const cv::Mat& R) {
    CV_Assert(!R.empty() && R.rows == 3 && R.cols == 3);
    cv::Mat R64;
    if (R.type() == CV_64F) {
        R64 = R;
    } else {
        R.convertTo(R64, CV_64F);
    }
    Eigen::Matrix3d Re;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Re(i, j) = R64.at<double>(i, j);
    Eigen::Quaterniond q(Re);
    q.normalize();
    return q;
}

cv::Vec3d OutpostESEKF::quaternionToRotationVector(const Eigen::Quaterniond& q) {
    Eigen::Quaterniond qn = q.normalized();
    Eigen::AngleAxisd aa(qn);
    Eigen::Vector3d rv = aa.angle() * aa.axis();
    return cv::Vec3d(rv[0], rv[1], rv[2]);
}

Eigen::Quaterniond OutpostESEKF::rotationVectorToQuaternion(const cv::Vec3d& rotvec) {
    double angle = std::sqrt(rotvec[0] * rotvec[0] +
                             rotvec[1] * rotvec[1] +
                             rotvec[2] * rotvec[2]);
    if (angle < 1e-12)
        return Eigen::Quaterniond::Identity();
    Eigen::Vector3d axis(rotvec[0], rotvec[1], rotvec[2]);
    axis /= angle;
    return Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
}

cv::Mat OutpostESEKF::quaternionToRotationMatrix(const Eigen::Quaterniond& q) {
    Eigen::Matrix3d Re = q.normalized().toRotationMatrix();
    cv::Mat R(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R.at<double>(i, j) = Re(i, j);
    return R;
}

cv::Mat OutpostESEKF::rotationVectorToRotationMatrix(const cv::Vec3d& rotvec) {
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << rotvec[0], rotvec[1], rotvec[2]);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    return R;
}

std::vector<cv::Point3f> OutpostESEKF::localToWorld(const cv::Vec3d& position, const cv::Mat& R,
                                             const std::vector<cv::Point3f>& local_points) {
    std::vector<cv::Point3f> world;
    world.reserve(local_points.size());
    for (const auto& p : local_points) {
        cv::Mat v = (cv::Mat_<double>(3, 1) << (double)p.x, (double)p.y, (double)p.z);
        cv::Mat w = R * v;
        world.emplace_back((float)(w.at<double>(0, 0) + position[0]),
                           (float)(w.at<double>(1, 0) + position[1]),
                           (float)(w.at<double>(2, 0) + position[2]));
    }
    return world;
}
