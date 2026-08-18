// PredictedBallisticSolver.cpp — 预测弹道解算器实现
#include "Ballistic/PredictedBallisticSolver.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "RobotConfig.h"

PredictedBallisticSolver::PredictedBallisticSolver(std::shared_ptr<GimbalSolver> gimbal)
    : gimbal_(gimbal),
      max_iterations_(std::max(1, RobotConfig::instance().predictedBallistic.maxIterations)),
      time_error_tolerance_(RobotConfig::instance().predictedBallistic.timeErrorTolerance) {}

PredictedBallisticSolver::Result PredictedBallisticSolver::solve(const Predictor& predictor,
                                                                 double extra_predict_time) const {
    Result final_result;
    if (!gimbal_) return final_result;

    // 当前时刻的目标点列表：提供目标点数量与首轮距离预估所需的当前坐标
    const std::vector<cv::Point3f> centers_now = predictor(0.0);
    if (centers_now.empty()) return final_result;

    const cv::Vec3f muzzle = gimbal_->muzzleWorldOrigin();
    const double v0 = gimbal_->bulletVelocity();
    if (v0 <= 0.0) return final_result;

    double best_muzzle_dist = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < centers_now.size(); ++i) {
        double flight_time = 0.0;
        double best_time_err = std::numeric_limits<double>::infinity();
        Result candidate;   // 该目标点时间误差最小的一次迭代结果

        for (int iter = 0; iter < max_iterations_; ++iter) {
            if (iter == 0) {
                // 首次迭代：当前 muzzle 原点(world) 到目标点的欧氏距离 / 弹速
                const cv::Vec3f target_now(centers_now[i].x, centers_now[i].y, centers_now[i].z);
                flight_time = cv::norm(muzzle - target_now) / v0;
            }

            // 预测时间 = 额外预测时间(传入) + 飞行时间；取预测列表中第 i 个点作为预测目标点
            const double pred_t = extra_predict_time + flight_time;
            const std::vector<cv::Point3f> pred_list = predictor(pred_t);
            if (i >= pred_list.size()) break;   // 防御：预测列表长度与当前不一致

            const cv::Point3f& pp = pred_list[i];
            const cv::Vec3f pred_point(pp.x, pp.y, pp.z);

            const GimbalSolver::AimResult aim = gimbal_->solveAim(pred_point);

            // 时间误差：本次用于预测的飞行时间 与 弹道实际飞行时间 之差
            const double time_err = std::fabs(flight_time - aim.flight_time);
            if (time_err < best_time_err) {
                best_time_err = time_err;
                candidate.success         = aim.success;
                candidate.predicted_point = pred_point;
                candidate.predict_time    = pred_t;
                candidate.gimbal          = aim;
            }

            flight_time = aim.flight_time;   // 下次迭代使用本次解算的飞行时间
            if (time_err <= time_error_tolerance_) break;  // 提前停止
        }

        // 跨目标点：选取预测点距离当前 muzzle 系原点最近者作为实际使用目标
        const double d = cv::norm(muzzle - candidate.predicted_point);
        if (d < best_muzzle_dist) {
            best_muzzle_dist = d;
            final_result = candidate;
        }
    }

    return final_result;
}
