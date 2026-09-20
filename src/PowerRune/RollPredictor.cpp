#include "PowerRune/RollPredictor.h"
#include "common/TransformTree/CoordinateTransform.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <mutex>
#include <ceres/ceres.h>

RollPredictor::RollPredictor(const Params& params)
    : max_time_window_(params.max_time_window)
    , queue_time_threshold_(std::min(params.queue_time_threshold, params.max_time_window))
    , min_data_points_(params.min_data_points)
    , big_range_strict_(params.strict)
    , big_range_loose_(params.loose)
    , loose_fit_(params.loose_fit)
    , big_linear_coefficient_(params.big_linear_coefficient)
    , big_omega_step_(params.big_omega_step)
    , big_ot_step_(params.big_ot_step)
    , small_slope_fixed_(params.small_slope_fixed)
    , unwrap_threshold_rad_(params.unwrap_threshold_rad)
    , correction_window_(std::max(1, std::min(params.correction_window, params.min_data_points)))
    , grid_search_interval_(params.grid_search_interval)
    , visualization_samples_(params.visualization_samples)
    , ceres_max_iterations_(params.ceres_max_iterations)
    , ceres_function_tolerance_(params.ceres_function_tolerance)
{
    resetState();
}

void RollPredictor::resetState()
{
    queue_.clear();
    big_params_   = BigParams();
    small_params_ = SmallParams();
    fit_valid_    = false;
    fit_method_   = FitMethod::BIG;
    correction_bias_ = 0.0f;
    first_update_ = true;
    continuous_roll_ = 0.0f;
    last_signed_roll_   = 0.0f;
    jump_count_          = 0;
    last_update_timestamp_ = std::chrono::steady_clock::time_point();
    last_grid_search_timestamp_ = std::chrono::steady_clock::time_point();
    posi_ = cv::Vec3f();
    y_axis_R_ = cv::Mat();
}

void RollPredictor::reset()
{
    if (!first_update_ && queue_.empty() && !fit_valid_) {
        return;  // 已经处于重置状态
    }
    resetState();
}

void RollPredictor::update(float roll_raw,
                           const std::chrono::steady_clock::time_point& timestamp,
                           cv::Vec3f posi,
                           cv::Mat y_axis_R)
{
    posi_ = posi;
    y_axis_R_ = y_axis_R;

    // 应用方向符号：反转时将 roll 取负，使拟合模型保持一致
    float signed_roll = roll_raw * static_cast<float>(direction_);

    // ---- 相位连续化 (unwrap)，仿照 YAxisFilter 的 jump 机制 ----
    if (!first_update_) {
        // 检测跨 ±π 边界跳变：roll 从负半周期跳到正半周期 → 逆跳
        //                       roll 从正半周期跳到负半周期 → 正跳
        if (last_signed_roll_ < -unwrap_threshold_rad_ &&
            signed_roll > unwrap_threshold_rad_) {
            jump_count_ -= 1;
        } else if (last_signed_roll_ > unwrap_threshold_rad_ &&
                   signed_roll < -unwrap_threshold_rad_) {
            jump_count_ += 1;
        }
    } else {
        first_update_ = false;
    }
    last_signed_roll_ = signed_roll;

    // 连续化值 = 带方向符号的原始值 + 累计整圈偏移
    constexpr float TWO_PI = 2.0f * static_cast<float>(M_PI);
    continuous_roll_ = signed_roll + static_cast<float>(jump_count_) * TWO_PI;

    // ---- 将当前数据点入队 (已带方向符号) ----
    queue_.push_back({continuous_roll_, timestamp});

    // ---- 移除队头过期数据 ----
    while (!queue_.empty()) {
        float dt = std::chrono::duration<float>(timestamp - queue_.front().timestamp).count();
        if (dt > max_time_window_) {
            queue_.pop_front();
        } else {
            break;
        }
    }

    // ---- 执行拟合 ----
    performFit();

    // 记录更新时间戳（在 performFit 之后，以便 performFit 内部计算帧间 dt 时
    // 拿到的是上一帧的时间；updateWithoutData 也是这样在平移 o_t 之后再更新时间戳）
    last_update_timestamp_ = timestamp;
}

void RollPredictor::updateWithoutData(
    const std::chrono::steady_clock::time_point& timestamp)
{
    // 无观测数据时，仅根据时间推移偏移 o_t（保持 t=0 对应当前时刻）
    if (fit_valid_ && last_update_timestamp_.time_since_epoch().count() > 0) {
        float dt = std::chrono::duration<float>(timestamp - last_update_timestamp_).count();
        if (dt > 0.0f) {
            big_params_.o_t += dt;
            small_params_.o_t += dt;
        }
    }

    // 移除队头过期数据
    while (!queue_.empty()) {
        float dt = std::chrono::duration<float>(timestamp - queue_.front().timestamp).count();
        if (dt > max_time_window_) {
            queue_.pop_front();
        } else {
            break;
        }
    }

    // 检查拟合有效判据
    if (!checkFitValidity()) {
        fit_valid_ = false;
    }

    last_update_timestamp_ = timestamp;
}

float RollPredictor::predict_roll(float delta_t) const
{
    if (!fit_valid_) {
        return 0.0f;
    }

    if (fit_method_ == FitMethod::SMALL) {
        // 线性模型: r = k * (t + o_t)（k 默认 π/3，loose_fit 开启时为拟合斜率）
        float tau = delta_t + small_params_.o_t;
        return (small_params_.slope * tau + correction_bias_) * static_cast<float>(direction_);
    }

    // 标准模型: r_signed = -a/ω * cos(ω*(t+o_t)) + (c - a)*(t+o_t)
    // 实际输出乘以 direction_ 恢复符号
    // delta_t 可为任意实数值（正值=未来，负值=过去）
    float tau = delta_t + big_params_.o_t;
    float cos_term = -big_params_.a / big_params_.omega * std::cos(big_params_.omega * tau);
    float linear_term =
        (static_cast<float>(big_linear_coefficient_) - big_params_.a) * tau;
    return (cos_term + linear_term + correction_bias_) * static_cast<float>(direction_);
}

std::pair<cv::Vec3f, cv::Mat> RollPredictor::predict_posi_and_R(float delta_t) const
{
    if (!fit_valid_) {
        return {cv::Vec3f(0.0f, 0.0f, 0.0f), cv::Mat::eye(3, 3, CV_32FC1)};
    }
    return {
        posi_,
        y_axis_R_ * CoordinateTransform::eulerToRotationMatrix({0.0f, 0.0f, predict_roll(delta_t)})
    };
}

float RollPredictor::computeRMSE() const
{
    if (!fit_valid_ || queue_.size() < 2) {
        return std::numeric_limits<float>::infinity();
    }

    const auto now = last_update_timestamp_;
    double sum_sq = 0.0;
    const size_t N = queue_.size();

    for (size_t i = 0; i < N; ++i) {
        float dt = std::chrono::duration<float>(
            queue_[i].timestamp - now).count();
        float predicted = predict_roll(dt) * static_cast<float>(direction_);
        float err = queue_[i].continuous_roll - predicted;
        sum_sq += static_cast<double>(err) * static_cast<double>(err);
    }

    return static_cast<float>(std::sqrt(sum_sq / static_cast<double>(N)));
}

std::unique_ptr<std::function<std::pair<cv::Vec3f, cv::Mat>(float)>> RollPredictor::capturePredictor() const
{
    // 捕获调用时刻的所有状态：基本类型按值，cv::Mat 深度克隆，cv::Vec3f 按值
    float a     = big_params_.a;
    float omega = big_params_.omega;
    float ot    = big_params_.o_t;
    int   dir   = direction_;
    bool  valid = fit_valid_;
    FitMethod method = fit_method_;
    float ot_small  = small_params_.o_t;
    float slope_small = small_params_.slope;
    float bias      = correction_bias_;
    cv::Vec3f pos   = posi_;
    cv::Mat   y_axis_R = y_axis_R_.clone();  // 深度复制旋转矩阵
    const float lin = static_cast<float>(big_linear_coefficient_);  // BIG 线性项系数

    auto func = [a, omega, ot, dir, valid, method, ot_small, slope_small, bias, pos, y_axis_R, lin](float delta_t) -> std::pair<cv::Vec3f, cv::Mat> {
        if (!valid) {
            return {cv::Vec3f(0.0f, 0.0f, 0.0f), cv::Mat::eye(3, 3, CV_32FC1)};
        }
        float predicted_roll;
        if (method == FitMethod::SMALL) {
            float tau = delta_t + ot_small;
            predicted_roll = (slope_small * tau + bias) * static_cast<float>(dir);
        } else {
            // 与 predict_roll 完全一致的逻辑
            float tau = delta_t + ot;
            float cos_term    = -a / omega * std::cos(omega * tau);
            float linear_term = (lin - a) * tau;
            predicted_roll = (cos_term + linear_term + bias) * static_cast<float>(dir);
        }

        cv::Mat roll_R = CoordinateTransform::eulerToRotationMatrix({0.0f, 0.0f, predicted_roll});
        cv::Mat predicted_R = y_axis_R * roll_R;

        return {pos, predicted_R};
    };

    return std::make_unique<std::function<std::pair<cv::Vec3f, cv::Mat>(float)>>(std::move(func));
}

void RollPredictor::getVisualizationPoints(
    std::vector<std::pair<float, float>>& fitted_curve,
    std::vector<std::pair<float, float>>& raw_points,
    int num_samples) const
{
    fitted_curve.clear();
    raw_points.clear();

    if (!fit_valid_ || queue_.empty()) {
        return;
    }

    // 最新更新时间戳作为 t=0
    const auto now = last_update_timestamp_;

    // ---- 处理原始观测点：t = 该点到最新数据的时间差（负数） ----
    for (const auto& dp : queue_) {
        float t = -std::chrono::duration<float>(now - dp.timestamp).count();
        raw_points.emplace_back(t, dp.continuous_roll * static_cast<float>(direction_));
    }

    // ---- 生成拟合曲线：从 t_min（最旧的 t）到 t=0 ----
    float t_min = raw_points.front().first;  // 最负的值

    if (num_samples < 2) num_samples = 2;
    float step = -t_min / static_cast<float>(num_samples - 1);

    for (int i = 0; i < num_samples; ++i) {
        float t = t_min + step * static_cast<float>(i);
        float r = predict_roll(t);
        fitted_curve.emplace_back(t, r);
    }
}

// Ceres 残差计算仿函数（AutoDiff 自动求导）
// 模型: r_pred = -a/ω * cos(ω*(t+o_t)) + (c - a)*(t+o_t)   （c = big_linear_coefficient）
// 残差: res = r_obs - r_pred
struct RollResidual {
    double t;
    double r_obs;
    double linear_coeff;   // BIG 模型线性项系数（配置 big_linear_coefficient）

    RollResidual(double ti, double ri, double coeff)
        : t(ti), r_obs(ri), linear_coeff(coeff) {}

    template <typename T>
    bool operator()(const T* const a, const T* const omega, const T* const ot, T* residual) const {
        T tau         = T(t) + ot[0];
        T cos_term    = -a[0] / omega[0] * ceres::cos(omega[0] * tau);
        T linear_term = (T(linear_coeff) - a[0]) * tau;
        residual[0]   = r_obs - (cos_term + linear_term);
        return true;
    }
};

bool RollPredictor::checkFitValidity() const
{
    if (queue_.size() < static_cast<size_t>(min_data_points_)) {
        return false;
    }
    // 队列时间跨度判据：最新点与最旧点的时间差必须 >= queue_time_threshold_
    auto time_span = std::chrono::duration<float>(
        queue_.back().timestamp - queue_.front().timestamp).count();
    if (time_span < queue_time_threshold_) {
        return false;
    }
    return true;
}

void RollPredictor::performFit()
{
    if (!checkFitValidity()) {
        fit_valid_ = false;
        return;
    }

    // ============================================================
    // 1. 数据准备
    // ============================================================
    const auto now = queue_.back().timestamp;
    const size_t N = queue_.size();
    std::vector<double> t_values(N);
    std::vector<double> r_values(N);

    for (size_t i = 0; i < N; ++i) {
        t_values[i] = -std::chrono::duration<double>(now - queue_[i].timestamp).count();
        r_values[i] = static_cast<double>(queue_[i].continuous_roll);
    }

    // 计算 o_t 范围 + SMALL 预计算
    //
    // BIG 模型取值范围：loose_fit_ = true 时用 Params::loose 一组，
    // false 时用 Params::strict 一组（均来自 config power_rune.roll_predictor）。
    const Params::BigRange& big_range = activeBigRange();
    const double A_MAX     = big_range.a_max;
    const double A_MIN     = big_range.a_min;
    const double OMEGA_MIN = big_range.omega_min;
    const double OMEGA_MAX = big_range.omega_max;
    // BIG 模型线性项斜率为 s = c - a（c = big_linear_coefficient_），用 a 的上下界
    // 推导 o_t 的搜索范围：
    //   o = r / s - t，s 越大 o 越小，故 o 的下界用 s 的上界（对应 A_MIN），
    //   o 的上界用 s 的下界（对应 A_MAX）。
    const double SLOPE_MIN = big_linear_coefficient_ - A_MIN;
    const double SLOPE_MAX = big_linear_coefficient_ - A_MAX;
    const double PERIOD_MARGIN = M_PI / OMEGA_MIN;

    double ot_lower = std::numeric_limits<double>::max();
    double ot_upper = std::numeric_limits<double>::lowest();
    double small_sum_t = 0.0, small_sum_r = 0.0;

    for (size_t i = 0; i < N; ++i) {
        double o_i_min = r_values[i] / SLOPE_MIN - t_values[i];
        double o_i_max = r_values[i] / SLOPE_MAX - t_values[i];
        if (o_i_min < ot_lower) ot_lower = o_i_min;
        if (o_i_max > ot_upper) ot_upper = o_i_max;
        small_sum_t += t_values[i];
        small_sum_r += r_values[i];
    }
    ot_lower -= PERIOD_MARGIN;
    ot_upper += PERIOD_MARGIN;

    // 网格搜索步长（配置）
    const double OMEGA_STEP = big_omega_step_;
    const double OT_STEP    = big_ot_step_;

    // ============================================================
    // 2. SMALL 模型（总是计算，代价极低）
    // ============================================================
    const double small_mean_t = small_sum_t / static_cast<double>(N);
    const double small_mean_r = small_sum_r / static_cast<double>(N);

    // 斜率固定 small_slope_fixed_ 时的闭式解（loose_fit_ = false，或退化时回退使用）
    auto small_ot_fixed_slope = [&]() {
        return (1.0 / small_slope_fixed_) * small_mean_r - small_mean_t;
    };

    double small_slope = small_slope_fixed_;
    double small_o_t   = 0.0;
    if (loose_fit_) {
        // 同时拟合斜率 k 与截距 c：r = k*t + c = k*(t + c/k)，故 o_t = c/k。
        double sxx = 0.0, sxy = 0.0;
        for (size_t i = 0; i < N; ++i) {
            const double dt = t_values[i] - small_mean_t;
            sxx += dt * dt;
            sxy += dt * (r_values[i] - small_mean_r);
        }
        if (sxx > 1e-12) {
            const double k = sxy / sxx;   // 最小二乘拟合斜率
            if (std::abs(k) > 1e-6) {
                const double c = small_mean_r - k * small_mean_t;  // 拟合截距
                small_slope = k;
                small_o_t   = c / k;
            } else {
                // 斜率退化到 0 附近：回退到固定斜率解，避免 o_t 爆炸
                small_o_t = small_ot_fixed_slope();
            }
        } else {
            small_o_t = small_ot_fixed_slope();
        }
    } else {
        small_o_t = small_ot_fixed_slope();
    }

    double small_sum_sq = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const double r_pred = small_slope * (t_values[i] + small_o_t);
        const double err = r_values[i] - r_pred;
        small_sum_sq += err * err;
    }
    const double small_rmse = std::sqrt(small_sum_sq / static_cast<double>(N));

    // ============================================================
    // 3. 判断是否需要完整网格搜索
    // ============================================================
    bool do_grid_search = !fit_valid_;
    if (fit_valid_ && grid_search_interval_ > 0.0f) {
        float elapsed = std::chrono::duration<float>(
            now - last_grid_search_timestamp_).count();
        if (elapsed >= grid_search_interval_) {
            do_grid_search = true;
        }
    }

    double big_final_a     = 0.0;
    double big_final_omega = 0.0;
    double big_final_ot    = 0.0;
    double big_refined_rmse = std::numeric_limits<double>::max();
    bool big_found = false;

    if (do_grid_search) {
        // ============================================================
        // 3a. BIG 网格搜索 — 通过持久线程池并行执行
        // ============================================================
        int num_omega = std::max(1,
            static_cast<int>((OMEGA_MAX - OMEGA_MIN) / OMEGA_STEP + 1.5));
        int num_ot    = std::max(1,
            static_cast<int>((ot_upper - ot_lower) / OT_STEP + 1.5));
        const int total_tasks = num_omega * num_ot;

        struct GridBest {
            double a = 0, omega = 0, ot = 0, rmse = std::numeric_limits<double>::max();
        };
        GridBest global_best;
        std::mutex best_mutex;

        // 提交网格搜索任务到持久线程池
        fit_pool_.submit(total_tasks, [&](int idx) {
            const int io = idx / num_ot;
            const int it = idx % num_ot;
            const double omega = OMEGA_MIN + io * OMEGA_STEP;
            const double ot    = ot_lower  + it * OT_STEP;

            // 闭式解 a
            double sum_fy = 0.0, sum_ff = 0.0;
            for (size_t i = 0; i < N; ++i) {
                const double tau = t_values[i] + ot;
                const double fi  = -std::cos(omega * tau) / omega - tau;
                const double yi  = r_values[i] - big_linear_coefficient_ * tau;
                sum_fy += fi * yi;
                sum_ff += fi * fi;
            }
            if (sum_ff < 1e-12) return;

            double a = sum_fy / sum_ff;
            a = std::max(A_MIN, std::min(A_MAX, a));

            // RMSE
            double sum_sq = 0.0;
            for (size_t i = 0; i < N; ++i) {
                const double tau = t_values[i] + ot;
                const double r_pred = -a / omega * std::cos(omega * tau)
                                    + (big_linear_coefficient_ - a) * tau;
                const double err = r_values[i] - r_pred;
                sum_sq += err * err;
            }
            const double rmse = std::sqrt(sum_sq / static_cast<double>(N));

            // 合并到全局最优
            {
                std::lock_guard<std::mutex> lock(best_mutex);
                if (rmse < global_best.rmse) {
                    global_best.rmse  = rmse;
                    global_best.a     = a;
                    global_best.omega = omega;
                    global_best.ot    = ot;
                }
            }
        });

        // 等待网格搜索完成
        fit_pool_.wait();

        big_found = (global_best.rmse < std::numeric_limits<double>::max());

        // ============================================================
        // 3b. Ceres 精化（从网格搜索最优出发）
        // ============================================================
        if (big_found) {
            const double big_best_a     = global_best.a;
            const double big_best_omega = global_best.omega;
            const double big_best_ot    = global_best.ot;

            double omega_lower_ref = std::max(OMEGA_MIN, big_best_omega - OMEGA_STEP * 2);
            double omega_upper_ref = std::min(OMEGA_MAX, big_best_omega + OMEGA_STEP * 2);
            double ot_lower_ref    = std::max(ot_lower,  big_best_ot - OT_STEP * 2);
            double ot_upper_ref    = std::min(ot_upper,  big_best_ot + OT_STEP * 2);

            ceres::Problem problem;
            double a_ref     = big_best_a;
            double omega_ref = big_best_omega;
            double ot_ref    = big_best_ot;

            for (size_t i = 0; i < N; ++i) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<RollResidual, 1, 1, 1, 1>(
                        new RollResidual(t_values[i], r_values[i], big_linear_coefficient_)),
                    nullptr, &a_ref, &omega_ref, &ot_ref);
            }
            problem.SetParameterLowerBound(&a_ref, 0, A_MIN);
            problem.SetParameterUpperBound(&a_ref, 0, A_MAX);
            problem.SetParameterLowerBound(&omega_ref, 0, omega_lower_ref);
            problem.SetParameterUpperBound(&omega_ref, 0, omega_upper_ref);
            problem.SetParameterLowerBound(&ot_ref, 0, ot_lower_ref);
            problem.SetParameterUpperBound(&ot_ref, 0, ot_upper_ref);

            ceres::Solver::Options options;
            options.linear_solver_type          = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = false;
            options.max_num_iterations          = ceres_max_iterations_;
            options.function_tolerance          = ceres_function_tolerance_;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            big_final_a     = a_ref;
            big_final_omega = omega_ref;
            big_final_ot    = ot_ref;
            if (!(summary.termination_type == ceres::CONVERGENCE ||
                  summary.termination_type == ceres::NO_CONVERGENCE)) {
                big_final_a     = big_best_a;
                big_final_omega = big_best_omega;
                big_final_ot    = big_best_ot;
            }

            // BIG RMSE
            double sum_sq = 0.0;
            for (size_t i = 0; i < N; ++i) {
                const double tau = t_values[i] + big_final_ot;
                const double r_pred = -big_final_a / big_final_omega
                                    * std::cos(big_final_omega * tau)
                                    + (big_linear_coefficient_ - big_final_a) * tau;
                const double err = r_values[i] - r_pred;
                sum_sq += err * err;
            }
            big_refined_rmse = std::sqrt(sum_sq / static_cast<double>(N));
        }

        // 记录本次网格搜索时间
        last_grid_search_timestamp_ = now;

    } else {
        // ============================================================
        // 3c. Warm‑start Ceres：从上一帧优化结果出发，仅做局部精化
        //     o_t 按时间差平移（无界），a / omega 在全局范围内限位
        // ============================================================
        // 用 last_update_timestamp_ 计算帧间 dt：它要么是上一帧 update() 完成后的时间，
        // 要么是 updateWithoutData 平移 o_t 后记录的时间，因此与 big_params_.o_t 的时基一致
        float elapsed = 0.0f;
        if (last_update_timestamp_.time_since_epoch().count() > 0) {
            elapsed = std::chrono::duration<float>(
                now - last_update_timestamp_).count();
        }
        double warm_ot = static_cast<double>(big_params_.o_t) + elapsed;

        ceres::Problem problem;
        double a_ref     = static_cast<double>(big_params_.a);
        double omega_ref = static_cast<double>(big_params_.omega);
        double ot_ref    = warm_ot;  // o_t 无界，不给上下界约束

        for (size_t i = 0; i < N; ++i) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<RollResidual, 1, 1, 1, 1>(
                    new RollResidual(t_values[i], r_values[i], big_linear_coefficient_)),
                nullptr, &a_ref, &omega_ref, &ot_ref);
        }
        // a / omega 保持全局范围限位；o_t 不给约束（无界）
        problem.SetParameterLowerBound(&a_ref, 0, A_MIN);
        problem.SetParameterUpperBound(&a_ref, 0, A_MAX);
        problem.SetParameterLowerBound(&omega_ref, 0, OMEGA_MIN);
        problem.SetParameterUpperBound(&omega_ref, 0, OMEGA_MAX);
        // 注意：不对 ot_ref 设置上下界，因为 o_t 是无界的

        ceres::Solver::Options options;
        options.linear_solver_type          = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations          = ceres_max_iterations_;
        options.function_tolerance          = ceres_function_tolerance_;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (summary.termination_type == ceres::CONVERGENCE ||
            summary.termination_type == ceres::NO_CONVERGENCE) {
            big_final_a     = a_ref;
            big_final_omega = omega_ref;
            big_final_ot    = ot_ref;
        } else {
            // 不收敛则回退到平移后的参数
            big_final_a     = static_cast<double>(big_params_.a);
            big_final_omega = static_cast<double>(big_params_.omega);
            big_final_ot    = warm_ot;
        }

        // BIG RMSE
        double sum_sq = 0.0;
        for (size_t i = 0; i < N; ++i) {
            const double tau = t_values[i] + big_final_ot;
            const double r_pred = -big_final_a / big_final_omega
                                * std::cos(big_final_omega * tau)
                                + (big_linear_coefficient_ - big_final_a) * tau;
            const double err = r_values[i] - r_pred;
            sum_sq += err * err;
        }
        big_refined_rmse = std::sqrt(sum_sq / static_cast<double>(N));
        big_found = true;
    }

    // ============================================================
    // 4. 模型选择
    // ============================================================
    if (!big_found || small_rmse < big_refined_rmse) {
        fit_method_         = FitMethod::SMALL;
        // 非宽松时写回固定的 π/3（float 表达，与旧实现一致）
        small_params_.slope = loose_fit_ ? static_cast<float>(small_slope)
                                            : static_cast<float>(small_slope_fixed_);
        small_params_.o_t   = static_cast<float>(small_o_t);
        fit_valid_          = true;
    } else {
        fit_method_      = FitMethod::BIG;
        big_params_.a    = static_cast<float>(big_final_a);
        big_params_.omega = static_cast<float>(big_final_omega);
        big_params_.o_t  = static_cast<float>(big_final_ot);
        fit_valid_       = true;
    }

    // ============================================================
    // 5. 修正偏置
    // ============================================================
    int n = std::min(correction_window_, static_cast<int>(N));
    std::vector<double> biases;
    biases.reserve(n);

    for (size_t i = N - n; i < N; ++i) {
        double pred;
        if (fit_method_ == FitMethod::SMALL) {
            // 与 predict_roll 使用同一份参数（float 舍入后的 o_t），保证偏置一致
            pred = small_slope * (t_values[i] + small_params_.o_t);
        } else {
            const double tau = t_values[i] + big_params_.o_t;
            pred = -big_params_.a / big_params_.omega * std::cos(big_params_.omega * tau)
                 + (big_linear_coefficient_ - big_params_.a) * tau;
        }
        biases.push_back(r_values[i] - pred);
    }

    std::sort(biases.begin(), biases.end());
    if (n % 2 == 1) {
        correction_bias_ = static_cast<float>(biases[n / 2]);
    } else {
        correction_bias_ = static_cast<float>((biases[n / 2 - 1] + biases[n / 2]) * 0.5);
    }
}
