// SequencePredictor 四阶段实现，公共接口位于 include/common/Ballistic/SequencePredictor.h。
#include "common/Ballistic/SequencePredictor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "Armor/ArmorModel.h"
#include "common/RobotConfig.h"
#include "common/TransformTree/TfTreeSync.h"

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kGeometryEpsilon = 1e-6;

cv::Vec3f asVec(const cv::Point3f& p) { return {p.x, p.y, p.z}; }
cv::Vec3f asVec(const Eigen::Vector3d& p) {
    return {static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())};
}
bool finitePoint(const cv::Vec3f& p) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}
double horizontalDistance(const cv::Vec3f& a, const cv::Vec3f& b) {
    return std::hypot(static_cast<double>(a[0]) - b[0], static_cast<double>(a[1]) - b[1]);
}
bool finiteItem(const SequencePredictor::Item& item) {
    return finitePoint(item.predicted_point) && std::isfinite(item.predict_time) &&
           std::isfinite(item.yaw) && std::isfinite(item.pitch) &&
           std::isfinite(item.gimbal_yaw) && std::isfinite(item.gimbal_pitch) &&
           std::isfinite(item.flight_time) && item.flight_time > 0.0;
}
} // namespace

SequencePredictor::SequencePredictor() : SequencePredictor(Options{}) {}

SequencePredictor::SequencePredictor(const Options& options)
    : options_(options),
      extra_predict_time_(RobotConfig::instance().common.predictedBallistic.extraPredictTime),
      dt_control_(RobotConfig::instance().common.dtControl()),
      pitch_bias_(RobotConfig::instance().common.predictSequence.pitchBias),
      yaw_bias_(RobotConfig::instance().common.predictSequence.yawBias),
      prediction_points_(RobotConfig::instance().common.predictSequence.predictionPoints),
      interpolation_refine_(RobotConfig::instance().common.predictSequence.interpolationRefine),
      exact_lead_points_(RobotConfig::instance().common.predictSequence.exactLeadPoints),
      aim_stick_ratio_(RobotConfig::instance().common.predictSequence.aimStickRatio) {
    // 这里只校验参数合法性，不把演示参数写入生产配置。
    if (!std::isfinite(options_.sector_w0) || options_.sector_w0 <= 0.0 ||
        !std::isfinite(options_.w_stick) || options_.w_stick <= 0.0 ||
        !std::isfinite(options_.armor_width_override) || options_.armor_width_override < 0.0 ||
        options_.max_prior_probes_per_plate < 1 || options_.boundary_refine_iterations < 0 ||
        !std::isfinite(dt_control_) || dt_control_ <= 0.0 ||
        !std::isfinite(options_.sample_merge_epsilon) || options_.sample_merge_epsilon <= 0.0 ||
        options_.sample_merge_epsilon >= dt_control_ / 2.0 ||
        prediction_points_ < 1 || interpolation_refine_ < 1 || exact_lead_points_ < 0) {
        throw std::invalid_argument("Invalid SequencePredictor options");
    }
    const long long total = (static_cast<long long>(prediction_points_) - 1) *
                            interpolation_refine_ + 1 + exact_lead_points_;
    if (total > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("SequencePredictor horizon is too large");
    }
    for (std::size_t worker = 0; worker < pool_.size(); ++worker) {
        auto gimbal = std::make_shared<GimbalSolver>();
        gimbals_.push_back(gimbal);
        armor_solvers_.emplace_back(gimbal);
        power_rune_solvers_.emplace_back(gimbal);
    }
}

SequencePredictor::Result SequencePredictor::predict(
    const tcs::RobotController::State& state, const Predictor& predictor, const TimePoint& timestamp) {
    return SequenceStageProcessor(snapshotFromSingle(state), predictor, timestamp);
}

SequencePredictor::Result SequencePredictor::predict(
    const bsy::RobotState& state, const Predictor& predictor, const TimePoint& timestamp) {
    return SequenceStageProcessor(snapshotFromBigSmall(state), predictor, timestamp);
}

// ==================== 总调度：两条支路、四个阶段 ====================
SequencePredictor::Result SequencePredictor::SequenceStageProcessor(
    const InputSnapshot& input, const Predictor& predictor, TimePoint timestamp) {
    switch (predictor.source.kind) {
    case Source::Kind::ARMOR: {
        auto prepared = prepareArmor(input, predictor, timestamp);
        if (!prepared) return {};
        const auto solved = solveArmor(*prepared);
        const auto selected = selectArmorTargets(*prepared, solved);
        return generateArmorSequence(*prepared, selected);
    }
    case Source::Kind::POWER_RUNE: {
        auto prepared = preparePowerRune(input, predictor, timestamp);
        if (!prepared) return {};
        const auto solved = solvePowerRune(*prepared);
        const auto selected = selectPowerRuneTarget(*prepared, solved);
        return generatePowerRuneSequence(*prepared, selected);
    }
    default:
        invalidate();
        return {};
    }
}

void SequencePredictor::invalidate() {
    active_source_ = Source{};
    last_first_target_index_ = -1;
    power_rune_selector_.reset();
    // 与旧实现一样，保留大 yaw 的历史几何退路，不把它当作目标选板状态。
}

// ==================== 阶段 1：准备 ====================
std::optional<SequencePredictor::PreparedFrame> SequencePredictor::prepareCommon(
    const InputSnapshot& input, const Predictor& predictor, TimePoint timestamp) {
    if (!predictor.function) {
        invalidate();
        return std::nullopt;
    }
    if (!(active_source_ == predictor.source)) {
        last_first_target_index_ = -1;
        power_rune_selector_.reset();
        active_source_ = predictor.source;
    }

    PreparedFrame frame;
    frame.input = input;
    frame.predictor = predictor;
    frame.timestamp = timestamp;
    frame.predictor_age = std::chrono::duration<double>(timestamp - predictor.timestamp).count();
    frame.base_predict_time = extra_predict_time_ + frame.predictor_age;
    // 已确定：沿用旧版输出长度，几何补点只增加锚点，不延长输出时间窗。
    frame.output_count = (prediction_points_ - 1) * interpolation_refine_ + 1 + exact_lead_points_;
    frame.solve_masked_indices = predictor.masked_indices;
    const auto initial = predictor.function(0.0);
    frame.initial_points = initial.second;
    if (frame.initial_points.empty() || !finitePoint(asVec(initial.first))) return std::nullopt;

    bool any_unmasked = false;
    double radius_scale_sum = 0.0;
    for (std::size_t j = 0; j < frame.initial_points.size(); ++j) {
        any_unmasked = any_unmasked || !predictor.isIndexMasked(static_cast<int>(j));
        const auto point = asVec(frame.initial_points[j]);
        frame.plate_radii.push_back(horizontalDistance(point, asVec(initial.first)));
        radius_scale_sum += cv::norm(point - asVec(initial.first));
    }
    frame.all_aims_masked = !any_unmasked;
    frame.base_stick_delta = aim_stick_ratio_ * radius_scale_sum / frame.initial_points.size();

    // 弹道坐标沿用原实现：底盘位置固定为世界原点，只同步姿态和当前构型关节角。
    for (auto& gimbal : gimbals_) {
        gimbal->setChassisPosition(0.0f, 0.0f, 0.0f);
        applyExtraInputInfoToTree(gimbal->tree(), input.info);
        gimbal->setChassisPosition(0.0f, 0.0f, 0.0f);
        if (input.has_bullet_velocity) gimbal->setBulletVelocity(input.bullet_velocity);
    }
    frame.current_muzzle = gimbals_.front()->muzzleWorldOrigin();
    frame.yaw_world_origin = gimbals_.front()->yawWorldOrigin();

    // 固定点保持原 M/K/n 布局；control_time 只含网格偏移，不重复计入快照延迟。
    for (int i = 0; i < exact_lead_points_; ++i) {
        frame.samples.push_back({(i + 1) * dt_control_, SampleOrigin::FIXED, i});
    }
    for (int j = 0; j < prediction_points_; ++j) {
        const int index = exact_lead_points_ + j * interpolation_refine_;
        frame.samples.push_back({(index + 1) * dt_control_, SampleOrigin::FIXED, index});
    }
    return frame;
}

std::optional<SequencePredictor::PreparedFrame> SequencePredictor::prepareArmor(
    const InputSnapshot& input, const Predictor& predictor, TimePoint timestamp) {
    auto frame = prepareCommon(input, predictor, timestamp);
    if (!frame) return std::nullopt;
    if (frame->all_aims_masked) {
        invalidate();                           // 沿用原 Armor 全屏蔽失效规则
        return std::nullopt;
    }
    // 已确定：omega_valid=false 或 omega 非有限值时，回退到固定采样与距离选板，
    // 不施加旋转扇区、不做几何补点，并关闭选板粘滞，沿用旧版无角速度行为。
    // 这与有效的 w=0 不同：后者仍启用扇区，使用 theta 的低速极限。
    const bool omega_available = predictor.omega_valid && std::isfinite(predictor.target_omega);
    frame->use_sector = omega_available;
    frame->use_sticky = options_.enable_sticky_selection && omega_available &&
                        std::isfinite(frame->base_stick_delta) && frame->base_stick_delta > 0.0;
    if (frame->use_sticky) {
        // 已确定：delta(w)=delta_base/(1+(|w|/w_stick)^2)，正反转使用同一规则。
        // w=0 保留全部粘滞；|w|=w_stick 时减半；高速时趋近纯距离选择。
        // 只降低切换余量，不强制换板，也不放宽候选的扇区硬约束。
        const double ratio = std::fabs(predictor.target_omega) / options_.w_stick;
        frame->stick_delta = frame->base_stick_delta / (1.0 + ratio * ratio);
    }
    // 与 ArmorPipeline 的大小装甲分类保持一致。后续应由上游显式传入几何属性。
    frame->plate_width = options_.armor_width_override > 0.0
        ? options_.armor_width_override
        : ((predictor.source.armor_label == 1 || predictor.source.armor_label == 8)
               ? ArmorModel::BIG_ARMOR_W : ArmorModel::SMALL_ARMOR_W);
    appendGeometrySamples(*frame);
    captureSampleGeometry(*frame);
    return frame;
}

std::optional<SequencePredictor::PreparedFrame> SequencePredictor::preparePowerRune(
    const InputSnapshot& input, const Predictor& predictor, TimePoint timestamp) {
    auto frame = prepareCommon(input, predictor, timestamp);
    if (!frame) return std::nullopt;
    // 仅解除解算 mask，原始 mask 仍交给状态机判断观测状态。
    for (int index : power_rune_selector_.preUpdateCandidateIndices()) {
        auto& mask = frame->solve_masked_indices;
        mask.erase(std::remove(mask.begin(), mask.end(), index), mask.end());
    }
    // PowerRune 不加 Armor 几何补点，也不使用 theta 筛选。
    captureSampleGeometry(*frame);
    return frame;
}

void SequencePredictor::captureSampleGeometry(PreparedFrame& frame) {
    for (const auto& sample : frame.samples) {
        const float yaw_big = frame.input.big_small
            ? yawBigAtTime(frame.input, sample.control_time)
            : std::numeric_limits<float>::quiet_NaN();
        frame.sample_yaw_big.push_back(yaw_big);
        frame.launch_geometries.push_back(gimbals_.front()->captureLaunchGeometry(yaw_big));
    }
    // 全部采样完成几何捕获后再写历史，避免本帧读取到被自己部分覆盖的退路。
    if (frame.input.big_small) {
        std::vector<float> next;
        next.reserve(static_cast<std::size_t>(frame.output_count));
        for (int i = 0; i < frame.output_count; ++i) {
            next.push_back(yawBigAtTime(frame.input, (i + 1) * dt_control_));
        }
        last_yaw_big_seq_ = std::move(next);
    }
}

// ==================== 阶段 2：预测解算（不选板） ====================
SequencePredictor::SolvedSamples SequencePredictor::solveArmor(const PreparedFrame& frame) {
    const std::size_t plates = frame.initial_points.size();
    SolvedSamples solved(frame.samples.size(), std::vector<Candidate>(plates));
    const int workers = static_cast<int>(std::min(gimbals_.size(), plates));
    if (workers == 0) return solved;
    // 每个任务独占一个 solver 槽位；不依赖 thread_local 全局编号及取模复用。
    // 板间并行；同板按合并后的实际采样时间热启动。几何补点同样参与这条链。
    pool_.run_parallel(workers, [&](int worker) {
        for (std::size_t plate = static_cast<std::size_t>(worker); plate < plates; plate += workers) {
            NewtonPredictedBallisticSolver::WarmStart warm_start;
            for (std::size_t u = 0; u < frame.samples.size(); ++u) {
                auto& candidate = solved[u][plate];
                candidate.target_index = static_cast<int>(plate);
                if (frame.predictor.isIndexMasked(candidate.target_index)) {
                    candidate.masked = true;
                    continue;
                }
                if (u > 0 && (frame.launch_geometries[u].yaw_position -
                              frame.launch_geometries[u - 1].yaw_position).squaredNorm() > 0.0) {
                    warm_start.time_derivative.setZero();
                }
                const double launch_time = frame.base_predict_time + frame.samples[u].control_time;
                const auto solution = armor_solvers_[static_cast<std::size_t>(worker)].solveTarget(
                    frame.predictor.function, static_cast<int>(plate), launch_time,
                    frame.launch_geometries[u], &warm_start);
                candidate = solution.result;
                warm_start = solution.warm_start; // 失败后由 solver 返回无效热启动
            }
        }
    });
    return solved;
}

SequencePredictor::SolvedSamples SequencePredictor::solvePowerRune(const PreparedFrame& frame) {
    SolvedSamples solved(frame.samples.size());
    const int workers = static_cast<int>(std::min(gimbals_.size(), frame.samples.size()));
    pool_.run_parallel(workers, [&](int worker) {
        for (std::size_t u = static_cast<std::size_t>(worker); u < frame.samples.size(); u += workers) {
            solved[u] = power_rune_solvers_[static_cast<std::size_t>(worker)].solve(
                frame.predictor.function, frame.base_predict_time + frame.samples[u].control_time,
                frame.sample_yaw_big[u], frame.solve_masked_indices);
        }
    });
    return solved;
}

// ==================== 阶段 3：目标选取 ====================
SequencePredictor::SelectedSamples SequencePredictor::selectArmorTargets(
    const PreparedFrame& frame, const SolvedSamples& solved) {
    SelectedSamples selected;
    selected.reserve(frame.samples.size());
    int sticky_index = frame.use_sticky ? last_first_target_index_ : -1;
    for (std::size_t u = 0; u < frame.samples.size(); ++u) {
        SelectedSample best;
        best.point = frame.samples[u];
        std::optional<SelectedSample> sticky;
        double best_distance = std::numeric_limits<double>::infinity();
        double sticky_distance = best_distance;
        bool any_solved = false;
        const bool record_sector = frame.samples[u].origin == SampleOrigin::FIXED &&
                                   frame.samples[u].fixed_output_index == 0;
        std::vector<SectorCandidateDiagnostic> sector_candidates;
        if (record_sector) sector_candidates.reserve(solved[u].size());
        for (const auto& candidate : solved[u]) {
            // 只记录首个控制时刻，且在每个提前 continue 前保留失败原因。
            SectorCandidateDiagnostic* diagnostic = nullptr;
            const bool masked = candidate.masked || frame.predictor.isIndexMasked(candidate.target_index);
            if (record_sector) {
                sector_candidates.emplace_back();
                diagnostic = &sector_candidates.back();
                diagnostic->target_index = candidate.target_index;
                if (masked) diagnostic->status = SectorCandidateStatus::MASKED;
            }
            if (!usable(candidate) || masked) continue;
            any_solved = true;
            if (diagnostic) diagnostic->status = SectorCandidateStatus::INVALID_GEOMETRY;
            const auto launch = GimbalSolver::evaluateLaunchState(
                frame.launch_geometries[u], candidate.gimbal.yaw, candidate.gimbal.pitch);
            if (!launch.position.allFinite()) continue;
            SelectedSample current;
            current.point = frame.samples[u];
            current.selected = candidate;
            current.launch_muzzle = asVec(launch.position);
            const auto predicted = frame.predictor.function(candidate.predict_time);
            if (candidate.target_index >= static_cast<int>(predicted.second.size()) ||
                !finitePoint(asVec(predicted.first))) continue;
            const auto center = asVec(predicted.first);
            const double radius = horizontalDistance(center, candidate.predicted_point);
            if (!std::isfinite(radius)) continue;
            if (diagnostic) {
                diagnostic->geometry_valid = true;
                diagnostic->center = center;
                diagnostic->point = candidate.predicted_point;
                diagnostic->muzzle = current.launch_muzzle;
                diagnostic->predict_time = candidate.predict_time;
                diagnostic->radius = radius;
                diagnostic->distance = horizontalDistance(center, current.launch_muzzle);
                diagnostic->sector_applied = frame.use_sector && radius > kGeometryEpsilon;
            }
            // 已确定：基地等中心点目标 r≈0 时不使用旋转扇区。
            if (frame.use_sector && radius > kGeometryEpsilon) {
                const double distance = horizontalDistance(center, current.launch_muzzle);
                const auto theta = sectorHalfAngle(frame.predictor.target_omega, distance,
                                                    radius, frame.plate_width, options_.sector_w0);
                if (!theta) continue;
                current.center_angle = centerAngle(center, candidate.predicted_point, current.launch_muzzle);
                current.allowed_half_angle = *theta;
                if (diagnostic) {
                    diagnostic->center_angle = current.center_angle;
                    diagnostic->half_angle = *theta;
                    if (std::isfinite(current.center_angle)) diagnostic->status = SectorCandidateStatus::OUTSIDE;
                }
                if (!std::isfinite(current.center_angle) || std::fabs(current.center_angle) > *theta) continue;
            }
            if (diagnostic) diagnostic->status = SectorCandidateStatus::ELIGIBLE;
            current.status = SampleStatus::SELECTED;
            // 当前采用：按距离最小选取；距离使用该候选自己的发射枪口。
            // 相同距离时保持输入索引顺序；后续粘滞规则另行判断。
            const double distance = cv::norm(candidate.predicted_point - current.launch_muzzle);
            if (distance < best_distance) {
                best_distance = distance;
                best = current;
            }
            if (candidate.target_index == sticky_index) {
                sticky = current;
                sticky_distance = distance;
            }
        }
        // 粘滞只能在已经通过扇区硬约束的候选中生效，不会绕过扇区。
        if (sticky && !(best_distance < sticky_distance - frame.stick_delta)) best = *sticky;
        if (best.status != SampleStatus::SELECTED) {
            best.status = any_solved ? SampleStatus::NO_ELIGIBLE_TARGET : SampleStatus::NO_BALLISTIC_SOLUTION;
            // 已确定：无候选时保留粘滞索引给下一采样时刻，但当前点仍无效。
            // 后续该板须重新通过解算与选板检查，才按衰减后的余量参与粘滞比较。
        } else if (frame.use_sticky) {
            sticky_index = best.selected.target_index;
        }
        // 即使首点没有选中结果，也把候选诊断传给 UI；result.valid=false 不丢失诊断。
        best.sector_candidates = std::move(sector_candidates);
        selected.push_back(std::move(best));
    }
    return selected;
}

SequencePredictor::SelectedSamples SequencePredictor::selectPowerRuneTarget(
    const PreparedFrame& frame, const SolvedSamples& solved) {
    SelectedSamples selected(frame.samples.size());
    const int target = solved.empty() ? -1 : power_rune_selector_.select(
        frame.predictor.masked_indices, solved.front(), frame.timestamp, frame.predictor.timestamp);
    if (frame.all_aims_masked && target < 0) invalidate();
    // 仍有未屏蔽观测点但尚未建立候选时，不重置状态机，保留建立计时。
    for (std::size_t u = 0; u < selected.size(); ++u) {
        auto& result = selected[u];
        result.point = frame.samples[u];
        for (const auto& candidate : solved[u]) {
            if (!usable(candidate)) continue;
            result.status = SampleStatus::NO_ELIGIBLE_TARGET;
            if (candidate.target_index == target) {
                result.selected = candidate;
                result.status = SampleStatus::SELECTED;
                break;
            }
        }
    }
    return selected;
}

// ==================== 阶段 4：序列生成 ====================
SequencePredictor::Result SequencePredictor::generateArmorSequence(
    const PreparedFrame& frame, const SelectedSamples& selected) {
    // 首个输出时刻必须有有效固定精确点；后续失败由合并后的有效锚点补齐参考。
    // 只有首点有效时沿用复制规则，不额外引入两锚点门槛。
    Result result = assembleSequence(frame, selected, false);
    last_first_target_index_ = result.valid
        ? result.items.front().target_index : -1;
    return result;
}

SequencePredictor::Result SequencePredictor::generatePowerRuneSequence(
    const PreparedFrame& frame, const SelectedSamples& selected) {
    // PowerRune 保留所有精确采样点都必须成功的原失败处理。
    Result result = assembleSequence(frame, selected, true);
    for (std::size_t i = 0; i < result.items.size(); ++i) {
        auto& item = result.items[i];
        PredictedPointSelector::TimePoint first_seen;
        if (power_rune_selector_.firstSeenTime(item.target_index, first_seen) && frame.timestamp >= first_seen) {
            item.target_age = std::chrono::duration<double>(frame.timestamp - first_seen).count() +
                              extra_predict_time_ + (i + 1) * dt_control_;
            item.target_age_valid = true;
        }
    }
    last_first_target_index_ = result.valid
        ? result.items.front().target_index : -1;
    return result;
}

SequencePredictor::Result SequencePredictor::assembleSequence(
    const PreparedFrame& frame, const SelectedSamples& selected, bool require_all_samples) const {
    Result result;
    result.samples = selected;
    result.geometry_points_added = frame.geometry_points_added;
    result.prior_budget_limited = frame.prior_budget_limited;
    const bool armor = frame.predictor.source.kind == Source::Kind::ARMOR;
    if (armor) {
        // 已确定：首点是输出下标 0（control_time=dt_control_）的固定精确点，
        // 不是时间排序后找到的第一个成功锚点。后续补点不能向前复制来替代它。
        const auto first = std::find_if(selected.begin(), selected.end(), [&](const auto& sample) {
            return sample.point.origin == SampleOrigin::FIXED &&
                   sample.point.fixed_output_index == 0 &&
                   std::fabs(sample.point.control_time - dt_control_) <= options_.sample_merge_epsilon;
        });
        if (first == selected.end() || first->status != SampleStatus::SELECTED ||
            !usable(first->selected) || !finiteItem(makeItem(frame, first->selected))) return result;
    }
    struct Anchor { double time; Item item; };
    std::vector<Anchor> anchors;
    for (const auto& sample : selected) {
        if (sample.status != SampleStatus::SELECTED || !usable(sample.selected)) continue;
        Item item = makeItem(frame, sample.selected);
        if (!finiteItem(item)) continue;
        if (!anchors.empty()) {
            const auto& previous = anchors.back().item;
            item.yaw = previous.yaw + std::remainder(item.yaw - previous.yaw, 2.0 * kPi);
            item.gimbal_yaw = previous.gimbal_yaw +
                std::remainder(item.gimbal_yaw - previous.gimbal_yaw, 2.0 * kPi);
        }
        anchors.push_back({sample.point.control_time, item});
    }
    if (anchors.empty() || (require_all_samples && anchors.size() != selected.size())) return result;

    auto& sequence = result;
    sequence.items.reserve(static_cast<std::size_t>(frame.output_count));
    result.point_info.reserve(static_cast<std::size_t>(frame.output_count));
    std::size_t right = 0;
    // 所有填充（含尾部外推）止于 output_count * dt_control_，不再额外延长。
    for (int i = 0; i < frame.output_count; ++i) {
        const double time = (i + 1) * dt_control_;
        while (right < anchors.size() && anchors[right].time < time - options_.sample_merge_epsilon) ++right;
        Item item;
        OutputPointInfo info;
        info.control_time = time;
        if (right < anchors.size() && std::fabs(anchors[right].time - time) <= options_.sample_merge_epsilon) {
            item = anchors[right].item;
            info.kind = FillKind::EXACT;
            info.exact_target_valid = true;
            info.left_anchor_time = info.right_anchor_time = anchors[right].time;
        } else if (right == 0) {
            // 防御性检查：正常已由首点约束排除这种情况，禁止复制未来锚点补首段。
            sequence.items.clear();
            result.point_info.clear();
            return result;
        } else {
            const std::size_t left = right - 1;
            const auto& a = anchors[left];
            // 默认沿用旧版退路：缺少同板外推参考时复制左端 A，不跨板估计变化率。
            item = a.item;
            info.left_anchor_time = info.right_anchor_time = a.time;
            if (right < anchors.size() && a.item.target_index == anchors[right].item.target_index) {
                const auto& b = anchors[right];
                // 已确定：允许前后同板的有效锚点跨过失败采样点，按真实时间间隔插值。
                // 只补齐控制参考，失败诊断仍保留；合成点不标记成经过验证的目标解。
                item = blendItem(a.item, b.item, (time - a.time) / (b.time - a.time));
                info.kind = FillKind::INTERPOLATED;
                info.right_anchor_time = b.time;
            } else if (left > 0 && anchors[left - 1].item.target_index == a.item.target_index) {
                const auto& previous = anchors[left - 1];
                // 已确定：当前段两端不同板（或没有右端锚点）时，只有前一个锚点 P
                // 与左端 A 同板才用 P、A 外推；否则保持上面的复制 A 退路。
                // 外推最多到原版序列末尾；若先遇到下一个有效锚点，则在那里重新分段。
                // 固定点与几何点间距不等，必须使用 P、A 的真实时间间隔。
                // 当前确定沿用旧方案：不额外限制外推角速度/加速度，保留下方数值异常退路。
                item = blendItem(previous.item, a.item,
                                 (time - previous.time) / (a.time - previous.time));
                info.kind = FillKind::EXTRAPOLATED;
                info.left_anchor_time = previous.time;
                info.right_anchor_time = a.time;
            }
            if (!finiteItem(item)) {
                // 防止纯数值外推产生 NaN 或非正飞行时间；退回复制有效锚点。
                item = a.item;
                info.kind = FillKind::COPIED;
                info.left_anchor_time = info.right_anchor_time = a.time;
            }
        }
        // 已确定：外推点允许作为控制参考进入序列并下发，不因 success=false 被剔除。
        // 首点必须精确有效；后续合成点的 success 仅表达是否经过独立精确验证，
        // 不直接决定 sequence.valid，也不作为外推点的下发开关。
        // 输出层不使用本标记屏蔽控制点或追加 Armor 火控：Armor 仅保留 track_ok。
        if (armor) item.success = info.exact_target_valid;
        // 复制点保留其来源锚点的 predict_time，不伪装成在当前时刻重新预测成功。
        sequence.items.push_back(item);
        result.point_info.push_back(info);
    }
    sequence.valid = true;
    sequence.first_point = sequence.items.front().predicted_point;
    sequence.first_predict_time = sequence.items.front().predict_time;
    sequence.yaw_world_origin = frame.yaw_world_origin;
    sequence.integral_enable = frame.input.auto_aim_switch;
    // 不生成高速对齐位置或枪线门控元数据；所有控制点按统一 Result 接口交给输出层。
    return result;
}

SequencePredictor::Item SequencePredictor::makeItem(
    const PreparedFrame& frame, const Candidate& candidate) const {
    Item item;
    item.success = candidate.success;
    item.predicted_point = candidate.predicted_point;
    item.predict_time = candidate.predict_time;
    item.yaw = static_cast<float>(candidate.gimbal.yaw + frame.input.chassis_yaw_correction + yaw_bias_);
    item.pitch = static_cast<float>(candidate.gimbal.pitch + pitch_bias_);
    item.gimbal_yaw = candidate.gimbal.yaw;
    item.gimbal_pitch = candidate.gimbal.pitch;
    item.flight_time = candidate.gimbal.flight_time;
    item.target_index = candidate.target_index;
    return item;
}

SequencePredictor::Item SequencePredictor::blendItem(
    const Item& lo, const Item& hi, double fraction) {
    Item result = lo;
    const auto blend = [fraction](double a, double b) { return a + fraction * (b - a); };
    result.predicted_point = lo.predicted_point +
        static_cast<float>(fraction) * (hi.predicted_point - lo.predicted_point);
    result.predict_time = blend(lo.predict_time, hi.predict_time);
    result.yaw = static_cast<float>(blend(lo.yaw, hi.yaw));
    result.pitch = static_cast<float>(blend(lo.pitch, hi.pitch));
    result.gimbal_yaw = static_cast<float>(blend(lo.gimbal_yaw, hi.gimbal_yaw));
    result.gimbal_pitch = static_cast<float>(blend(lo.gimbal_pitch, hi.gimbal_pitch));
    result.flight_time = blend(lo.flight_time, hi.flight_time);
    return result;
}

// ==================== 扇区几何：先验和最终选取复用 ====================
std::optional<double> SequencePredictor::sectorHalfAngle(
    double omega, double distance, double radius, double width, double w0) {
    if (!std::isfinite(omega) || !std::isfinite(distance) || !std::isfinite(radius) ||
        !std::isfinite(width) || !std::isfinite(w0) || radius <= kGeometryEpsilon ||
        distance <= radius || width <= 0.0 || w0 <= 0.0) return std::nullopt;
    // theta(w,s) = theta_min + (theta_max-theta_min)/(1+(|w|/w0)^2)。
    // s 是枪口到敌方中心的水平距离，r 是当前板半径，L 是板宽。
    // theta_max=acos(r/s)：低速极限对应圆的可见切点；
    // theta_min=atan(L/(2r))：高速极限保留正面一块板宽的中心角范围。
    const double maximum = std::acos(std::clamp(radius / distance, 0.0, 1.0));
    // 当前规则：极近距离时板宽角可能大于可见弧角，裁剪到可见弧上限。
    const double minimum = std::min(maximum, std::atan2(width / 2.0, radius));
    const double ratio = std::fabs(omega) / w0;
    return minimum + (maximum - minimum) / (1.0 + ratio * ratio);
}

double SequencePredictor::centerAngle(
    const cv::Vec3f& center, const cv::Vec3f& plate, const cv::Vec3f& muzzle) {
    // 两个向量都以敌方中心为起点，只投影到 world XY，不投影到枪口自身 XY。
    const double mx = static_cast<double>(muzzle[0]) - center[0];
    const double my = static_cast<double>(muzzle[1]) - center[1];
    const double px = static_cast<double>(plate[0]) - center[0];
    const double py = static_cast<double>(plate[1]) - center[1];
    if (std::hypot(mx, my) <= kGeometryEpsilon || std::hypot(px, py) <= kGeometryEpsilon)
        return std::numeric_limits<double>::quiet_NaN();
    return std::atan2(mx * py - my * px, mx * px + my * py);
}

bool SequencePredictor::usable(const Candidate& candidate) {
    return candidate.success && !candidate.masked && candidate.target_index >= 0 &&
           finitePoint(candidate.predicted_point) && std::isfinite(candidate.predict_time) &&
           std::isfinite(candidate.gimbal.yaw) && std::isfinite(candidate.gimbal.pitch) &&
           std::isfinite(candidate.gimbal.flight_time) && candidate.gimbal.flight_time > 0.0;
}

bool SequencePredictor::priorInsideSector(
    const PreparedFrame& frame, int plate, double control_time, double flight_time) const {
    // 把控制偏移转换到预测器快照时间轴，再加粗估飞行时间。
    // 这里只作窗口猜测；最终枪口使用求解姿态，最终时间使用 candidate.predict_time。
    const auto predicted = frame.predictor.function(frame.base_predict_time + control_time + flight_time);
    if (plate < 0 || plate >= static_cast<int>(predicted.second.size())) return false;
    const auto center = asVec(predicted.first);
    const auto point = asVec(predicted.second[static_cast<std::size_t>(plate)]);
    if (!finitePoint(center) || !finitePoint(point)) return false;
    const auto theta = sectorHalfAngle(frame.predictor.target_omega,
        horizontalDistance(center, frame.current_muzzle), horizontalDistance(center, point),
        frame.plate_width, options_.sector_w0);
    const double angle = centerAngle(center, point, frame.current_muzzle);
    return theta && std::isfinite(angle) && std::fabs(angle) <= *theta;
}

void SequencePredictor::appendGeometrySamples(PreparedFrame& frame) const {
    if (!frame.use_sector || options_.max_geometry_points == 0 || frame.samples.size() < 2) return;
    const double first = frame.samples.front().control_time;
    const double last = frame.samples.back().control_time;
    const double span = last - first;
    const double velocity = gimbals_.front()->bulletVelocity();
    if (!(span > options_.sample_merge_epsilon) || !std::isfinite(velocity) || velocity <= 0.0) return;
    std::vector<double> proposals;

    for (std::size_t plate = 0; plate < frame.initial_points.size(); ++plate) {
        const int index = static_cast<int>(plate);
        const double radius = frame.plate_radii[plate];
        if (frame.predictor.isIndexMasked(index) || !std::isfinite(radius) || radius <= kGeometryEpsilon) continue;
        // 当前规则：用当前板到当前枪口的距离/弹速粗估；不是完整飞行时间。
        // 这保证补点准备不依赖 Newton，不引入“选板阶段反调解算阶段”的循环依赖。
        const double flight_time = cv::norm(asVec(frame.initial_points[plate]) - frame.current_muzzle) / velocity;
        if (!std::isfinite(flight_time)) continue;
        const double angular_scale = std::atan2(frame.plate_width / 2.0, radius);
        const double omega = std::fabs(frame.predictor.target_omega);
        // 几何探测可比弹道精算密；按半个窄窗口半角限制每步相位变化。
        // 已接受的局限：步长是启发式；中心平动也会改变连线方向，可能漏掉短窗口。
        double step = interpolation_refine_ * dt_control_;
        if (omega > kGeometryEpsilon) step = std::min(step, angular_scale / (2.0 * omega));
        const double requested = std::ceil(span / std::max(step, options_.sample_merge_epsilon));
        const int probes = static_cast<int>(std::clamp(requested, 1.0,
                                    static_cast<double>(options_.max_prior_probes_per_plate)));
        if (requested > options_.max_prior_probes_per_plate) frame.prior_budget_limited = true;

        // 只对观测到的 false/true 边界二分；两端都 false 时不宣称区间绝对无窗口。
        const auto refine = [&](double lo, double hi, bool inside_at_hi) {
            for (int iteration = 0; iteration < options_.boundary_refine_iterations; ++iteration) {
                if (hi - lo <= options_.sample_merge_epsilon) break;
                const double middle = (lo + hi) / 2.0;
                if (priorInsideSector(frame, index, middle, flight_time) == inside_at_hi) hi = middle;
                else lo = middle;
            }
            return inside_at_hi ? hi : lo;       // 返回位于窗口内的一侧
        };
        bool previous_inside = priorInsideSector(frame, index, first, flight_time);
        double window_start = first;
        double previous_time = first;
        for (int probe = 1; probe <= probes; ++probe) {
            const double time = first + span * probe / probes;
            const bool inside = priorInsideSector(frame, index, time, flight_time);
            if (!previous_inside && inside) {
                window_start = refine(previous_time, time, true);
            } else if (previous_inside && !inside) {
                const double window_end = refine(previous_time, time, false);
                // 当前规则：每个窗口在中点补一个点，留出粗估误差余量；
                // 不再增加入口/出口点，补点精算失败也不反复追加搜索。
                proposals.push_back((window_start + window_end) / 2.0);
            }
            previous_inside = inside;
            previous_time = time;
        }
        if (previous_inside) proposals.push_back((window_start + last) / 2.0);
    }

    std::sort(proposals.begin(), proposals.end());
    // 当前规则：预算不足时优先较早的窗口，剩余窗口不再补点。
    for (double time : proposals) {
        if (!std::isfinite(time) || time < first || time > last) continue;
        const bool duplicate = std::any_of(frame.samples.begin(), frame.samples.end(), [&](const auto& point) {
            return std::fabs(point.control_time - time) <= options_.sample_merge_epsilon;
        });
        if (duplicate) continue;
        if (frame.geometry_points_added >= options_.max_geometry_points) {
            frame.prior_budget_limited = true;
            break;
        }
        frame.samples.push_back({time, SampleOrigin::GEOMETRY_PRIOR, -1});
        ++frame.geometry_points_added;
    }
    std::sort(frame.samples.begin(), frame.samples.end(), [](const auto& a, const auto& b) {
        return a.control_time < b.control_time;
    });
}

// ==================== 输入适配与大 yaw 时间轴 ====================
SequencePredictor::InputSnapshot SequencePredictor::snapshotFromSingle(
    const tcs::RobotController::State& state) {
    InputSnapshot input;
    input.info.single.yaw_pos = state.strict.yaw_pos;
    input.info.single.pitch_angle = state.strict.pitch_angle;
    input.info.single.imu_euler_yaw = state.strict.imu_euler_yaw;
    input.info.single.imu_euler_pitch = state.strict.imu_euler_pitch;
    input.info.single.imu_euler_roll = state.strict.imu_euler_roll;
    input.info.chassis_yaw = state.strict.chassis_yaw;
    input.info.chassis_pitch = state.strict.chassis_pitch;
    input.info.chassis_roll = state.strict.chassis_roll;
    input.has_bullet_velocity = state.mcu.valid;
    input.bullet_velocity = state.mcu.bullet_velocity;
    input.auto_aim_switch = state.mcu.auto_aim_switch == 1;
    input.chassis_yaw_correction = state.strict.imu_euler_yaw - state.strict.yaw_pos;
    return input;
}

SequencePredictor::InputSnapshot SequencePredictor::snapshotFromBigSmall(
    const bsy::RobotState& state) {
    InputSnapshot input;
    input.big_small = true;
    input.info = bsy::toExtraInputInfo(state);
    input.has_bullet_velocity = state.valid && std::isfinite(state.bullet_velocity) && state.bullet_velocity > 0.0;
    input.bullet_velocity = state.bullet_velocity;
    input.auto_aim_switch = state.auto_aim_switch == 1;
    input.chassis_yaw_correction = state.info_chassis_yaw;
    input.pred_big_azimuth_seq = state.pred_big_azimuth_seq;
    return input;
}

float SequencePredictor::yawBigAtTime(const InputSnapshot& input, double time) const {
    // 沿用旧版 time/dt 索引及末值保持约定，本次重构不改变大 yaw 时间映射。
    // 注意：旧数组说明中索引 0 对应 dt 的表述与此映射存在一拍差异，实验时需留意。
    if (!input.pred_big_azimuth_seq.empty()) {
        const auto& sequence = input.pred_big_azimuth_seq;
        const double index = std::clamp(time / dt_control_, 0.0, static_cast<double>(sequence.size() - 1));
        const auto lo = static_cast<std::size_t>(std::floor(index));
        const auto hi = std::min(lo + 1, sequence.size() - 1);
        const double fraction = index - lo;
        return static_cast<float>(sequence[lo] * (1.0 - fraction) + sequence[hi] * fraction - input.info.chassis_yaw);
    }
    if (!last_yaw_big_seq_.empty()) {
        const auto index = static_cast<std::size_t>(std::max(0.0, time / dt_control_));
        return last_yaw_big_seq_[std::min(index, last_yaw_big_seq_.size() - 1)];
    }
    return static_cast<float>(input.info.big_small.yaw_big_pos);
}
