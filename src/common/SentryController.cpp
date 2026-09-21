// SentryController.cpp — 哨兵扫描控制器实现（见 SentryController.h）
#include "common/SentryController.h"

#include <algorithm>
#include <cmath>

namespace sentry {

SentryController::SentryController(const RobotConfig::SentryControllerParams& params)
    : params_(params) {}

void SentryController::update(bool valid, const TimePoint& now) {
    if (!params_.enabled) return;

    if (valid) {
        // 重新瞄准到目标：立即复位计时并退出扫描
        idle_active_ = false;
        scanning_    = false;
        return;
    }

    if (!idle_active_) {
        idle_active_ = true;
        idle_since_  = now;
    }
    if (!scanning_) {
        const double idle_s = std::chrono::duration<double>(now - idle_since_).count();
        if (idle_s >= params_.idleTimeoutSec) {
            scanning_   = true;
            scan_start_ = now;
        }
    }
}

double SentryController::scanElapsed(const TimePoint& now) const {
    if (!scanning_) return 0.0;
    return std::chrono::duration<double>(now - scan_start_).count();
}

double SentryController::yawTargetAt(double current, double t) const {
    const double target = current + params_.yawScanAngularVelocity * t;
    const double lo     = current - params_.yawScanMaxDeviation;
    const double hi     = current + params_.yawScanMaxDeviation;
    return std::min(std::max(target, lo), hi);
}

double SentryController::pitchTargetAt(double t) const {
    const double range = params_.pitchScanMax - params_.pitchScanMin;
    const double rise  = params_.pitchScanRiseTimeSec;
    const double fall  = params_.pitchScanFallTimeSec;
    const double period = rise + fall;

    double phase = std::fmod(t, period);
    if (phase < 0.0) phase += period;

    if (phase < rise) {
        return params_.pitchScanMin + range * (phase / rise);
    }
    return params_.pitchScanMax - range * ((phase - rise) / fall);
}

void SentryController::buildYawSequence(int n, double dt, double current_yaw,
                                        const TimePoint& now,
                                        std::vector<double>& out) const {
    out.clear();
    if (n <= 0) return;
    out.reserve((size_t)n);
    const double elapsed = scanElapsed(now);
    for (int k = 0; k < n; ++k) {
        const double t = elapsed + (double)(k + 1) * dt;
        out.push_back(yawTargetAt(current_yaw, t));
    }
}

void SentryController::buildPitchSequence(int n, double dt, const TimePoint& now,
                                          std::vector<double>& out) const {
    out.clear();
    if (n <= 0) return;
    out.reserve((size_t)n);
    const double elapsed = scanElapsed(now);
    for (int k = 0; k < n; ++k) {
        const double t = elapsed + (double)(k + 1) * dt;
        out.push_back(pitchTargetAt(t));
    }
}

} // namespace sentry
