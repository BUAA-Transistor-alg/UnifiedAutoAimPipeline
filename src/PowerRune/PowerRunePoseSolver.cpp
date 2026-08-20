#include "PowerRune/PowerRunePoseSolver.h"
#include "PowerRune/world_keypoints.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>

PowerRunePoseSolver::PowerRunePoseSolver(std::shared_ptr<CameraProjection> camera_proj)
    : camera_proj_(std::move(camera_proj))
{
}

// Rodrigues: v 绕单位轴 axis 旋转 angle 弧度
cv::Vec3f PowerRunePoseSolver::rotateVector(const cv::Vec3f& v,
                                            const cv::Vec3f& axis,
                                            float angle) {
    float c = std::cos(angle);
    float s = std::sin(angle);
    float dot = axis.dot(v);
    // Rodrigues' rotation formula: v*cos + (k×v)*sin + k*(k·v)*(1-cos)
    return v * c + axis.cross(v) * s + axis * dot * (1.0f - c);
}

// 将 v 投影到以 normal(单位向量) 为法向的平面上，结果归一化
cv::Vec3f PowerRunePoseSolver::projectToPlane(const cv::Vec3f& v,
                                              const cv::Vec3f& normal) {
    // v - (v·n)*n  投影到法平面
    cv::Vec3f proj = v - normal * normal.dot(v);
    float norm_val = std::sqrt(proj.dot(proj));
    if (norm_val < 1e-12f) {
        return cv::Vec3f(0, 0, 0);
    }
    return proj / norm_val;
}

cv::Vec3f PowerRunePoseSolver::getObjectZAxis(const cv::Mat& rvec) {
    // 物体局部坐标系的 z 轴是 (0,0,1)，通过 Rodrigues 变换到相机坐标系
    cv::Vec3f z_local(0, 0, 1);
    cv::Vec3f rvec_vec(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
    float angle = std::sqrt(rvec_vec.dot(rvec_vec));
    if (angle < 1e-12f) {
        return z_local;
    }
    cv::Vec3f axis = rvec_vec / angle;
    return rotateVector(z_local, axis, angle);
}

cv::Vec3f PowerRunePoseSolver::getObjectNegYAxis(const cv::Mat& rvec) {
    // 物体局部坐标系的 -y 轴是 (0,-1,0)，通过 Rodrigues 变换到相机坐标系
    cv::Vec3f ny_local(0, -1, 0);
    cv::Vec3f rvec_vec(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
    float angle = std::sqrt(rvec_vec.dot(rvec_vec));
    if (angle < 1e-12f) {
        return ny_local;
    }
    cv::Vec3f axis = rvec_vec / angle;
    return rotateVector(ny_local, axis, angle);
}

cv::Point3f PowerRunePoseSolver::rotateAroundZ(const cv::Point3f& pt, float angle) {
    float c = std::cos(angle);
    float s = std::sin(angle);
    return cv::Point3f(pt.x * c - pt.y * s, pt.x * s + pt.y * c, pt.z);
}

CombinedPoseResult PowerRunePoseSolver::estimateCombinedPose(
    const std::vector<PoseDetection>& detections) const
{
    constexpr float PI = 3.14159265358979323846f;
    CombinedPoseResult result;

    // ---------- 步骤 1: 对每个物体独立做 IPPE PnP 获取 rvec/tvec ----------
    struct ObjectInfo {
        int index;                      // 在原始detections中的索引
        const PoseDetection* det;
        cv::Mat rvec, tvec;
        std::vector<cv::Point2f> image_keypoints;
        const std::vector<cv::Point3f>* world_keypoints;
        cv::Vec3f z_axis;               // 物体 z 轴（相机坐标系）
        cv::Vec3f rot_dir;              // 物体旋转方向向量（-y投影到法平面）
        int rotation_count;             // 旋转次数 (0~4)
    };

    std::vector<ObjectInfo> objs;
    objs.reserve(detections.size());

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];

        std::vector<cv::Point2f> all_kpts_2d;
        all_kpts_2d.reserve(det.keypoints.size());
        for (const auto& kpt : det.keypoints) {
            all_kpts_2d.emplace_back(kpt.x, kpt.y);
        }

        auto extracted = power_rune_keypoints::extract_keypoints(det.class_id, all_kpts_2d);
        if (extracted.world_keypoints == nullptr ||
            extracted.world_keypoints->size() < 4 ||
            extracted.world_keypoints->size() != extracted.image_keypoints.size())
        {
            continue;
        }

        cv::Mat rvec, tvec;
        bool pnp_ok = camera_proj_->solvePnP(
            *extracted.world_keypoints,
            extracted.image_keypoints,
            rvec, tvec,
            false, cv::SOLVEPNP_IPPE);

        if (!pnp_ok) continue;

        ObjectInfo info;
        info.index = static_cast<int>(i);
        info.det = &det;
        info.rvec = rvec;
        info.tvec = tvec;
        info.world_keypoints = extracted.world_keypoints;
        info.image_keypoints = extracted.image_keypoints;
        info.z_axis = getObjectZAxis(rvec);
        info.rot_dir = cv::Vec3f(0, 0, 0);  // 稍后填充
        info.rotation_count = 0;
        objs.push_back(info);
    }

    if (objs.empty()) {
        result.error_code = 1;
        return result;
    }

    // ---------- 步骤 2: 轴向量 = 非特殊物体(class_id不为0/4) z 轴的平均 ----------
    // 特殊物体(class_id为0/4,即yolo_cls==0的R类)不参与轴向量计算
    cv::Vec3f axis_vector(0, 0, 0);
    int normal_count = 0;
    int special_count = 0;
    for (const auto& obj : objs) {
        int cls = obj.det->class_id % 4;
        if (cls == 0) {
            special_count++;
            continue;
        }
        normal_count++;
        axis_vector += obj.z_axis;
    }
    // 没有非特殊物体时: 若有特殊物体返回新错误码6, 否则返回1
    if (normal_count == 0) {
        if (special_count > 0) {
            result.error_code = 6;
            return result;
        }
        result.error_code = 1;
        return result;
    }
    float len = std::sqrt(axis_vector.dot(axis_vector));
    if (len < 1e-12f) {
        result.error_code = 2;
        return result;
    }
    axis_vector /= len;  // 单位化

    // ---------- 步骤 3: 物体旋转方向向量 = -y 投影到轴法平面并归一化 ----------
    for (auto& obj : objs) {
        cv::Vec3f neg_y = getObjectNegYAxis(obj.rvec);
        obj.rot_dir = projectToPlane(neg_y, axis_vector);
        if (obj.rot_dir.dot(obj.rot_dir) < 1e-12f) {
            // 退化情况，跳过该物体
            obj.rot_dir = cv::Vec3f(0, 0, 0);
        }
    }

    // ---------- 步骤 4: 相机旋转方向向量 = 相机 -y 投影到轴法平面并归一化 ----------
    // 相机坐标系中 -y 是 (0, -1, 0)
    cv::Vec3f cam_neg_y(0, -1, 0);
    cv::Vec3f cam_rot_dir = projectToPlane(cam_neg_y, axis_vector);

    // ---------- 步骤 5-7: 在相机方向附近优化初始参考方向 ----------
    // 特殊物体(class_id为0/4)不参与初始参考方向和旋转次数的计算
    float step_angle = 2.0f * PI / 5.0f;

    // 权重数组，特殊物体权重置零
    std::vector<float> weights;
    weights.reserve(objs.size());
    for (const auto& obj : objs) {
        int cls = obj.det->class_id % 4;
        if (cls == 0) {
            weights.push_back(0.0f);
        } else {
            weights.push_back(obj.det->rect.width * obj.det->rect.height);
        }
    }

    constexpr int kSearchSamples = 200;
    const float search_half_range = PI / 5.0f;  // ±π/5
    float best_offset = 0.0f;
    float best_cost  = std::numeric_limits<float>::max();

    for (int s = 0; s <= kSearchSamples; ++s) {
        float offset = -search_half_range
                     + (2.0f * search_half_range) * s / kSearchSamples;

        // 初始参考方向 = 相机方向绕轴旋转 offset
        cv::Vec3f init_ref = rotateVector(cam_rot_dir, axis_vector, offset);

        // 生成 5 个参考方向
        std::vector<cv::Vec3f> ref_dirs(5);
        for (int k = 0; k < 5; ++k) {
            ref_dirs[k] = rotateVector(init_ref, axis_vector, k * step_angle);
        }

        // 计算当前 offset 下的代价，仅考虑非特殊物体
        float cost = 0.0f;
        for (size_t i = 0; i < objs.size(); ++i) {
            int cls = objs[i].det->class_id % 4;
            if (cls == 0) continue;
            if (objs[i].rot_dir.dot(objs[i].rot_dir) < 1e-12f) continue;

            float closest_angle = std::numeric_limits<float>::max();
            for (int k = 0; k < 5; ++k) {
                float dot_val = objs[i].rot_dir.dot(ref_dirs[k]);
                if (dot_val >  1.0f) dot_val =  1.0f;
                if (dot_val < -1.0f) dot_val = -1.0f;
                float angle = std::acos(dot_val);
                if (angle < closest_angle) closest_angle = angle;
            }
            cost += weights[i] * closest_angle * closest_angle;
        }

        if (cost < best_cost) {
            best_cost   = cost;
            best_offset = offset;
        }
    }

    // 使用最优偏移得到最终的初始参考方向和 5 个参考方向
    cv::Vec3f init_ref_dir = rotateVector(cam_rot_dir, axis_vector, best_offset);
    std::vector<cv::Vec3f> ref_dirs(5);
    for (int k = 0; k < 5; ++k) {
        ref_dirs[k] = rotateVector(init_ref_dir, axis_vector, k * step_angle);
    }

    // 为每个非特殊物体分配旋转次数，特殊物体跳过(后续使用有向夹角)
    for (auto& obj : objs) {
        int cls = obj.det->class_id % 4;
        if (cls == 0) {
            obj.rotation_count = 0;
            continue;
        }
        if (obj.rot_dir.dot(obj.rot_dir) < 1e-12f) {
            obj.rotation_count = 0;
            continue;
        }
        float max_dot = -std::numeric_limits<float>::max();
        int best_rot = 0;
        for (int k = 0; k < 5; ++k) {
            float dot_val = obj.rot_dir.dot(ref_dirs[k]);
            if (dot_val > max_dot) {
                max_dot = dot_val;
                best_rot = k;
            }
        }
        obj.rotation_count = best_rot;
    }

    // ---------- 步骤 8: 构造全体 3D 坐标和 2D 坐标 ----------
    std::vector<cv::Point3f> all_world_pts;
    std::vector<cv::Point2f> all_image_pts;

    for (const auto& obj : objs) {
        float rot_angle;
        int cls = obj.det->class_id % 4;
        if (cls == 0) {
            // 特殊物体: 使用rot_dir与0号参考方向的有向夹角作为旋转角
            // 该夹角为在轴向量上的投影(有向),即init_ref_dir绕axis_vector
            // 旋转该角度后与obj.rot_dir重合
            if (obj.rot_dir.dot(obj.rot_dir) < 1e-12f) {
                rot_angle = 0.0f;
            } else {
                float dot_val = init_ref_dir.dot(obj.rot_dir);
                if (dot_val >  1.0f) dot_val =  1.0f;
                if (dot_val < -1.0f) dot_val = -1.0f;
                float unsigned_angle = std::acos(dot_val);
                cv::Vec3f cross = init_ref_dir.cross(obj.rot_dir);
                float sign = (axis_vector.dot(cross) >= 0) ? 1.0f : -1.0f;
                rot_angle = sign * unsigned_angle;
            }
        } else {
            rot_angle = obj.rotation_count * step_angle;
        }
        const auto& world_kps = *obj.world_keypoints;

        for (size_t j = 0; j < obj.image_keypoints.size(); ++j) {
            cv::Point3f rotated_pt = rotateAroundZ(world_kps[j], rot_angle);
            all_world_pts.push_back(rotated_pt);
            all_image_pts.push_back(obj.image_keypoints[j]);
        }
    }

    if (all_world_pts.size() < 4) {
        result.error_code = 4;
        return result;
    }

    // ---------- 步骤 9: 联合 PnP 解算（不使用 IPPE）----------
    cv::Mat rvec_combined, tvec_combined;

    // 1. 使用 EPNP 获取初始估计（快速非迭代）
    bool pnp_ok = camera_proj_->solvePnP(
        all_world_pts, all_image_pts,
        rvec_combined, tvec_combined,
        false,                         // useExtrinsicGuess = false
        cv::SOLVEPNP_EPNP
    );

    if (pnp_ok) {
        // 2. 以 EPNP 的结果作为初始值，进行迭代精炼
        pnp_ok = camera_proj_->solvePnP(
            all_world_pts, all_image_pts,
            rvec_combined, tvec_combined,
            true,                            // useExtrinsicGuess = true，使用传入的初值
            cv::SOLVEPNP_ITERATIVE
        );
    }

    if (!pnp_ok) {
        result.error_code = 5;
        return result;
    }

    // 保存联合位姿到返回值
    result.rvec = rvec_combined;
    result.tvec = tvec_combined;
    result.error_code = 0;

    // 额外保存 class_id 为 1 或 5 的物体的 rotation_count
    for (const auto& obj : objs) {
        int cid = obj.det->class_id;
        if (cid == 1 || cid == 5) {
            result.rotation_counts.push_back(obj.rotation_count);
        }
    }

    return result;
}

std::vector<cv::Point2f> PowerRunePoseSolver::projectPoints(std::vector<cv::Point3f> pnp_points_3d) const {
    std::vector<cv::Point2f> result;
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    camera_proj_->projectPoints(pnp_points_3d, rvec, tvec, result);
    
    return result;
}