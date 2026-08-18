// TransformTreeManager.h
#ifndef TRANSFORM_TREE_MANAGER_H
#define TRANSFORM_TREE_MANAGER_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

#include "TransformNode.h"

// 统一管理一棵坐标系变换树中的所有 TransformNode。
// 通过节点名字对节点进行操作；支持解锁修改、上锁后构建快速变换缓存。
class TransformTreeManager {
public:
    // 构造时创建根节点，默认命名为 "root"。
    explicit TransformTreeManager(const std::string& rootName = "root");

    // ── 树结构管理 ──
    // 在 parentName 节点下新建一个节点（根节点已在构造时创建）。
    std::shared_ptr<TransformNode> addNode(const std::string& name, const std::string& parentName);
    // 移除节点及其所有子节点。
    void removeNode(const std::string& name);
    // 更换节点的父节点。
    void reparentNode(const std::string& name, const std::string& newParentName);

    // ── 节点在父节点坐标系下的位姿（仅解锁时可修改）──
    void setPosition(const std::string& name, float x, float y, float z);
    void setPosition(const std::string& name, const cv::Vec3f& position);
    void setEuler(const std::string& name, float yaw, float pitch, float roll);
    void setEuler(const std::string& name, const cv::Vec3f& euler);

    // ── 解锁 / 上锁 ──
    // 解锁：允许修改节点的坐标与欧拉角，同时清空旧的变换缓存。
    void unlock();
    // 上锁并递归计算所有节点相对根节点的快速变换缓存。
    void lockAndComputeCache();
    bool isLocked() const;

    // ── 基于缓存的快速坐标 / 欧拉角变换（仅上锁后可用）──
    cv::Vec3f transformPoint(const std::string& from, const std::string& to, const cv::Vec3f& point) const;
    cv::Vec3f transformEuler(const std::string& from, const std::string& to, const cv::Vec3f& euler) const;

    const std::string& getRootName() const;
    std::shared_ptr<TransformNode> getNode(const std::string& name) const;

private:
    struct CachedTransform {
        cv::Mat R;    // 3x3 旋转矩阵：节点坐标系 -> 根节点坐标系
        cv::Vec3f t;  // 平移向量：节点坐标系 -> 根节点坐标系
    };

    std::shared_ptr<TransformNode> findNode(const std::string& name) const;
    void ensureUnlocked() const;
    void ensureLocked() const;
    void computeCache(const std::shared_ptr<TransformNode>& node,
                      const cv::Mat& parentToRootR,
                      const cv::Vec3f& parentToRootT);
    bool isDescendantOf(const std::shared_ptr<TransformNode>& node,
                        const std::shared_ptr<TransformNode>& ancestor) const;

    std::string rootName_;
    std::shared_ptr<TransformNode> root_;
    std::unordered_map<std::string, std::shared_ptr<TransformNode>> nodes_;
    bool locked_;
    std::unordered_map<std::string, CachedTransform> cache_;
};

#endif // TRANSFORM_TREE_MANAGER_H
