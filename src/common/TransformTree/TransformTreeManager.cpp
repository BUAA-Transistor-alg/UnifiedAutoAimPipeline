// TransformTreeManager.cpp
#include "TransformTree/TransformTreeManager.h"

#include <functional>
#include <stdexcept>

TransformTreeManager::TransformTreeManager(const std::string& rootName)
    : rootName_(rootName), locked_(false) {
    root_ = std::make_shared<TransformNode>(rootName);
    nodes_[rootName_] = root_;
}

std::shared_ptr<TransformNode> TransformTreeManager::findNode(const std::string& name) const {
    auto it = nodes_.find(name);
    if (it == nodes_.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<TransformNode> TransformTreeManager::getNode(const std::string& name) const {
    return findNode(name);
}

const std::string& TransformTreeManager::getRootName() const {
    return rootName_;
}

void TransformTreeManager::ensureUnlocked() const {
    if (locked_) {
        throw std::logic_error("TransformTreeManager: 变换已上锁，请先调用 unlock() 再修改坐标/欧拉角");
    }
}

void TransformTreeManager::ensureLocked() const {
    if (!locked_) {
        throw std::logic_error("TransformTreeManager: 变换未上锁，请先调用 lockAndComputeCache() 再查询");
    }
}

std::shared_ptr<TransformNode> TransformTreeManager::addNode(const std::string& name, const std::string& parentName) {
    if (nodes_.find(name) != nodes_.end()) {
        throw std::runtime_error("TransformTreeManager: 节点 '" + name + "' 已存在");
    }
    auto parent = findNode(parentName);
    if (!parent) {
        throw std::runtime_error("TransformTreeManager: 父节点 '" + parentName + "' 不存在");
    }
    auto node = std::make_shared<TransformNode>(name);
    nodes_[name] = node;
    node->setParent(parent);
    return node;
}

void TransformTreeManager::removeNode(const std::string& name) {
    if (name == rootName_) {
        throw std::runtime_error("TransformTreeManager: 不能移除根节点 '" + name + "'");
    }
    auto node = findNode(name);
    if (!node) {
        throw std::runtime_error("TransformTreeManager: 节点 '" + name + "' 不存在");
    }

    // 先收集所有子孙节点，再统一摘除链接并从管理表中移除。
    std::vector<std::shared_ptr<TransformNode>> toRemove;
    std::function<void(const std::shared_ptr<TransformNode>&)> collect =
        [&](const std::shared_ptr<TransformNode>& n) {
            for (const auto& child : n->getChildren()) {
                toRemove.push_back(child);
                collect(child);
            }
        };
    collect(node);

    for (const auto& n : toRemove) {
        n->setParent(nullptr);
        nodes_.erase(n->getName());
    }
    node->setParent(nullptr);
    nodes_.erase(name);
}

bool TransformTreeManager::isDescendantOf(const std::shared_ptr<TransformNode>& node,
                                          const std::shared_ptr<TransformNode>& ancestor) const {
    auto current = node->getParent();
    while (current) {
        if (current == ancestor) {
            return true;
        }
        current = current->getParent();
    }
    return false;
}

void TransformTreeManager::reparentNode(const std::string& name, const std::string& newParentName) {
    if (name == rootName_) {
        throw std::runtime_error("TransformTreeManager: 不能更换根节点 '" + name + "' 的父节点");
    }
    auto node = findNode(name);
    auto newParent = findNode(newParentName);
    if (!node) {
        throw std::runtime_error("TransformTreeManager: 节点 '" + name + "' 不存在");
    }
    if (!newParent) {
        throw std::runtime_error("TransformTreeManager: 父节点 '" + newParentName + "' 不存在");
    }
    if (newParent == node || isDescendantOf(newParent, node)) {
        throw std::runtime_error("TransformTreeManager: 不能把节点 '" + name + "' 的父节点设为自身或其子孙节点");
    }
    node->setParent(newParent);
}

void TransformTreeManager::setPosition(const std::string& name, float x, float y, float z) {
    setPosition(name, cv::Vec3f(x, y, z));
}

void TransformTreeManager::setPosition(const std::string& name, const cv::Vec3f& position) {
    ensureUnlocked();
    auto node = findNode(name);
    if (!node) {
        throw std::runtime_error("TransformTreeManager: 节点 '" + name + "' 不存在");
    }
    node->setPosition(position);
}

void TransformTreeManager::setEuler(const std::string& name, float yaw, float pitch, float roll) {
    setEuler(name, cv::Vec3f(yaw, pitch, roll));
}

void TransformTreeManager::setEuler(const std::string& name, const cv::Vec3f& euler) {
    ensureUnlocked();
    auto node = findNode(name);
    if (!node) {
        throw std::runtime_error("TransformTreeManager: 节点 '" + name + "' 不存在");
    }
    node->setEuler(euler);
}

void TransformTreeManager::unlock() {
    locked_ = false;
    cache_.clear();
}

void TransformTreeManager::lockAndComputeCache() {
    cache_.clear();
    computeCache(root_, cv::Mat::eye(3, 3, CV_32F), cv::Vec3f(0, 0, 0));
    locked_ = true;
}

bool TransformTreeManager::isLocked() const {
    return locked_;
}
void TransformTreeManager::computeCache(const std::shared_ptr<TransformNode>& node,
                                        const cv::Mat& parentToRootR,
                                        const cv::Vec3f& parentToRootT) {
    cv::Mat localR = CoordinateTransform::eulerToRotationMatrix(node->getEuler());
    cv::Vec3f localT = node->getPosition();

    cv::Mat nodeToRootR = parentToRootR * localR;

    cv::Mat localTMat = (cv::Mat_<float>(3, 1) << localT[0], localT[1], localT[2]);
    cv::Mat rotatedT = parentToRootR * localTMat;
    cv::Vec3f nodeToRootT(
        rotatedT.at<float>(0, 0) + parentToRootT[0],
        rotatedT.at<float>(1, 0) + parentToRootT[1],
        rotatedT.at<float>(2, 0) + parentToRootT[2]
    );

    cache_[node->getName()] = CachedTransform{ nodeToRootR.clone(), nodeToRootT };

    for (const auto& child : node->getChildren()) {
        computeCache(child, nodeToRootR, nodeToRootT);
    }
}

cv::Vec3f TransformTreeManager::transformPoint(const std::string& from, const std::string& to, const cv::Vec3f& point) const {
    ensureLocked();
    auto fromIt = cache_.find(from);
    auto toIt = cache_.find(to);
    if (fromIt == cache_.end()) {
        throw std::runtime_error("TransformTreeManager: 节点 '" + from + "' 不在缓存中");
    }
    if (toIt == cache_.end()) {
        throw std::runtime_error("TransformTreeManager: 节点 '" + to + "' 不在缓存中");
    }

    const CachedTransform& fromCache = fromIt->second;
    const CachedTransform& toCache = toIt->second;

    cv::Mat p = (cv::Mat_<float>(3, 1) << point[0], point[1], point[2]);
    cv::Mat fromT = (cv::Mat_<float>(3, 1) << fromCache.t[0], fromCache.t[1], fromCache.t[2]);
    cv::Mat toT = (cv::Mat_<float>(3, 1) << toCache.t[0], toCache.t[1], toCache.t[2]);

    // 先转到根节点坐标系，再用目标节点变换的逆变换转回目标坐标系。
    cv::Mat pRoot = fromCache.R * p + fromT;
    cv::Mat pTo = toCache.R.t() * (pRoot - toT);

    return cv::Vec3f(pTo.at<float>(0, 0), pTo.at<float>(1, 0), pTo.at<float>(2, 0));
}

cv::Vec3f TransformTreeManager::transformEuler(const std::string& from, const std::string& to, const cv::Vec3f& euler) const {
    ensureLocked();
    auto fromIt = cache_.find(from);
    auto toIt = cache_.find(to);
    if (fromIt == cache_.end()) {
        throw std::runtime_error("TransformTreeManager: 节点 '" + from + "' 不在缓存中");
    }
    if (toIt == cache_.end()) {
        throw std::runtime_error("TransformTreeManager: 节点 '" + to + "' 不在缓存中");
    }

    const CachedTransform& fromCache = fromIt->second;
    const CachedTransform& toCache = toIt->second;

    cv::Mat rotationInFrom = CoordinateTransform::eulerToRotationMatrix(euler);
    cv::Mat rotationInRoot = fromCache.R * rotationInFrom;
    cv::Mat rotationInTo = toCache.R.t() * rotationInRoot;

    return CoordinateTransform::rotationMatrixToEuler(rotationInTo);
}

