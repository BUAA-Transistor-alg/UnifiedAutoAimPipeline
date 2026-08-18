// TransformNode.cpp
#include "TransformTree/TransformNode.h"

#include <algorithm>

TransformNode::TransformNode(const std::string& name)
    : name_(name), position_(cv::Vec3f(0, 0, 0)), euler_(cv::Vec3f(0, 0, 0)) {}

const std::string& TransformNode::getName() const {
    return name_;
}

void TransformNode::setParent(std::shared_ptr<TransformNode> parent) {
    if (auto old = parent_.lock()) {
        if (old == parent) {
            return; // 父节点未变化
        }
        old->removeChild(shared_from_this());
    }
    parent_ = parent;
    if (parent) {
        parent->addChild(shared_from_this());
    }
}

std::shared_ptr<TransformNode> TransformNode::getParent() const {
    return parent_.lock();
}

void TransformNode::addChild(std::shared_ptr<TransformNode> child) {
    if (!child) {
        return;
    }
    for (const auto& c : children_) {
        if (c == child) {
            return; // 已存在
        }
    }
    children_.push_back(std::move(child));
}

void TransformNode::removeChild(const std::shared_ptr<TransformNode>& child) {
    children_.erase(std::remove(children_.begin(), children_.end(), child), children_.end());
}

const std::vector<std::shared_ptr<TransformNode>>& TransformNode::getChildren() const {
    return children_;
}

void TransformNode::setPosition(float x, float y, float z) {
    position_ = cv::Vec3f(x, y, z);
}

void TransformNode::setPosition(const cv::Vec3f& position) {
    position_ = position;
}

cv::Vec3f TransformNode::getPosition() const {
    return position_;
}

void TransformNode::setEuler(float yaw, float pitch, float roll) {
    euler_ = cv::Vec3f(yaw, pitch, roll);
}

void TransformNode::setEuler(const cv::Vec3f& euler) {
    euler_ = euler;
}

cv::Vec3f TransformNode::getEuler() const {
    return euler_;
}

cv::Vec3f TransformNode::toParentPoint(const cv::Vec3f& pointInSelf) const {
    cv::Mat R = CoordinateTransform::eulerToRotationMatrix(euler_);
    cv::Mat p = (cv::Mat_<float>(3, 1) << pointInSelf[0], pointInSelf[1], pointInSelf[2]);
    cv::Mat rotated = R * p;
    return cv::Vec3f(
        rotated.at<float>(0, 0) + position_[0],
        rotated.at<float>(1, 0) + position_[1],
        rotated.at<float>(2, 0) + position_[2]
    );
}

cv::Vec3f TransformNode::toSelfPoint(const cv::Vec3f& pointInParent) const {
    cv::Mat R = CoordinateTransform::eulerToRotationMatrix(euler_);
    cv::Mat p = (cv::Mat_<float>(3, 1) <<
        pointInParent[0] - position_[0],
        pointInParent[1] - position_[1],
        pointInParent[2] - position_[2]
    );
    cv::Mat rotated = R.t() * p;
    return cv::Vec3f(rotated.at<float>(0, 0), rotated.at<float>(1, 0), rotated.at<float>(2, 0));
}

cv::Vec3f TransformNode::toParentEuler(const cv::Vec3f& eulerInSelf) const {
    cv::Mat nodeRotation = CoordinateTransform::eulerToRotationMatrix(euler_);
    cv::Mat selfRotation = CoordinateTransform::eulerToRotationMatrix(eulerInSelf);
    return CoordinateTransform::rotationMatrixToEuler(nodeRotation * selfRotation);
}

cv::Vec3f TransformNode::toSelfEuler(const cv::Vec3f& eulerInParent) const {
    cv::Mat nodeRotation = CoordinateTransform::eulerToRotationMatrix(euler_);
    cv::Mat parentRotation = CoordinateTransform::eulerToRotationMatrix(eulerInParent);
    return CoordinateTransform::rotationMatrixToEuler(nodeRotation.t() * parentRotation);
}
