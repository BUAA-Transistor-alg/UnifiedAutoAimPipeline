// TransformNode.h
#ifndef TRANSFORM_NODE_H
#define TRANSFORM_NODE_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "CoordinateTransform.h"

// 坐标系变换树中的一个节点。
// 每个节点记录自身坐标系相对父节点坐标系的平移（position）与欧拉角（euler）。
// 坐标系定义与 CoordinateTransform 保持一致：
//   x：向右，y：向前，z：向上
//   yaw：绕 z 轴，从上方看逆时针（x 轴转向 y 轴）
//   pitch：绕 x 轴，抬头（y 轴转向 z 轴）
//   roll：绕 y 轴，从画面看顺时针（z 轴转向 x 轴）
class TransformNode : public std::enable_shared_from_this<TransformNode> {
public:
    explicit TransformNode(const std::string& name);

    const std::string& getName() const;

    // 父子关系维护（setParent 会同步维护父节点的 children 列表）
    void setParent(std::shared_ptr<TransformNode> parent);
    std::shared_ptr<TransformNode> getParent() const;

    void addChild(std::shared_ptr<TransformNode> child);
    void removeChild(const std::shared_ptr<TransformNode>& child);
    const std::vector<std::shared_ptr<TransformNode>>& getChildren() const;

    // 自身坐标系相对父节点坐标系的位置与欧拉角
    void setPosition(float x, float y, float z);
    void setPosition(const cv::Vec3f& position);
    cv::Vec3f getPosition() const;

    void setEuler(float yaw, float pitch, float roll);
    void setEuler(const cv::Vec3f& euler);
    cv::Vec3f getEuler() const;

    // 自身坐标系 -> 父节点坐标系
    cv::Vec3f toParentPoint(const cv::Vec3f& pointInSelf) const;
    cv::Vec3f toParentEuler(const cv::Vec3f& eulerInSelf) const;

    // 父节点坐标系 -> 自身坐标系
    cv::Vec3f toSelfPoint(const cv::Vec3f& pointInParent) const;
    cv::Vec3f toSelfEuler(const cv::Vec3f& eulerInParent) const;

private:
    std::string name_;
    std::weak_ptr<TransformNode> parent_;                   // 父节点（弱引用，避免循环引用）
    std::vector<std::shared_ptr<TransformNode>> children_;  // 子节点
    cv::Vec3f position_;                                    // 在父节点坐标系下的平移
    cv::Vec3f euler_;                                       // 在父节点坐标系下的欧拉角(yaw, pitch, roll)
};

#endif // TRANSFORM_NODE_H
