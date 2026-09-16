// TfTreeSync.h — ExtraInputInfo → RobotTfTree 同步（**按 yaw 构型选包**的唯一入口）
//
// 背景：ExtraInputInfo 把关节角按构型分成两包（single / big_small），当前构型只填
// 其中一包、另一包整体为 NaN（见 include/common/Input/IInputMode.h）。各流水线阶段
// （ArmorPipeline stage4/stage5、PowerRunePipeline stage4、VisualizeOutput）每帧都要把
// 该帧信息同步进自己那棵 RobotTfTree，因此**统一走本文件的两个函数**：
//   - 按 RobotConfig 的构型取对应包（不会把单 yaw 的关节角按双 yaw 语义写进树）；
//   - 当前构型包为 NaN（未填充/构型不匹配：例如用单 yaw 录制的 v2 视频回放新模式）
//     时**立刻抛 std::runtime_error**，而不是用 NaN 算出一条看似正常的姿态。
#ifndef TF_TREE_SYNC_H
#define TF_TREE_SYNC_H

#include <stdexcept>
#include <string>

#include "common/Input/IInputMode.h"
#include "common/RobotConfig.h"
#include "common/TransformTree/RobotTfTree.h"

// 把 ExtraInputInfo 的底盘位姿 + 当前构型关节角写入树（调用前树需处于 unlock 状态）
inline void applyExtraInputInfoToTree(RobotTfTree& tree, const ExtraInputInfo& info) {
    const YawMode mode = RobotConfig::instance().common.bigSmallYaw.mode;
    if (tree.yawMode() != mode) {
        throw std::logic_error("TfTreeSync: 变换树构型与配置构型不一致（树构造时定型，"
                               "配置运行中不应变化）");
    }
    tree.setChassisPosition((float)info.chassis_x, (float)info.chassis_y,
                            (float)info.chassis_z);
    tree.setChassisEuler((float)info.chassis_yaw, (float)info.chassis_pitch,
                         (float)info.chassis_roll);
    if (mode == YawMode::SINGLE) {
        if (!info.single.valid()) {
            throw std::runtime_error(
                "TfTreeSync: 当前为单 yaw 构型（common.big_small_yaw.mode = single），"
                "但本帧 ExtraInputInfo.single 包未填充（NaN）——输入模式与构型不匹配"
                "（例如用大小 yaw 构型录制的数据回放单 yaw 模式）");
        }
        tree.setYaw((float)info.single.yaw_pos);
        tree.setPitch((float)info.single.pitch_angle);
    } else {
        if (!info.big_small.valid()) {
            throw std::runtime_error(
                "TfTreeSync: 当前为大小 yaw 构型（common.big_small_yaw.mode = big_small），"
                "但本帧 ExtraInputInfo.big_small 包未填充（NaN）——输入模式与构型不匹配"
                "（例如用单 yaw 录制的 v1/v2 数据回放大小 yaw 模式）");
        }
        tree.setYawBig((float)info.big_small.yaw_big_pos);
        tree.setYawSmall((float)info.big_small.yaw_small_pos);
        tree.setPitch((float)info.big_small.pitch_angle);
    }
}

// 便捷版：unlock → 写入 → （可选）lockAndComputeCache
inline void syncTreeFromExtraInfo(RobotTfTree& tree, const ExtraInputInfo& info,
                                  bool lock = true) {
    tree.unlock();
    applyExtraInputInfoToTree(tree, info);
    if (lock) tree.lockAndComputeCache();
}

#endif // TF_TREE_SYNC_H
