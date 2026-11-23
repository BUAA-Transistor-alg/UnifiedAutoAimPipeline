#pragma once

// 仅用于显示的设置，由可视化线程持有，在 v/1/2 切换之间保留。
struct ArmorVisualizationOptions {
    bool detections = true;
    bool details = false;
    bool pnp = false;
    bool filtered_projection = true;
    bool centers = true;
    bool raw_pose = false;
    bool filtered_pose = false;
    bool roi = true;   // 本帧识别视图框（config armor.roi：ROI 截取矩形与全画面边缘框）
};
