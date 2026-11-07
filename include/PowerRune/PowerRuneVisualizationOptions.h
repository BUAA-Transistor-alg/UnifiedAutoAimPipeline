#pragma once

// 仅用于显示的设置，在流水线 / 可视化切换之间保留。
struct PowerRuneVisualizationOptions {
    bool detections = true;
    bool details = false;
    bool raw_pose = false;
    bool filtered_pose = true;
    bool predicted_pose = true;
    bool fit_window = true;
};
