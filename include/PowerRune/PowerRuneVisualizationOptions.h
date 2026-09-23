#pragma once

// Display-only settings, retained across pipeline / visualization toggles.
struct PowerRuneVisualizationOptions {
    bool detections = true;
    bool details = false;
    bool raw_pose = false;
    bool filtered_pose = true;
    bool predicted_pose = true;
    bool fit_window = true;
};
