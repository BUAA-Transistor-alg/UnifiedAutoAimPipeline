#pragma once

// Display-only settings. Owned by the visualization thread, retained across v/1/2.
struct ArmorVisualizationOptions {
    bool detections = true;
    bool details = false;
    bool pnp = false;
    bool filtered_projection = true;
    bool centers = true;
    bool raw_pose = false;
    bool filtered_pose = false;
};
