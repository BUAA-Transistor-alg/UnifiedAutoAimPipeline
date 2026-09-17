#pragma once
#include "Armor/ArmorVisualizationOptions.h"
#include "common/Output/CommonVisualizationOptions.h"
#include <opencv2/opencv.hpp>
#include <array>

// All methods and the HighGUI callback run on the existing visualization thread.
class VisualizationControls {
public:
    ArmorVisualizationOptions options;
    CommonVisualizationOptions common;
    bool changed = true;
    void open() {
        cv::namedWindow(name_, cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback(name_, onMouse, this);
        open_ = true;
        dirty_ = true;
    }
    void show(bool active, bool armor_active) {
        if (!open_) return;
        if (!windowExists()) {
            open_ = false;
            return;
        }
        if (!dirty_ && active == last_active_ && armor_active == last_armor_active_) return;
        dirty_ = false;
        last_active_ = active;
        last_armor_active_ = armor_active;
        cv::Mat panel(600, 500, CV_8UC3, cv::Scalar(30, 30, 30));
        cv::putText(panel, "Visualization controls", {18, 30}, 0, .7, {235,235,235}, 1, cv::LINE_AA);
        cv::putText(panel, active ? "Click a row to toggle" : "Inactive (settings retained)",
                    {18, 55}, 0, .45, {170,170,170}, 1, cv::LINE_AA);
        const auto row = [&](int y, const char* label, bool enabled, bool available) {
            const cv::Scalar color = available ? cv::Scalar(90,220,150) : cv::Scalar(120,120,120);
            cv::rectangle(panel, {18,y+7,22,22}, color, enabled ? -1 : 1);
            cv::putText(panel, label, {54,y+24}, 0, .5, color, 1, cv::LINE_AA);
        };
        for (int i = 0; i < 4; ++i)
            row(72 + i * 40, common_labels_[i], common.*common_fields_[i], active);
        cv::putText(panel, "Armor", {18,260}, 0, .6, {235,235,235}, 1, cv::LINE_AA);
        for (int i = 0; i < 7; ++i)
            row(280 + i * 40, labels_[i], options.*fields_[i],
                armor_active && (i != 1 || options.detections));
        cv::putText(panel, "v: visualization on/off   c: reopen controls", {18,583},
                    0, .43, {170,170,170}, 1, cv::LINE_AA);
        cv::imshow(name_, panel);
    }
    void close() {
        if (open_ && windowExists()) cv::destroyWindow(name_);
        open_ = false;
    }
private:
    bool windowExists() const {
        // GTK does not support WND_PROP_VISIBLE (-1 even for an open window).
        try { return cv::getWindowProperty(name_, cv::WND_PROP_AUTOSIZE) >= 0; }
        catch (const cv::Exception&) { return false; }
    }
    static void onMouse(int event, int x, int y, int, void* userdata) {
        if (event != cv::EVENT_LBUTTONDOWN || x < 12 || x >= 488) return;
        auto& self = *static_cast<VisualizationControls*>(userdata);
        if (!self.last_active_) return;
        if (y >= 72 && y < 232) {
            bool& value = self.common.*common_fields_[(y - 72) / 40];
            value = !value;
        } else if (y >= 280 && y < 560) {
            const int row = (y - 280) / 40;
            if (!self.last_armor_active_ || (row == 1 && !self.options.detections)) return;
            bool& value = self.options.*fields_[row];
            value = !value;
        } else return;
        self.changed = true;
        self.dirty_ = true;
    }
    inline static constexpr const char* name_ = "Visualization Controls";
    inline static constexpr std::array<bool ArmorVisualizationOptions::*,7> fields_ = {
        &ArmorVisualizationOptions::detections, &ArmorVisualizationOptions::details,
        &ArmorVisualizationOptions::pnp, &ArmorVisualizationOptions::filtered_projection,
        &ArmorVisualizationOptions::centers, &ArmorVisualizationOptions::raw_pose,
        &ArmorVisualizationOptions::filtered_pose};
    inline static constexpr std::array<const char*,7> labels_ = {
        "Detection results", "Detection details", "PnP reprojection", "Filtered geometry",
        "Armor centers (t=0)", "Raw pose text", "Filtered pose text"};
    inline static constexpr std::array<bool CommonVisualizationOptions::*,4> common_fields_ = {
        &CommonVisualizationOptions::status, &CommonVisualizationOptions::performance,
        &CommonVisualizationOptions::aim, &CommonVisualizationOptions::gimbal};
    inline static constexpr std::array<const char*,4> common_labels_ = {
        "Common: status text", "Common: performance", "Common: aim point", "Common: gimbal directions"};
    bool last_armor_active_ = false;
    bool open_ = false;
    bool dirty_ = true;
    bool last_active_ = false;
};
