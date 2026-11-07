#ifndef INTERACTIVEINPUTMODE_H
#define INTERACTIVEINPUTMODE_H

#include "common/Input/IInputMode.h"
#include <chrono>
#include <string>

/**
 * @brief 交互式单图输入模式。
 *
 * 用户从 stdin 输入图像文件路径。
 * 空行回车表示跳过该帧，输入 'q' 或 'quit' 退出。
 */
class InteractiveInputMode : public IInputMode {
public:
    InteractiveInputMode();

    bool getNextFrame(cv::Mat& frame,
                      std::chrono::steady_clock::time_point& timestamp,
                      ExtraInputInfo& extra_info) override;
    std::string getName() const override { return "Interactive"; }
    float getFrameDelay() const override { return 0.0f; }  // 不做额外延迟

private:
    std::string stripQuotes(const std::string& s);
};

#endif // INTERACTIVEINPUTMODE_H