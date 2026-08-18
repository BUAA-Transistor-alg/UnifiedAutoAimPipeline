#ifndef INTERACTIVEINPUTMODE_H
#define INTERACTIVEINPUTMODE_H

#include "IInputMode.h"
#include <chrono>
#include <string>

/**
 * @brief Interactive single-image input mode.
 *
 * The user types image file paths into stdin.
 * Press Enter on an empty line to skip,
 * 'q' or 'quit' to exit.
 */
class InteractiveInputMode : public IInputMode {
public:
    InteractiveInputMode();

    bool getNextFrame(cv::Mat& frame,
                      std::chrono::steady_clock::time_point& timestamp,
                      ExtraInputInfo& extra_info) override;
    std::string getName() const override { return "Interactive"; }
    float getFrameDelay() const override { return 0.0f; }  // no artificial delay

private:
    std::string stripQuotes(const std::string& s);
};

#endif // INTERACTIVEINPUTMODE_H