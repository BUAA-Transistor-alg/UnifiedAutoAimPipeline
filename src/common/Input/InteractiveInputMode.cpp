#include "Input/InteractiveInputMode.h"
#include <iostream>
#include <string>

using namespace std;

InteractiveInputMode::InteractiveInputMode() {
    cout << "\nInteractive mode: enter image path (type 'q' to quit):" << endl;
}

bool InteractiveInputMode::getNextFrame(cv::Mat& frame,
                                        std::chrono::steady_clock::time_point& timestamp,
                                        ExtraInputInfo& extra_info) {
    extra_info = ExtraInputInfo{};  // default all-zero
    while (true) {
        cout << "> " << flush;
        string input_path;
        getline(cin, input_path);

        if (input_path.empty()) continue;

        input_path = stripQuotes(input_path);
        if (input_path == "q" || input_path == "Q" || input_path == "quit") {
            return false;
        }

        frame = cv::imread(input_path);
        if (frame.empty()) {
            cerr << "Failed to read image: " << input_path << endl;
            continue;
        }
        timestamp = std::chrono::steady_clock::now();
        return true;
    }
}

string InteractiveInputMode::stripQuotes(const string& s) {
    if (s.empty()) return s;
    string temp = s;
    while (!temp.empty() && (temp.front() == '\'' || temp.front() == ' ' ||
                              temp.front() == '"' || temp.front() == '\n')) {
        temp = temp.substr(1);
    }
    while (!temp.empty() && (temp.back() == '\'' || temp.back() == ' ' ||
                              temp.back() == '"' || temp.back() == '\n')) {
        temp = temp.substr(0, temp.size() - 1);
    }
    return temp;
}