#include "common/PathResolver.h"

#include <unistd.h>
#include <limits.h>
#include <cstring>
#include <libgen.h>

namespace PathResolver {

std::string getExecutablePath() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count > 0) {
        return std::string(result, static_cast<size_t>(count));
    }
    // fallback: 如果 readlink 失败（非 Linux 环境），返回空字符串
    return "";
}

std::string getProjectRoot() {
    // 如果编译时定义了 PROJECT_ROOT，优先使用
#ifdef PROJECT_ROOT
    return PROJECT_ROOT;
#endif
    std::string exe_path = getExecutablePath();
    if (exe_path.empty()) {
        // 如果无法获取可执行文件路径，fallback 到当前工作目录
        return ".";
    }

    // 复制字符串用于 dirname（dirname 可能修改参数）
    char* path_copy1 = strdup(exe_path.c_str());
    char* bin_dir = dirname(path_copy1);  // 可执行文件所在目录 (例如 .../bin)

    char* path_copy2 = strdup(bin_dir);
    char* project_root = dirname(path_copy2);  // 项目根目录

    std::string result(project_root);
    free(path_copy1);
    free(path_copy2);
    return result;
}

std::string resolvePath(const std::string& relative_path) {
    std::string root = getProjectRoot();
    if (root == ".") {
        return relative_path;  // fallback 情况下保持原样
    }
    if (root.back() == '/') {
        return root + relative_path;
    }
    return root + "/" + relative_path;
}

} // namespace PathResolver
