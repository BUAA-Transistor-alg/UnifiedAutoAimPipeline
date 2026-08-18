#ifndef PATH_RESOLVER_H
#define PATH_RESOLVER_H

#include <string>

namespace PathResolver {

// 获取可执行文件的绝对路径
// 使用 Linux 的 /proc/self/exe 来获取当前进程的可执行文件路径
// 无论从哪个目录执行程序，都能正确返回可执行文件所在的真实路径
std::string getExecutablePath();

// 获取项目根目录
// 假设可执行文件位于 <project_root>/bin/ 下
// 因此项目根目录 = 可执行文件所在目录的父目录
std::string getProjectRoot();

// 获取相对于项目根目录的绝对路径
std::string resolvePath(const std::string& relative_path);

} // namespace PathResolver

#endif // PATH_RESOLVER_H
