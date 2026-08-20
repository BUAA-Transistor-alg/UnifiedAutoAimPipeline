#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
path_resolver.py — PathResolver 的 Python 版本（对应 C++ 的 include/common/PathResolver.h）

C++ 版假设可执行文件位于 <project_root>/bin/ 下，因此：
    项目根目录 = 可执行文件所在目录的父目录（dirname(dirname(exe_path))）

Python 版把「脚本自身」当作可执行文件（脚本位于 <project_root>/python/ 下），
项目根目录 = 脚本所在目录的父目录；若该目录下找不到 config/config.yaml
（例如脚本被移动/软链），则逐级向上查找包含 config/config.yaml 的目录兜底。
"""

import os
import sys


def get_executable_path():
    """获取脚本自身（或打包入口）的绝对路径，等价于 C++ 的 getExecutablePath()。"""
    # 优先用 __file__（被 import 时也正确）；作为脚本运行时与 sys.argv[0] 一致
    path = os.path.realpath(os.path.abspath(__file__))
    if not os.path.exists(path) and sys.argv and sys.argv[0]:
        path = os.path.realpath(os.path.abspath(sys.argv[0]))
    return path


def get_project_root():
    """获取项目根目录。

    优先采用 C++ 语义：脚本位于 <project_root>/python/，
    根目录 = 脚本所在目录的父目录；校验其中存在 config/config.yaml，
    不存在则逐级向上查找兜底（保证脚本被移动后仍可用）。
    """
    exe_path = get_executable_path()
    root = os.path.dirname(os.path.dirname(exe_path))  # 脚本所在目录的父目录

    # 兜底：向上查找包含 config/config.yaml 的目录
    cur = os.path.dirname(exe_path)
    while True:
        if os.path.isfile(os.path.join(cur, "config", "config.yaml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:  # 已到文件系统根
            break
        cur = parent

    # 未能通过兜底找到：退回 C++ 语义结果（与 C++ 行为一致）
    return root


def resolve_path(relative_path):
    """获取相对于项目根目录的绝对路径（等价于 C++ 的 resolvePath）。"""
    root = get_project_root()
    if not root:
        return relative_path  # 无法确定根目录时保持原样（C++ fallback 行为）
    if root.endswith(os.sep):
        return root + relative_path
    return root + os.sep + relative_path


if __name__ == "__main__":
    print("executable:", get_executable_path())
    print("root:      ", get_project_root())
    print("config:    ", resolve_path("config/config.yaml"))
