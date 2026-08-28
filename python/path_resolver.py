#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
path_resolver.py — PathResolver 的 Python 版本（对应 C++ 的 include/common/PathResolver.h）

C++ 版假设可执行文件位于 <project_root>/bin/ 下，因此：
    项目根目录 = 可执行文件所在目录的父目录（dirname(dirname(exe_path))）

Python 版把「脚本自身」当作可执行文件（脚本位于 <project_root>/python/ 下），
项目根目录 = 脚本所在目录的父目录；若该目录下找不到 config/selector.yaml
（例如脚本被移动/软链），则逐级向上查找包含 config/selector.yaml 的目录兜底。

config 读取规则（v2，与 C++ 的 RobotConfig::instance() 保持一致）：
    config/selector.yaml          —— 机器配置选择器（仅含 active_config 一个条目）
    config/robots/<active_config>.yaml —— 当前机器的完整参数（如 Infantry1.yaml）
本模块的 resolve_machine_config_path() 先读选择器，再返回对应机器配置文件的绝对路径。
"""

import os
import sys

import yaml


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
    根目录 = 脚本所在目录的父目录；校验其中存在 config/selector.yaml，
    不存在则逐级向上查找兜底（保证脚本被移动后仍可用）。
    """
    exe_path = get_executable_path()
    root = os.path.dirname(os.path.dirname(exe_path))  # 脚本所在目录的父目录

    # 兜底：向上查找包含 config/selector.yaml 的目录
    cur = os.path.dirname(exe_path)
    while True:
        if os.path.isfile(os.path.join(cur, "config", "selector.yaml")):
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


def resolve_machine_config_path():
    """返回当前机器配置文件的绝对路径（config 读取规则 v2，与 C++ RobotConfig 一致）：

      1) 读取机器配置选择器 config/selector.yaml 的 active_config 条目；
      2) 返回 config/robots/<active_config>.yaml 的绝对路径。

    选择器缺失 / 条目缺失或非法时抛出 RuntimeError。
    """
    selector_path = resolve_path("config/selector.yaml")
    if not os.path.isfile(selector_path):
        raise RuntimeError(f"找不到机器配置选择器: {selector_path}")
    with open(selector_path, "r", encoding="utf-8") as f:
        selector = yaml.safe_load(f) or {}
    if not isinstance(selector, dict) or "active_config" not in selector:
        raise RuntimeError(
            f"机器配置选择器 '{selector_path}' 缺少 'active_config' 条目"
            f"（应填写 config/robots/ 下的配置文件名，不含 .yaml 后缀）")
    name = str(selector["active_config"]).strip()
    # 仅允许纯文件名（不含路径分隔符 / ..），避免越出 config/robots/ 目录
    if not name or "/" in name or "\\" in name or ".." in name:
        raise RuntimeError(f"机器配置选择器 'active_config' 非法: '{name}'")
    cfg_path = resolve_path(os.path.join("config", "robots", name + ".yaml"))
    if not os.path.isfile(cfg_path):
        raise RuntimeError(f"机器配置选择器指向的配置文件不存在: {cfg_path}"
                          f"（请检查 config/robots/ 下的文件名与 active_config 是否一致）")
    return cfg_path


if __name__ == "__main__":
    print("executable:", get_executable_path())
    print("root:      ", get_project_root())
    print("config:    ", resolve_machine_config_path())
