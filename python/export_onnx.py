#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_onnx.py — 把训练好的 YOLO-pose 权重（.pt）导出为 ONNX。

移植自 Power_Rune_Train/export_onnx.py，并按本工程（UnifiedAutoAimPipeline）的
约定做了适配：
  * 项目根目录与模型路径统一走 python/path_resolver.py
    （include/common/PathResolver.h 的 Python 翻译版），模型路径相对项目根目录解析，
    不再依赖训练工程的 ROOT / runs/ 目录，也就不需要 power_rune_common.py；
  * 不接收任何命令行参数：模型名以常量 MODEL_NAME 固定在代码中，
    想换模型只需手动改脚本顶部那一行；导出参数同样以常量形式固定在代码中；
  * 默认处理 Model/PowerRune/power_rune_finetune2（相对项目根目录）：
        输入  Model/PowerRune/power_rune_finetune2.pt
        输出  Model/PowerRune/power_rune_finetune2.onnx（与权重同名，不覆盖 power_rune.onnx）

导出参数与原脚本 / Power_Rune_Auto_Aim/update_onnx.py 保持一致：
    dynamic + simplify + opset 11，不带 NMS（yolo26）。

用法（无参数）：
    python3 python/export_onnx.py

部署提示：C++ 侧读取 Model/PowerRune/power_rune.onnx，导出后若要直接部署：
    cp Model/PowerRune/power_rune_finetune2.onnx Model/PowerRune/power_rune.onnx
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# 使脚本可直接运行（python3 python/export_onnx.py）也能被 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_resolver import get_project_root, resolve_path  # noqa: E402

ROOT = Path(get_project_root())
# ultralytics / matplotlib 的缓存目录放到工程 cache/ 下（该目录已在 .gitignore 中）
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache" / "matplotlib"))

# ==================== 要处理的模型（手动改这里即可切换） ====================
MODEL_DIR = "Model/PowerRune"          # 相对项目根目录
MODEL_NAME = "power_rune_finetune"    # 不含扩展名：读 <MODEL_NAME>.pt，写 <MODEL_NAME>.onnx

# ==================== 导出参数（与 update_onnx.py 一致） ====================
IMGSZ = 640
OPSET = 11
DYNAMIC = True         # 动态输入尺寸（默认开启）
SIMPLIFY = True
HALF = False

# 可选：设为图片路径时，导出前先用 .pt 模型推理一张图，把标注结果存到输出目录
TEST_IMAGE = None
SHOW_PREVIEW = False   # 配合 TEST_IMAGE 弹窗显示（无显示环境请保持 False）


def model_paths() -> "tuple[Path, Path]":
    """返回当前模型的 (weights.pt, onnx) 绝对路径（均经 PathResolver 解析）。"""
    stem = os.path.join(MODEL_DIR, MODEL_NAME)
    return Path(resolve_path(stem + ".pt")), Path(resolve_path(stem + ".onnx"))


def pick_weights(weights: Path) -> Path:
    """校验权重存在并返回其绝对路径。"""
    if not weights.is_file():
        raise FileNotFoundError(
            f"权重不存在：{weights}\n"
            f"（请把 {MODEL_NAME}.pt 放到 {MODEL_DIR}/ 下，或修改脚本顶部的 MODEL_NAME）")
    return weights.resolve()


def preview_pt(model, test_image: str, save_path: Path, show: bool) -> None:
    """导出前用 .pt 模型跑一张图，保存标注结果（可选弹窗显示）。"""
    import cv2

    img = cv2.imread(test_image)
    if img is None:
        print(f"[预览] 无法读取图片：{test_image}，跳过预览")
        return
    print(f"[预览] 用 .pt 模型推理 {test_image} (conf=0.5) ...")
    results = model(img, conf=0.5)
    out = results[0].plot()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), out)
    print(f"[预览] 标注结果已保存：{save_path}")
    if show:
        try:
            cv2.imshow("preview", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:  # noqa: BLE001  （无显示环境直接跳过）
            print(f"[预览] 弹窗显示不可用（{e}），已自动跳过。")


def verify_onnx(onnx_path: Path) -> None:
    """结构校验：能 load 并通过 onnx.checker 即可。"""
    try:
        import onnx
    except Exception:  # noqa: BLE001
        print("[校验] 未安装 onnx，跳过结构校验。")
        return
    try:
        m = onnx.load(str(onnx_path))
        onnx.checker.check_model(m)
        print(f"[校验] ONNX 结构检查通过（{onnx_path.stat().st_size / 1e6:.1f} MB）")
    except Exception as e:  # noqa: BLE001
        print(f"[校验] ONNX 结构检查失败：{e}")


def main() -> int:
    weights, onnx_path = model_paths()
    weights = pick_weights(weights)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print("== 导出 ONNX ==")
    print(f"   模型: {MODEL_DIR}/{MODEL_NAME}")
    print(f"   权重: {weights}")
    print(f"   输出: {onnx_path}")

    from ultralytics import YOLO

    model = YOLO(str(weights)).to("cpu")

    # 可选：导出前先看一张图的效果（对应原脚本的预览步骤）
    if TEST_IMAGE:
        preview_pt(model, TEST_IMAGE, onnx_path.with_name(MODEL_NAME + "_preview.jpg"),
                   SHOW_PREVIEW)

    # 清理旧的同名导出文件（onnx 及可能的 sidecar），再导出
    for suffix in (".onnx", ".bin", ".xml"):
        old = onnx_path.with_suffix(suffix)
        if old.exists():
            old.unlink()

    # 导出参数与 Power_Rune_Auto_Aim/update_onnx.py 保持一致
    print("[导出] 开始导出 ONNX ...")
    exported = model.export(
        format="onnx",
        imgsz=IMGSZ,
        half=HALF,
        dynamic=DYNAMIC,
        simplify=SIMPLIFY,
        opset=OPSET,
        int8=False,
        nms=False,            # yolo26 不带 NMS
        agnostic_nms=False,
        iou=0.7,
        conf=0.25,
        max_det=300,
    )

    # ultralytics 把 onnx 写在源权重同目录（此处与目标同名同目录），兜底搬移
    produced = Path(exported) if exported and Path(exported).exists() else weights.with_suffix(".onnx")
    if produced.exists() and produced.resolve() != onnx_path.resolve():
        if onnx_path.exists():
            onnx_path.unlink()
        shutil.move(str(produced), str(onnx_path))
    print(f"ONNX exported to: {onnx_path}")
    verify_onnx(onnx_path)
    print("\n导出完成。C++ 侧读取 Model/PowerRune/power_rune.onnx，如需直接部署：")
    print(f"  cp {os.path.relpath(onnx_path, ROOT)} Model/PowerRune/power_rune.onnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
