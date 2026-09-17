#!/usr/bin/env python3
"""Make Infantry 0726 batch/height/width dynamic, retaining the 21-channel layout.

Like modify_0526_dynamic.py, patch the graph, clear stale intermediate shapes,
and validate against the original. Generate spatial grids and strides from the three detection feature maps.
Reshape dimension 0 copies the corresponding input dimension (allowzero=0),
so [1,C,-1] becomes [0,C,-1], including the [1,4,2,-1] keypoint reshape.
The original ONNX is never overwritten. Default validation runs CPU inference
on four different inputs, comparing batches 1/2/3/4 with single-image results.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Model/Armor/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx"
TARGET = ROOT / "Model/Armor/0726_dynamic_hw.onnx"


GRID = "/model.22/Constant_15_output_0"
STRIDES = "/model.22/Constant_16_output_0"


def static_grids(height, width):
    grids, strides = [], []
    for stride in (8, 16, 32):
        y, x = np.mgrid[:height // stride, :width // stride]
        grids.append(np.stack([x.ravel(), y.ravel()]))
        strides.append(np.full((1, x.size), stride))
    return np.concatenate(grids, axis=1).astype(np.float16), np.concatenate(strides, axis=1).astype(np.float16)


def dynamic_grids(model):
    """Replace FP16 [2,6300] grid / [1,6300] stride with Shape/Range/Expand."""
    values = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    for name, expected in zip((GRID, STRIDES), static_grids(480, 640)):
        np.testing.assert_array_equal(values[name], expected)
    def const(name, value):
        name = "0726_hw/" + name
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), name))
        return name
    zero, one = const("zero", 0), const("one", 1)
    idx_h, idx_w = const("idx_h", 2), const("idx_w", 3)
    axes0 = const("axes0", [0])
    row_shape, col_shape = const("row_shape", [1, -1]), const("col_shape", [-1, 1])
    grid_parts, stride_parts = [], []
    def node(op, inputs, tag, **attrs):
        output = "0726_hw/" + tag
        model.graph.node.append(helper.make_node(op, inputs, [output], name=output, **attrs))
        return output
    for i, stride in enumerate((8, 16, 32)):
        tag = f"s{stride}/"
        feature = f"/model.22/cv4.{i}/cv4.{i}.2/Conv_output_0"
        shape = node("Shape", [feature], tag + "shape")
        h = node("Gather", [shape, idx_h], tag + "h", axis=0)
        w = node("Gather", [shape, idx_w], tag + "w", axis=0)
        hv = node("Unsqueeze", [h, axes0], tag + "hv")
        wv = node("Unsqueeze", [w, axes0], tag + "wv")
        hw = node("Concat", [hv, wv], tag + "hw", axis=0)
        x = node("Range", [zero, w, one], tag + "x")
        y = node("Range", [zero, h, one], tag + "y")
        x = node("Reshape", [x, row_shape], tag + "xrow")
        y = node("Reshape", [y, col_shape], tag + "ycol")
        x = node("Expand", [x, hw], tag + "xgrid")
        y = node("Expand", [y, hw], tag + "ygrid")
        x = node("Reshape", [x, row_shape], tag + "xflat")
        y = node("Reshape", [y, row_shape], tag + "yflat")
        grid = node("Concat", [x, y], tag + "grid_int", axis=0)
        grid_parts.append(node("Cast", [grid], tag + "grid", to=TensorProto.FLOAT16))
        flat_shape = node("Shape", [x], tag + "flat_shape")
        stride_value = const(tag + "stride_value", stride)
        stride_tensor = node("Expand", [stride_value, flat_shape], tag + "stride_int")
        stride_parts.append(node("Cast", [stride_tensor], tag + "strides", to=TensorProto.FLOAT16))
    dynamic_grid = node("Concat", grid_parts, "grid", axis=1)
    dynamic_stride = node("Concat", stride_parts, "strides", axis=1)
    for n in model.graph.node:
        for i, value in enumerate(n.input):
            if value == GRID: n.input[i] = dynamic_grid
            if value == STRIDES: n.input[i] = dynamic_stride
    keep = [t for t in model.graph.initializer if t.name not in (GRID, STRIDES)]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)


def convert(source, target):
    if source.resolve() == target.resolve():
        raise ValueError("Input and output paths must differ")
    model = onnx.load(str(source))
    def dims(value):
        return [d.dim_value for d in value.type.tensor_type.shape.dim]
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Expected one input and one output")
    if dims(model.graph.input[0]) != [1, 3, 480, 640] or dims(model.graph.output[0]) != [1, 21, 6300]:
        raise ValueError("This converter only accepts the original static 0726 export")
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attr.t)
    count = 0
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        shape = constants.get(node.input[1])
        if shape is None or shape.tolist() not in ([1, 8, -1], [1, 4, -1], [1, 9, -1], [1, 4, 2, -1]):
            raise ValueError(f"Unexpected Reshape contract: {node.name}")
        if any(a.name == "allowzero" and a.i != 0 for a in node.attribute):
            raise ValueError("Reshape must use allowzero=0")
        shape = shape.copy()
        shape[0] = 0
        name = f"0726_batch_shape_{count}"
        if name in constants:
            raise ValueError("Generated initializer name collision")
        model.graph.initializer.append(numpy_helper.from_array(shape, name))
        node.input[1] = name
        count += 1
    if count != 11:
        raise ValueError(f"Expected 11 Reshape nodes, found {count}")
    for value in list(model.graph.input) + list(model.graph.output):
        value.type.tensor_type.shape.dim[0].dim_param = "batch"
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = "height"
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = "width"
    model.graph.output[0].type.tensor_type.shape.dim[2].dim_param = "num_anchors"
    dynamic_grids(model)
    del model.graph.value_info[:]
    metadata = {p.key: p.value for p in model.metadata_props}
    metadata.update(batch="dynamic", dynamic_batch="true", imgsz="[height, width]", dynamic_spatial="true")
    if "args" in metadata:
        import ast
        args = ast.literal_eval(metadata["args"])
        args.update(batch="dynamic", dynamic=True, dynamic_spatial=True)
        metadata["args"] = repr(args)
    onnx.helper.set_model_props(model, metadata)
    # Original mixed-precision export places input Cast nodes after consumers.
    # Stable topological ordering changes no operations or weights.
    known = {v.name for v in model.graph.input} | {v.name for v in model.graph.initializer} | {""}
    pending = list(model.graph.node)
    ordered = []
    while pending:
        ready = [n for n in pending if all(x in known for x in n.input)]
        if not ready:
            raise ValueError("Graph contains a cycle or missing input")
        for node in ready:
            ordered.append(node)
            known.update(node.output)
        ready_names = {n.output[0] for n in ready}
        pending = [n for n in pending if n.output[0] not in ready_names]
    del model.graph.node[:]
    model.graph.node.extend(ordered)
    onnx.checker.check_model(model, full_check=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(target))
    print(f"Patched {count} Reshape nodes: {target}", flush=True)


def verify(source, target):
    import openvino as ov
    core = ov.Core()
    settings = {"INFERENCE_PRECISION_HINT": "f32", "INFERENCE_NUM_THREADS": 2}
    reports = []
    for height, width in ((480, 640), (320, 320), (384, 512), (640, 480), (640, 640)):
        # Independent static reference: specialize the original export's constants
        # with numpy grids. Its Conv/Reshape weights and operators remain original.
        reference_model = onnx.load(str(source))
        reference_model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = height
        reference_model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = width
        grids, strides = static_grids(height, width)
        count = grids.shape[1]
        reference_model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = count
        del reference_model.graph.value_info[:]
        for t in reference_model.graph.initializer:
            if t.name == GRID: t.CopyFrom(numpy_helper.from_array(grids, GRID))
            elif t.name == STRIDES: t.CopyFrom(numpy_helper.from_array(strides, STRIDES))
        original = core.compile_model(core.read_model(reference_model.SerializeToString()), "CPU", settings)
        inputs = np.random.default_rng(726).random((4, 3, height, width), dtype=np.float32)
        reference = np.concatenate([np.array(original(x[None])[original.output(0)], copy=True)
                                    for x in inputs], axis=0)
        for batch in (1, 2, 3, 4):
            model = core.read_model(str(target))
            model.reshape({model.input().get_any_name(): [batch, 3, height, width]})
            compiled = core.compile_model(model, "CPU", settings)
            actual = np.array(compiled(inputs[:batch])[compiled.output(0)], copy=True)
            assert actual.shape == (batch, 21, count), actual.shape
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, reference[:batch], atol=1e-3, rtol=1e-4)
            reports.append({"batch": batch, "width": width, "height": height,
                            "shape": list(actual.shape),
                            "max_score_diff": float(np.max(np.abs(actual[:, :13] - reference[:batch, :13]))),
                            "max_xy_diff": float(np.max(np.abs(actual[:, 13:] - reference[:batch, 13:]))),
                            "passed": True})
            print(json.dumps(reports[-1]), flush=True)
    return {"openvino": ov.__version__, "device": "CPU", "precision": "f32",
            "reference": "original graph with independently generated static numpy grids",
            "seed": 726, "atol": 1e-3, "rtol": 1e-4, "results": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=TARGET)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    convert(args.input, args.output)
    if not args.no_verify:
        report = verify(args.input, args.output)
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
