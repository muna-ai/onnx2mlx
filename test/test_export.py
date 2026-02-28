#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import io
import os
import tempfile
import numpy as np
import torch
import onnx
import onnxruntime as ort
import mlx.core as mx

from onnx2mlx import onnx2mlx

def _export_to_onnx(model, input_shape=(1, 3, 224, 224), opset_version=17):
    model.eval()
    dummy = torch.randn(*input_shape)
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        dummy,
        buf,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )
    buf.seek(0)
    return onnx.load(buf), dummy.numpy()

def _run_onnxruntime(onnx_model, input_np):
    buf = io.BytesIO()
    onnx.save(onnx_model, buf)
    session = ort.InferenceSession(buf.getvalue())
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_np})[0]


def test_export_resnet18():
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    onnx_model, input_np = _export_to_onnx(model)
    expected = _run_onnxruntime(onnx_model, input_np)
    fn = onnx2mlx(onnx_model)
    mlx_input = mx.array(input_np)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "resnet18.mlxfn")
        mx.export_function(path, fn, mlx_input)
        imported_fn = mx.import_function(path)
        result = imported_fn(mlx_input)
        mx.eval(result)
        np.testing.assert_allclose(np.array(result[0]), expected, rtol=1e-3, atol=1e-3)

def test_export_squeezenet1_1():
    from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
    model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
    onnx_model, input_np = _export_to_onnx(model)
    expected = _run_onnxruntime(onnx_model, input_np)
    fn = onnx2mlx(onnx_model)
    mlx_input = mx.array(input_np)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "squeezenet.mlxfn")
        mx.export_function(path, fn, mlx_input)
        imported_fn = mx.import_function(path)
        result = imported_fn(mlx_input)
        mx.eval(result)
        np.testing.assert_allclose(np.array(result[0]), expected, rtol=1e-3, atol=1e-3)

def test_export_mobilenet_v2():
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    onnx_model, input_np = _export_to_onnx(model)
    expected = _run_onnxruntime(onnx_model, input_np)
    fn = onnx2mlx(onnx_model)
    mlx_input = mx.array(input_np)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mobilenet_v2.mlxfn")
        mx.export_function(path, fn, mlx_input)
        imported_fn = mx.import_function(path)
        result = imported_fn(mlx_input)
        mx.eval(result)
        np.testing.assert_allclose(np.array(result[0]), expected, rtol=1e-3, atol=1e-3)

def test_export_nomic_layout_v1():
    from transformers import RTDetrForObjectDetection
    model = RTDetrForObjectDetection.from_pretrained("muna-ai/nomic-layout-v1").eval()
    torch.manual_seed(42)
    dummy = torch.randn(1, 3, 800, 800)
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        dummy,
        buf,
        opset_version=17,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamo=False,
    )
    buf.seek(0)
    onnx_model = onnx.load(buf)
    input_np = dummy.numpy()
    # Run with ONNX Runtime
    buf = io.BytesIO()
    onnx.save(onnx_model, buf)
    session = ort.InferenceSession(buf.getvalue())
    ort_outputs = session.run(None, {"pixel_values": input_np})
    ort_logits = ort_outputs[0]
    ort_boxes = ort_outputs[1]
    # Convert and export
    fn = onnx2mlx(onnx_model)
    mlx_input = mx.array(input_np)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nomic_layout.mlxfn")
        mx.export_function(path, fn, mlx_input)
        imported_fn = mx.import_function(path)
        result = imported_fn(mlx_input)
        mx.eval(result)
        mlx_logits = np.array(result[0])
        mlx_boxes = np.array(result[1])
    # Sort along query axis to handle TopK tie-breaking differences
    mlx_order = np.argsort(mlx_logits[:, :, 0], axis=1)
    ort_order = np.argsort(ort_logits[:, :, 0], axis=1)
    mlx_logits = np.take_along_axis(mlx_logits, mlx_order[:, :, None], axis=1)
    ort_logits = np.take_along_axis(ort_logits, ort_order[:, :, None], axis=1)
    mlx_boxes = np.take_along_axis(mlx_boxes, mlx_order[:, :, None], axis=1)
    ort_boxes = np.take_along_axis(ort_boxes, ort_order[:, :, None], axis=1)
    np.testing.assert_allclose(mlx_logits, ort_logits, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(mlx_boxes, ort_boxes, rtol=1e-3, atol=1e-3)
