#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import io
import pytest
import numpy as np
import torch
import onnx
import onnxruntime as ort
import mlx.core as mx

from onnx2mlx import onnx2mlx

def test_resnet18():
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    onnx_model, input_np = _export_to_onnx(model)
    _compare(onnx_model, input_np, rtol=1e-3, atol=1e-3)

def test_mobilenet_v2():
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    onnx_model, input_np = _export_to_onnx(model)
    _compare(onnx_model, input_np, rtol=1e-3, atol=1e-3)

def test_squeezenet1_1():
    from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
    model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
    onnx_model, input_np = _export_to_onnx(model)
    _compare(onnx_model, input_np, rtol=1e-3, atol=1e-3)

def _run_onnxruntime(onnx_model, input_np):
    buf = io.BytesIO()
    onnx.save(onnx_model, buf)
    session = ort.InferenceSession(buf.getvalue())
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_np})[0]

def _run_onnx2mlx(onnx_model, input_np):
    fn = onnx2mlx(onnx_model)
    mlx_input = mx.array(input_np)
    result = fn(mlx_input)
    mx.eval(result)
    return np.array(result)

def _compare(onnx_model, input_np, rtol=1e-4, atol=1e-4):
    expected = _run_onnxruntime(onnx_model, input_np)
    actual = _run_onnx2mlx(onnx_model, input_np)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

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