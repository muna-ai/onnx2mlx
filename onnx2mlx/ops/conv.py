#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import mlx.core as mx
from . import register
from .._utils import onnx_pads_to_mlx

@register("Conv")
def conv(inputs, attrs):
    x = inputs[0]
    w = inputs[1]
    b = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
    strides, dilations, group, pads, auto_pad = _get_conv_params(attrs)
    padding = _compute_padding(auto_pad, pads, x.ndim - 2)
    ndim = w.ndim - 2
    if ndim == 1:
        # ONNX: (N, C, L) -> MLX: (N, L, C)
        x = mx.transpose(x, (0, 2, 1))
        # ONNX: (Cout, Cin/g, K) -> MLX: (Cout, K, Cin/g)
        w = mx.transpose(w, (0, 2, 1))
        y = mx.conv1d(x, w, stride=strides[0], padding=padding, dilation=dilations[0], groups=group)
        if b is not None:
            y = mx.add(y, mx.reshape(b, (1, 1, -1)))
        y = mx.transpose(y, (0, 2, 1))
    elif ndim == 2:
        # ONNX: (N, C, H, W) -> MLX: (N, H, W, C)
        x = mx.transpose(x, (0, 2, 3, 1))
        # ONNX: (Cout, Cin/g, kH, kW) -> MLX: (Cout, kH, kW, Cin/g)
        w = mx.transpose(w, (0, 2, 3, 1))
        stride = tuple(strides) if len(strides) > 1 else strides[0]
        dilation = tuple(dilations) if len(dilations) > 1 else dilations[0]
        y = mx.conv2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=group)
        if b is not None:
            y = mx.add(y, mx.reshape(b, (1, 1, 1, -1)))
        y = mx.transpose(y, (0, 3, 1, 2))
    elif ndim == 3:
        # ONNX: (N, C, D, H, W) -> MLX: (N, D, H, W, C)
        x = mx.transpose(x, (0, 2, 3, 4, 1))
        # ONNX: (Cout, Cin/g, kD, kH, kW) -> MLX: (Cout, kD, kH, kW, Cin/g)
        w = mx.transpose(w, (0, 2, 3, 4, 1))
        stride = tuple(strides)
        dilation = tuple(dilations)
        y = mx.conv3d(x, w, stride=stride, padding=padding, dilation=dilation, groups=group)
        if b is not None:
            y = mx.add(y, mx.reshape(b, (1, 1, 1, 1, -1)))
        y = mx.transpose(y, (0, 4, 1, 2, 3))
    else:
        raise NotImplementedError(f"Conv with {ndim}D spatial dims is not supported")
    return [y]

@register("ConvTranspose")
def conv_transpose(inputs, attrs):
    x = inputs[0]
    w = inputs[1]
    b = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
    strides, dilations, group, pads, auto_pad = _get_conv_params(attrs)
    padding = _compute_padding(auto_pad, pads, x.ndim - 2)
    ndim = w.ndim - 2
    if ndim == 1:
        x = mx.transpose(x, (0, 2, 1))
        # ONNX ConvTranspose: (Cin, Cout/g, K) -> MLX: (Cout/g, K, Cin) -- note ONNX swaps in/out for transpose
        w = mx.transpose(w, (1, 2, 0))
        y = mx.conv_transpose1d(x, w, stride=strides[0], padding=padding, dilation=dilations[0], groups=group)
        if b is not None:
            y = mx.add(y, mx.reshape(b, (1, 1, -1)))
        y = mx.transpose(y, (0, 2, 1))
    elif ndim == 2:
        x = mx.transpose(x, (0, 2, 3, 1))
        # ONNX ConvTranspose: (Cin, Cout/g, kH, kW) -> MLX: (Cout/g, kH, kW, Cin)
        w = mx.transpose(w, (1, 2, 3, 0))
        stride = tuple(strides) if len(strides) > 1 else strides[0]
        dilation = tuple(dilations) if len(dilations) > 1 else dilations[0]
        y = mx.conv_transpose2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=group)
        if b is not None:
            y = mx.add(y, mx.reshape(b, (1, 1, 1, -1)))
        y = mx.transpose(y, (0, 3, 1, 2))
    elif ndim == 3:
        x = mx.transpose(x, (0, 2, 3, 4, 1))
        w = mx.transpose(w, (1, 2, 3, 4, 0))
        stride = tuple(strides)
        dilation = tuple(dilations)
        y = mx.conv_transpose3d(x, w, stride=stride, padding=padding, dilation=dilation, groups=group)
        if b is not None:
            y = mx.add(y, mx.reshape(b, (1, 1, 1, 1, -1)))
        y = mx.transpose(y, (0, 4, 1, 2, 3))
    else:
        raise NotImplementedError(f"ConvTranspose with {ndim}D spatial dims is not supported")
    return [y]

def _get_conv_params(attrs):
    strides = attrs.get("strides", [1])
    dilations = attrs.get("dilations", [1])
    group = attrs.get("group", 1)
    pads = attrs.get("pads", None)
    auto_pad = attrs.get("auto_pad", "NOTSET")
    return strides, dilations, group, pads, auto_pad

def _compute_padding(auto_pad, pads, ndim):
    """
    Compute MLX-compatible padding from ONNX padding spec.
    """
    if auto_pad == "VALID":
        return 0
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        return "same"
    if pads is not None:
        pairs = onnx_pads_to_mlx(pads)
        if all(p[0] == p[1] for p in pairs):
            symmetric = tuple(p[0] for p in pairs)
            return symmetric[0] if len(symmetric) == 1 else symmetric
        return pairs
    return 0