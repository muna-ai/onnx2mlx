#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import mlx.core as mx
import mlx.nn as nn
from . import register
from .._utils import onnx_pads_to_mlx

@register("MaxPool")
def max_pool(inputs, attrs):
    x = inputs[0]
    kernel_shape, strides, pads, auto_pad, ceil_mode = _get_pool_params(attrs)
    padding = _pool_padding(auto_pad, pads)
    ndim = x.ndim - 2
    if ndim == 1:
        x = mx.transpose(x, (0, 2, 1))
        pool = nn.MaxPool1d(kernel_size=kernel_shape[0], stride=strides[0], padding=padding)
        y = pool(x)
        y = mx.transpose(y, (0, 2, 1))
    elif ndim == 2:
        x = mx.transpose(x, (0, 2, 3, 1))
        ks = tuple(kernel_shape) if len(kernel_shape) > 1 else kernel_shape[0]
        st = tuple(strides) if len(strides) > 1 else strides[0]
        pool = nn.MaxPool2d(kernel_size=ks, stride=st, padding=padding)
        y = pool(x)
        y = mx.transpose(y, (0, 3, 1, 2))
    else:
        raise NotImplementedError(f"MaxPool with {ndim}D spatial dims is not supported")
    return [y]

@register("AveragePool")
def average_pool(inputs, attrs):
    x = inputs[0]
    kernel_shape, strides, pads, auto_pad, ceil_mode = _get_pool_params(attrs)
    padding = _pool_padding(auto_pad, pads)
    ndim = x.ndim - 2
    if ndim == 1:
        x = mx.transpose(x, (0, 2, 1))
        pool = nn.AvgPool1d(kernel_size=kernel_shape[0], stride=strides[0], padding=padding)
        y = pool(x)
        y = mx.transpose(y, (0, 2, 1))
    elif ndim == 2:
        x = mx.transpose(x, (0, 2, 3, 1))
        ks = tuple(kernel_shape) if len(kernel_shape) > 1 else kernel_shape[0]
        st = tuple(strides) if len(strides) > 1 else strides[0]
        pool = nn.AvgPool2d(kernel_size=ks, stride=st, padding=padding)
        y = pool(x)
        y = mx.transpose(y, (0, 3, 1, 2))
    else:
        raise NotImplementedError(f"AveragePool with {ndim}D spatial dims is not supported")
    return [y]

@register("GlobalAveragePool")
def global_average_pool(inputs, attrs):
    x = inputs[0]
    spatial_axes = tuple(range(2, x.ndim))
    return [mx.mean(x, axis=spatial_axes, keepdims=True)]

@register("GlobalMaxPool")
def global_max_pool(inputs, attrs):
    x = inputs[0]
    spatial_axes = tuple(range(2, x.ndim))
    return [mx.max(x, axis=spatial_axes, keepdims=True)]

def _get_pool_params(attrs):
    kernel_shape = attrs.get("kernel_shape", [1])
    strides = attrs.get("strides", [1])
    pads = attrs.get("pads", None)
    auto_pad = attrs.get("auto_pad", "NOTSET")
    ceil_mode = attrs.get("ceil_mode", 0)
    return kernel_shape, strides, pads, auto_pad, ceil_mode

def _pool_padding(auto_pad, pads):
    if auto_pad == "VALID":
        return 0
    if pads is not None:
        pairs = onnx_pads_to_mlx(pads)
        if all(p[0] == p[1] for p in pairs):
            symmetric = tuple(p[0] for p in pairs)
            return symmetric[0] if len(symmetric) == 1 else symmetric
        return pairs
    return 0