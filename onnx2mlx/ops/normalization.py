#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import mlx.core as mx
from . import register

@register("BatchNormalization")
def batch_normalization(inputs, attrs, ctx):
    x = inputs[0]
    scale = inputs[1]
    bias = inputs[2]
    mean = inputs[3]
    var = inputs[4]
    epsilon = attrs.get("epsilon", 1e-5)
    # ONNX: x is (N, C, ...), normalize over C dimension
    # Reshape scale/bias/mean/var to broadcast: (1, C, 1, 1, ...)
    shape = [1, -1] + [1] * (x.ndim - 2)
    scale = mx.reshape(scale, shape)
    bias = mx.reshape(bias, shape)
    mean = mx.reshape(mean, shape)
    var = mx.reshape(var, shape)
    y = mx.add(mx.multiply(mx.multiply(mx.subtract(x, mean), mx.rsqrt(mx.add(var, epsilon))), scale), bias)
    return [y]

@register("LayerNormalization")
def layer_normalization(inputs, attrs, ctx):
    x = inputs[0]
    scale = inputs[1]
    bias = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
    axis = attrs.get("axis", -1)
    epsilon = attrs.get("epsilon", 1e-5)
    axes = list(range(axis if axis >= 0 else x.ndim + axis, x.ndim))
    mean = mx.mean(x, axis=axes, keepdims=True)
    var = mx.mean(mx.square(mx.subtract(x, mean)), axis=axes, keepdims=True)
    y = mx.multiply(mx.multiply(mx.subtract(x, mean), mx.rsqrt(mx.add(var, epsilon))), scale)
    if bias is not None:
        y = mx.add(y, bias)
    return [y]

@register("InstanceNormalization")
def instance_normalization(inputs, attrs, ctx):
    x = inputs[0]
    scale = inputs[1]
    bias = inputs[2]
    epsilon = attrs.get("epsilon", 1e-5)
    # Normalize over spatial dims (2, 3, ...)
    spatial_axes = list(range(2, x.ndim))
    mean = mx.mean(x, axis=spatial_axes, keepdims=True)
    var = mx.mean(mx.square(mx.subtract(x, mean)), axis=spatial_axes, keepdims=True)
    shape = [1, -1] + [1] * (x.ndim - 2)
    scale = mx.reshape(scale, shape)
    bias = mx.reshape(bias, shape)
    y = mx.add(mx.multiply(mx.multiply(mx.subtract(x, mean), mx.rsqrt(mx.add(var, epsilon))), scale), bias)
    return [y]

@register("GroupNormalization")
def group_normalization(inputs, attrs, ctx):
    x = inputs[0]
    scale = inputs[1]
    bias = inputs[2]
    epsilon = attrs.get("epsilon", 1e-5)
    num_groups = attrs.get("num_groups", 1)
    N, C = x.shape[0], x.shape[1]
    spatial = x.shape[2:]
    channels_per_group = C // num_groups
    x_grouped = mx.reshape(x, [N, num_groups, channels_per_group] + list(spatial))
    axes = list(range(2, x_grouped.ndim))
    mean = mx.mean(x_grouped, axis=axes, keepdims=True)
    var = mx.mean(mx.square(mx.subtract(x_grouped, mean)), axis=axes, keepdims=True)
    x_norm = mx.multiply(mx.subtract(x_grouped, mean), mx.rsqrt(mx.add(var, epsilon)))
    x_norm = mx.reshape(x_norm, x.shape)
    shape = [1, -1] + [1] * (x.ndim - 2)
    y = mx.add(mx.multiply(x_norm, mx.reshape(scale, shape)), mx.reshape(bias, shape))
    return [y]