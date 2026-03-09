#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import mlx.core as mx

from ..context import ConvertContext
from . import register

@register("ReduceSum")
def reduce_sum(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    axes = _reduce_axes(inputs, attrs)
    keepdims = bool(attrs.get("keepdims", 1))
    return [mx.sum(inputs[0], axis=axes, keepdims=keepdims)]

@register("ReduceMean")
def reduce_mean(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    axes = _reduce_axes(inputs, attrs)
    keepdims = bool(attrs.get("keepdims", 1))
    return [mx.mean(inputs[0], axis=axes, keepdims=keepdims)]

@register("ReduceMax")
def reduce_max(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    axes = _reduce_axes(inputs, attrs)
    keepdims = bool(attrs.get("keepdims", 1))
    return [mx.max(inputs[0], axis=axes, keepdims=keepdims)]

@register("ReduceMin")
def reduce_min(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    axes = _reduce_axes(inputs, attrs)
    keepdims = bool(attrs.get("keepdims", 1))
    return [mx.min(inputs[0], axis=axes, keepdims=keepdims)]

@register("ReduceProd")
def reduce_prod(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    axes = _reduce_axes(inputs, attrs)
    keepdims = bool(attrs.get("keepdims", 1))
    return [mx.prod(inputs[0], axis=axes, keepdims=keepdims)]

@register("ArgMax")
def argmax(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    axis = attrs.get("axis", 0)
    keepdims = bool(attrs.get("keepdims", 1))
    return [mx.argmax(inputs[0], axis=axis, keepdims=keepdims)]

@register("ArgMin")
def argmin(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    axis = attrs.get("axis", 0)
    keepdims = bool(attrs.get("keepdims", 1))
    return [mx.argmin(inputs[0], axis=axis, keepdims=keepdims)]

@register("ReduceLogSumExp")
def reduce_logsumexp(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    axes = _reduce_axes(inputs, attrs)
    keepdims = bool(attrs.get("keepdims", 1))
    return [mx.logsumexp(inputs[0], axis=axes, keepdims=keepdims)]

@register("TopK")
def topk(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    x = inputs[0]
    k = inputs[1].item()
    axis = attrs.get("axis", -1)
    largest = bool(attrs.get("largest", 1))
    sorted_ = bool(attrs.get("sorted", 1))
    n = x.shape[axis]
    sort_x = mx.negative(x) if not largest else x
    # Get top-k indices via argpartition (O(n) average)
    kth = n - k
    indices = mx.argpartition(sort_x, kth=kth, axis=axis)
    indices = mx.take(indices, mx.arange(kth, n), axis=axis)
    if sorted_:
        # Sort the k candidates in descending order
        topk_vals = mx.take_along_axis(sort_x, indices, axis=axis)
        order = mx.argsort(mx.negative(topk_vals), axis=axis)
        indices = mx.take_along_axis(indices, order, axis=axis)
    values = mx.take_along_axis(x, indices, axis=axis)
    return [values, indices]

@register("CumSum")
def cumsum(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    x = inputs[0]
    axis = inputs[1].item()
    exclusive = bool(attrs.get("exclusive", 0))
    reverse = bool(attrs.get("reverse", 0))
    if reverse:
        x = mx.flip(x, axis=axis)
    if exclusive:
        x = mx.cumsum(x, axis=axis)
        # Shift forward and fill first element with 0
        slices = [slice(None)] * x.ndim
        slices[axis] = slice(0, -1)
        zero_shape = list(x.shape)
        zero_shape[axis] = 1
        x = mx.concatenate([mx.zeros(zero_shape, dtype=x.dtype), x[tuple(slices)]], axis=axis)
    else:
        x = mx.cumsum(x, axis=axis)
    if reverse:
        x = mx.flip(x, axis=axis)
    return [x]

def _reduce_axes(inputs, attrs):
    if len(inputs) > 1 and inputs[1] is not None:
        return inputs[1].tolist()
    axes = attrs.get("axes", None)
    return list(axes) if axes is not None else None