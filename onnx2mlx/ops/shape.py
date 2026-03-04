#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import mlx.core as mx
from . import register

@register("Reshape")
def reshape(inputs, attrs):
    data = inputs[0]
    shape = inputs[1]
    shape_list = shape.tolist()
    allowzero = int(attrs.get("allowzero", 0))
    if not allowzero:
        shape_list = [
            data.shape[i] if s == 0 and i < data.ndim else s
            for i, s in enumerate(shape_list)
        ]
    return [mx.reshape(data, shape_list)]

@register("Transpose")
def transpose(inputs, attrs):
    perm = attrs.get("perm", None)
    if perm is not None:
        perm = list(perm)
    return [mx.transpose(inputs[0], axes=perm)]

@register("Flatten")
def flatten(inputs, attrs):
    axis = attrs.get("axis", 1)
    return [mx.flatten(inputs[0], start_axis=axis)]

@register("Squeeze")
def squeeze(inputs, attrs):
    x = inputs[0]
    if len(inputs) > 1 and inputs[1] is not None:
        axes = inputs[1].tolist()
    else:
        axes = attrs.get("axes", None)
    if axes is not None:
        if isinstance(axes, int):
            axes = [axes]
        for ax in sorted(axes, reverse=True):
            x = mx.squeeze(x, axis=ax)
        return [x]
    return [mx.squeeze(x)]

@register("Unsqueeze")
def unsqueeze(inputs, attrs):
    x = inputs[0]
    if len(inputs) > 1 and inputs[1] is not None:
        axes = sorted(inputs[1].tolist())
    else:
        axes = sorted(attrs.get("axes", []))
    for ax in axes:
        x = mx.expand_dims(x, axis=ax)
    return [x]

@register("Concat")
def concat(inputs, attrs):
    axis = attrs.get("axis", 0)
    valid = [inp for inp in inputs if inp is not None]
    return [mx.concatenate(valid, axis=axis)]

@register("Split")
def split(inputs, attrs):
    x = inputs[0]
    axis = attrs.get("axis", 0)
    num_outputs = attrs.get("num_outputs", None)
    if len(inputs) > 1 and inputs[1] is not None:
        split_sizes = inputs[1].tolist()
        indices = []
        acc = 0
        for s in split_sizes[:-1]:
            acc += s
            indices.append(acc)
        return list(mx.split(x, indices, axis=axis))
    elif num_outputs is not None:
        return list(mx.split(x, num_outputs, axis=axis))
    return [x]

@register("Slice")
def slice_(inputs, attrs):
    data = inputs[0]
    starts = [int(x) for x in inputs[1].tolist()]
    ends = [int(x) for x in inputs[2].tolist()]
    axes = [int(x) for x in inputs[3].tolist()] if len(inputs) > 3 and inputs[3] is not None else list(range(len(starts)))
    steps = [int(x) for x in inputs[4].tolist()] if len(inputs) > 4 and inputs[4] is not None else [1] * len(starts)
    slices = [slice(None)] * data.ndim
    for ax, start, end, step in zip(axes, starts, ends, steps):
        s = start if abs(start) < _INT64_MAX else None
        e = end if abs(end) < _INT64_MAX else None
        slices[ax] = slice(s, e, step)
    return [data[tuple(slices)]]

@register("Gather")
def gather(inputs, attrs):
    axis = attrs.get("axis", 0)
    return [mx.take(inputs[0], inputs[1], axis=axis)]

@register("GatherElements")
def gather_elements(inputs, attrs):
    axis = attrs.get("axis", 0)
    return [mx.take_along_axis(inputs[0], inputs[1], axis=axis)]

@register("Expand")
def expand(inputs, attrs):
    data = inputs[0]
    target = [int(s) for s in inputs[1].tolist()]
    data_shape = list(data.shape)
    # Pad shapes to same rank (prepend 1s to shorter one)
    while len(data_shape) < len(target):
        data_shape.insert(0, 1)
    while len(target) < len(data_shape):
        target.insert(0, 1)
    out_shape = [max(d, t) for d, t in zip(data_shape, target)]
    return [mx.broadcast_to(data, out_shape)]

@register("Pad")
def pad(inputs, attrs):
    data = inputs[0]
    pads_tensor = inputs[1]
    constant_value = inputs[2] if len(inputs) > 2 and inputs[2] is not None else mx.array(0.0)
    axes = inputs[3].tolist() if len(inputs) > 3 and inputs[3] is not None else None
    mode = attrs.get("mode", "constant")
    pads_list = [int(p) for p in pads_tensor.tolist()]
    n = len(pads_list) // 2
    if axes is not None:
        pad_widths = [(0, 0)] * data.ndim
        for i, ax in enumerate(axes):
            ax = int(ax)
            if ax < 0:
                ax += data.ndim
            pad_widths[ax] = (pads_list[i], pads_list[i + n])
    elif n < data.ndim:
        pad_widths = [(0, 0)] * (data.ndim - n) + [(pads_list[i], pads_list[i + n]) for i in range(n)]
    else:
        pad_widths = [(pads_list[i], pads_list[i + n]) for i in range(n)]
    if mode == "constant":
        return [mx.pad(data, pad_widths, constant_values=constant_value)]
    return [mx.pad(data, pad_widths)]

@register("Tile")
def tile(inputs, attrs):
    return [mx.tile(inputs[0], inputs[1].tolist())]

@register("Shape")
def shape(inputs, attrs):
    s = list(inputs[0].shape)
    start = int(attrs.get("start", 0))
    end = int(attrs.get("end", len(s)))
    return [mx.array(s[start:end], dtype=mx.int64)]

@register("ConstantOfShape")
def constant_of_shape(inputs, attrs):
    shape = inputs[0].tolist()
    value = attrs.get("value", None)
    if value is not None:
        from onnx import numpy_helper
        val = numpy_helper.to_array(value).item()
    else:
        val = 0.0
    return [mx.full(shape, val)]

_INT64_MAX = 2**63 - 1
_INT64_MIN = -(2**63 - 1)