#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import mlx.core as mx
from . import register

@register("SequenceEmpty")
def sequence_empty(inputs, attrs):
    return [[]]

@register("SplitToSequence")
def split_to_sequence(inputs, attrs):
    x = inputs[0]
    split = inputs[1] if len(inputs) > 1 and inputs[1] is not None else None
    axis = int(attrs.get("axis", 0))
    keepdims = bool(attrs.get("keepdims", 1))
    if split is None:
        chunks = [mx.take(x, mx.array([i]), axis=axis) if keepdims
                  else mx.squeeze(mx.take(x, mx.array([i]), axis=axis), axis=axis)
                  for i in range(x.shape[axis])]
    else:
        split_sizes = split.tolist()
        if isinstance(split_sizes, (int, float)):
            split_sizes = [int(split_sizes)] * (x.shape[axis] // int(split_sizes))
        chunks = []
        start = 0
        for s in split_sizes:
            s = int(s)
            idx = mx.arange(start, start + s)
            chunk = mx.take(x, idx, axis=axis)
            if not keepdims and s == 1:
                chunk = mx.squeeze(chunk, axis=axis)
            chunks.append(chunk)
            start += s
    return [chunks]

@register("ConcatFromSequence")
def concat_from_sequence(inputs, attrs):
    seq = inputs[0]
    axis = int(attrs.get("axis", 0))
    new_axis = bool(attrs.get("new_axis", 0))
    if new_axis:
        return [mx.stack(seq, axis=axis)]
    return [mx.concatenate(seq, axis=axis)]

@register("SequenceAt")
def sequence_at(inputs, attrs):
    seq = inputs[0]
    idx = inputs[1].item()
    return [seq[int(idx)]]

@register("SequenceInsert")
def sequence_insert(inputs, attrs):
    seq = list(inputs[0])
    tensor = inputs[1]
    if len(inputs) > 2 and inputs[2] is not None:
        pos = inputs[2].item()
        seq.insert(int(pos), tensor)
    else:
        seq.append(tensor)
    return [seq]