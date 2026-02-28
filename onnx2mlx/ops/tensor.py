#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from math import prod
import mlx.core as mx
from . import register
from .._utils import onnx_dtype_to_mlx, onnx_tensor_to_mlx

@register("Constant")
def constant(
    inputs: list[mx.array | None],
    attrs: dict[str, object]
) -> list[mx.array]:
    if "value" in attrs:
        return [onnx_tensor_to_mlx(attrs["value"])]
    if "value_float" in attrs:
        return [mx.array(attrs["value_float"])]
    if "value_int" in attrs:
        return [mx.array(attrs["value_int"], dtype=mx.int64)]
    if "value_floats" in attrs:
        return [mx.array(list(attrs["value_floats"]))]
    if "value_ints" in attrs:
        return [mx.array(list(attrs["value_ints"]), dtype=mx.int64)]
    raise ValueError("Constant node has no value attribute")

@register("Cast")
def cast(
    inputs: list[mx.array | None],
    attrs: dict[str, object]
) -> list[mx.array]:
    to = attrs.get("to")
    dtype = onnx_dtype_to_mlx(to)
    return [inputs[0].astype(dtype)]

@register("CastLike")
def cast_like(
    inputs: list[mx.array | None],
    attrs: dict[str, object]
) -> list[mx.array]:
    return [inputs[0].astype(inputs[1].dtype)]

@register("Identity")
def identity(
    inputs: list[mx.array | None],
    attrs: dict[str, object]
) -> list[mx.array]:
    return [inputs[0]]

@register("Dropout")
def dropout(
    inputs: list[mx.array | None],
    attrs: dict[str, object]
) -> list[mx.array]:
    outputs = [inputs[0]]
    if len(inputs) > 1:
        outputs.append(mx.ones_like(inputs[0]).astype(mx.bool_))
    return outputs

@register("Range")
def range_(
    inputs: list[mx.array | None],
    attrs: dict[str, object]
) -> list[mx.array]:
    start = inputs[0].item()
    limit = inputs[1].item()
    delta = inputs[2].item()
    return [mx.arange(start, limit, delta)]

@register("NonZero")
def non_zero(
    inputs: list[mx.array | None],
    attrs: dict[str, object]
) -> list[mx.array]:
    raise NotImplementedError("NonZero is not supported in MLX")

@register("ScatterND")
def scatter_nd(
    inputs: list[mx.array | None],
    attrs: dict[str, object]
) -> list[mx.array]:
    data = inputs[0]
    indices = inputs[1]
    updates = inputs[2]
    reduction = attrs.get("reduction", "none")
    index_depth = indices.shape[-1]
    indexed_shape = data.shape[:index_depth]
    slice_shape = data.shape[index_depth:]
    strides = [prod(data.shape[i + 1:index_depth]) for i in range(index_depth)]
    strides_arr = mx.array(strides, dtype=mx.int32)
    flat_indices = mx.sum(indices.astype(mx.int32) * strides_arr, axis=-1)
    flat_indices = mx.reshape(flat_indices, (-1,))
    num_slices = 1
    for s in indexed_shape:
        num_slices *= s
    slice_size = 1
    for s in slice_shape:
        slice_size *= s
    data_flat = mx.reshape(data, (num_slices, slice_size))
    updates_flat = mx.reshape(updates.astype(data.dtype), (-1, slice_size))
    match reduction:
        case "none":
            data_at_targets = mx.take(data_flat, flat_indices, axis=0)
            scatter_values = mx.subtract(updates_flat, data_at_targets)
        case "add":
            scatter_values = updates_flat
        case _:
            raise NotImplementedError(f"ScatterND reduction '{reduction}' is not supported")
    # One-hot scatter-add: indicator @ values -> aggregated per-slice updates
    arange = mx.arange(num_slices)
    indicator = mx.equal(arange[None, :], flat_indices[:, None]).astype(data.dtype)
    scattered = indicator.T @ scatter_values
    result_flat = mx.add(data_flat, scattered)
    return [mx.reshape(result_flat, data.shape)]
