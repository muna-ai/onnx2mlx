#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from onnx import TensorProto, numpy_helper
from onnx.helper import get_attribute_value
import mlx.core as mx
import numpy as np

def onnx_dtype_to_mlx(onnx_dtype: int) -> mx.Dtype:
    match onnx_dtype:
        case TensorProto.BFLOAT16:  return mx.bfloat16
        case TensorProto.FLOAT16:   return mx.float16
        case TensorProto.FLOAT:     return mx.float32
        case TensorProto.DOUBLE:    return mx.float64
        case TensorProto.INT8:      return mx.int8
        case TensorProto.INT16:     return mx.int16
        case TensorProto.INT32:     return mx.int32
        case TensorProto.INT64:     return mx.int64
        case TensorProto.UINT8:     return mx.uint8
        case TensorProto.UINT16:    return mx.uint16
        case TensorProto.UINT32:    return mx.uint32
        case TensorProto.UINT64:    return mx.uint64
        case TensorProto.BOOL:      return mx.bool_
        case _: raise ValueError(f"Unsupported ONNX data type: {onnx_dtype}")

def get_attrs(node) -> dict[str, object]:
    def _decode(v):
        return v.decode("utf-8") if isinstance(v, bytes) else v
    return { attr.name: _decode(get_attribute_value(attr)) for attr in node.attribute }

def onnx_pads_to_mlx(pads: list[int]) -> list[tuple[int, int]]:
    n = len(pads) // 2
    return [(pads[i], pads[i + n]) for i in range(n)]

def onnx_tensor_to_mlx(tensor: TensorProto) -> mx.array:
    arr = numpy_helper.to_array(tensor)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    return mx.array(arr)