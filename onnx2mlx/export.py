#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
import mlx.core as mx
from onnx import ModelProto
from pathlib import Path
from struct import pack, pack_into, unpack_from
from typing import Literal

from .convert import convert_to_mlx, ConvertContext

def export_to_mlx(
    model: ModelProto,
    path: str | Path,
    inputs: tuple[mx.array, ...],
    *,
    device: Literal["auto", "cpu", "gpu"]="auto",
    shapeless: bool=False,
    float64_mode: ConvertContext.Float64Mode="raise",
) -> None:
    """
    Convert an ONNX model to MLX and export it to a `.mlxfn` file.

    This combines `convert_to_mlx` conversion with `mx.export_function`, and
    optionally remaps all primitive streams to a specific device. This is
    useful when exporting on a CPU-only machine (e.g. Linux without Metal)
    but the `.mlxfn` will be loaded on a GPU-capable device.

    Parameters:
        path (str | Path): Output `.mlxfn` file path.
        model (ModelProto): ONNX model to convert and export.
        inputs (tuple): Sample `mx.array` inputs matching the model's runtime inputs.
        device (str): Target device for all primitives in the exported graph.
        shapeless (bool): Whether to export without baked-in shapes.
        float64_mode (str): How to handle float64 casts during conversion.
    """
    path = Path(path)
    # Convert and export
    mlx_fn = convert_to_mlx(model, float64_mode=float64_mode)
    def _export_fn(*args: mx.array):
        out = mlx_fn(*args)
        return [out] if isinstance(out, mx.array) else list(out)
    mx.export_function(str(path), _export_fn, *inputs, shapeless=shapeless)
    # Remap device streams
    if device != "auto":
        target = _DEVICE_TYPE_GPU if device == "gpu" else _DEVICE_TYPE_CPU
        source = _DEVICE_TYPE_CPU if device == "gpu" else _DEVICE_TYPE_GPU
        _remap_mlxfn_streams(path, source=source, target=target)

def _remap_mlxfn_streams(
    path: Path,
    *,
    source: int,
    target: int,
) -> int:
    """
    Rewrite primitive device streams in a serialized `.mlxfn` file.

    Each primitive's stream is serialized as 12 bytes right before its name:

        [int32 stream_index | int32 device_type | int32 device_index]

    where `device_type` 0 = CPU, 1 = GPU.

    We locate each primitive by searching for its serialized name pattern
    (uint64 length + ASCII bytes) then patch the `device_type` field
    8 bytes before the match.

    Parameters:
        path (Path): Path to the `.mlxfn` file to patch in-place.
        source (int): Device type value to replace (0 for CPU, 1 for GPU).
        target (int): Device type value to write.

    Returns:
        int: Number of streams patched.
    """
    data = bytearray(path.read_bytes())
    name_patterns = [
        pack("<Q", len(n)) + n.encode("ascii")
        for n in _MLX_PRIMITIVE_NAMES
    ]
    patched_offsets: set[int] = set()
    for pattern in name_patterns:
        pos = 0
        while True:
            idx = data.find(pattern, pos)
            if idx == -1:
                break
            dt_offset = idx - 8
            if dt_offset < 4:
                pos = idx + 1
                continue
            stream_index = unpack_from("<i", data, dt_offset - 4)[0]
            device_type = unpack_from("<i", data, dt_offset)[0]
            device_index = unpack_from("<i", data, dt_offset + 4)[0]
            if (
                device_type == source
                and device_index == 0
                and 0 <= stream_index < 256
                and dt_offset not in patched_offsets
            ):
                pack_into("<i", data, dt_offset, target)
                patched_offsets.add(dt_offset)
            pos = idx + len(pattern)
    path.write_bytes(data)
    return len(patched_offsets)

_MLX_PRIMITIVE_NAMES = [
    "Abs", "Add", "AddMM", "Arange", "ArcCos", "ArcCosh", "ArcSin",
    "ArcSinh", "ArcTan", "ArcTan2", "ArcTanh", "ArgPartition", "ArgReduce",
    "ArgSort", "AsType", "AsStrided", "BitwiseBinary", "BlockMaskedMM",
    "Broadcast", "BroadcastAxes", "Ceil", "Concatenate", "Conjugate",
    "Convolution", "Copy", "Cos", "Cosh", "Depends", "Divide", "DivMod",
    "DynamicSlice", "DynamicSliceUpdate", "Equal", "Erf", "ErfInv", "Exp",
    "Expm1", "ExpandDims", "FFT", "Flatten", "Floor", "Full", "Gather",
    "GatherAxis", "GatherMM", "Greater", "GreaterEqual", "Hadamard", "Imag",
    "Less", "LessEqual", "Log", "Log1p", "LogicalNot", "LogicalAnd",
    "LogicalOr", "LogAddExp", "LogSumExp", "MaskedScatter", "Matmul",
    "Maximum", "Minimum", "Multiply", "Negative", "NotEqual", "Reshape",
    "NumberOfElements", "Pad", "Partition", "Power", "QuantizedMatmul",
    "GatherQMM", "RandomBits", "Real", "Remainder", "Reduce", "Round",
    "Scan", "Scatter", "ScatterAxis", "Select", "Sigmoid", "Sign", "Sin",
    "Sinh", "Slice", "SliceUpdate", "Softmax", "Sort", "Split", "Square",
    "Squeeze", "Sqrt", "StopGradient", "Subtract", "Tan", "Tanh", "View",
    "Transpose", "Unflatten", "QRF", "SVD", "Inverse", "Cholesky", "Eig",
    "Eigh", "Quantize", "RMSNorm", "RMSNormVJP", "LayerNorm",
    "LayerNormVJP", "RoPE", "ScaledDotProductAttention", "CustomKernel",
]

_DEVICE_TYPE_CPU = 0
_DEVICE_TYPE_GPU = 1