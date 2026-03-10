#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import mlx.core as mx

from ..context import ConvertContext
from .._utils import onnx_pads_to_mlx
from . import register

@register("Conv")
def conv(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    x = inputs[0]
    w = inputs[1]
    b = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
    return [_conv_impl(x, w, b, attrs)]

@register("ConvTranspose")
def conv_transpose(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    x = inputs[0]
    w = inputs[1]
    b = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
    strides, dilations, group, pads, auto_pad, output_padding = _get_conv_params(attrs)
    padding = _compute_padding(auto_pad, pads, x.ndim - 2)
    ndim = w.ndim - 2
    w = _conv_transpose_weight(w, group, ndim)
    if ndim == 1:
        x = mx.transpose(x, (0, 2, 1))
        y = mx.conv_transpose1d(x, w, stride=strides[0], padding=padding, dilation=dilations[0], groups=group)
        y = _apply_output_padding(y, output_padding, ndim)
        if b is not None:
            y = mx.add(y, mx.reshape(b, (1, 1, -1)))
        y = mx.transpose(y, (0, 2, 1))
    elif ndim == 2:
        x = mx.transpose(x, (0, 2, 3, 1))
        stride = tuple(strides) if len(strides) > 1 else strides[0]
        dilation = tuple(dilations) if len(dilations) > 1 else dilations[0]
        y = mx.conv_transpose2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=group)
        y = _apply_output_padding(y, output_padding, ndim)
        if b is not None:
            y = mx.add(y, mx.reshape(b, (1, 1, 1, -1)))
        y = mx.transpose(y, (0, 3, 1, 2))
    elif ndim == 3:
        x = mx.transpose(x, (0, 2, 3, 4, 1))
        stride = tuple(strides)
        dilation = tuple(dilations)
        y = mx.conv_transpose3d(x, w, stride=stride, padding=padding, dilation=dilation, groups=group)
        y = _apply_output_padding(y, output_padding, ndim)
        if b is not None:
            y = mx.add(y, mx.reshape(b, (1, 1, 1, 1, -1)))
        y = mx.transpose(y, (0, 4, 1, 2, 3))
    else:
        raise NotImplementedError(f"ConvTranspose with {ndim}D spatial dims is not supported")
    return [y]

def _conv_impl(x, w, b, attrs):
    """
    Core convolution logic shared by Conv and ConvInteger.
    """
    strides, dilations, group, pads, auto_pad, _ = _get_conv_params(attrs)
    padding = _compute_padding(auto_pad, pads, x.ndim - 2)
    ndim = w.ndim - 2
    if ndim == 1:
        x = mx.transpose(x, (0, 2, 1))
        w = mx.transpose(w, (0, 2, 1))
        x, padding = _apply_asymmetric_padding(x, padding, ndim)
        y = mx.conv1d(x, w, stride=strides[0], padding=padding, dilation=dilations[0], groups=group)
        if b is not None:
            y = mx.add(y, mx.reshape(b, (1, 1, -1)))
        return mx.transpose(y, (0, 2, 1))
    elif ndim == 2:
        x = mx.transpose(x, (0, 2, 3, 1))
        w = mx.transpose(w, (0, 2, 3, 1))
        x, padding = _apply_asymmetric_padding(x, padding, ndim)
        stride = tuple(strides) if len(strides) > 1 else strides[0]
        dilation = tuple(dilations) if len(dilations) > 1 else dilations[0]
        y = mx.conv2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=group)
        if b is not None:
            y = mx.add(y, mx.reshape(b, (1, 1, 1, -1)))
        return mx.transpose(y, (0, 3, 1, 2))
    elif ndim == 3:
        x = mx.transpose(x, (0, 2, 3, 4, 1))
        w = mx.transpose(w, (0, 2, 3, 4, 1))
        x, padding = _apply_asymmetric_padding(x, padding, ndim)
        stride = tuple(strides)
        dilation = tuple(dilations)
        y = mx.conv3d(x, w, stride=stride, padding=padding, dilation=dilation, groups=group)
        if b is not None:
            y = mx.add(y, mx.reshape(b, (1, 1, 1, 1, -1)))
        return mx.transpose(y, (0, 4, 1, 2, 3))
    raise NotImplementedError(f"Conv with {ndim}D spatial dims is not supported")

def _conv_transpose_weight(w, group, ndim):
    """
    Reshape ONNX ConvTranspose weight to MLX layout.

    ONNX: (C_in, C_out/groups, K...) -> MLX: (C_out, K..., C_in/groups)
    """
    c_in = w.shape[0]
    c_out_g = w.shape[1]
    spatial = w.shape[2:]
    c_in_g = c_in // group
    w = mx.reshape(w, (group, c_in_g, c_out_g, *spatial))
    # Move c_in_g to last, merge group*c_out_g
    perm = [0, 2] + list(range(3, 3 + ndim)) + [1]
    w = mx.transpose(w, perm)
    return mx.reshape(w, (group * c_out_g, *spatial, c_in_g))

def _apply_output_padding(y, output_padding, ndim):
    """
    Pad the spatial dimensions of the output by output_padding.
    """
    if output_padding is None:
        return y
    # y is in NHWC layout; spatial dims are axes 1..ndim
    pad_widths = [(0, 0)]  # batch
    for i in range(ndim):
        op = output_padding[i] if i < len(output_padding) else 0
        pad_widths.append((0, op))
    pad_widths.append((0, 0))  # channels
    return mx.pad(y, pad_widths)

def _get_conv_params(attrs):
    strides = attrs.get("strides", [1])
    dilations = attrs.get("dilations", [1])
    group = attrs.get("group", 1)
    pads = attrs.get("pads", None)
    auto_pad = attrs.get("auto_pad", "NOTSET")
    output_padding = attrs.get("output_padding", None)
    return strides, dilations, group, pads, auto_pad, output_padding

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

def _apply_asymmetric_padding(x, padding, ndim):
    """
    Handle asymmetric padding by pre-padding the input with zeros.

    When padding is a list of (before, after) pairs with unequal values,
    mx.conv* can't handle it directly. We pad the input manually and
    return padding=0 for the conv call.
    Input x is in channels-last layout (N, spatial..., C).
    """
    if not isinstance(padding, list):
        return x, padding
    pad_widths = [(0, 0)] + padding + [(0, 0)]
    return mx.pad(x, pad_widths), 0