#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import mlx.core as mx
import mlx.nn as nn

from ..context import ConvertContext
from . import register

@register("Resize")
def resize(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    x = inputs[0]
    # inputs: X, roi, scales, [sizes]
    scales = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
    sizes = inputs[3] if len(inputs) > 3 and inputs[3] is not None else None
    mode = attrs.get("mode", "nearest")
    coord_transform = attrs.get("coordinate_transformation_mode", "half_pixel")
    align_corners = coord_transform == "align_corners"
    ndim = x.ndim - 2
    if sizes is not None:
        sizes_list = sizes.tolist()
        spatial_out = sizes_list[2:]
        spatial_in = list(x.shape[2:])
        scale_factors = tuple(spatial_out[i] / spatial_in[i] for i in range(ndim))
    elif scales is not None:
        scales_list = scales.tolist()
        scale_factors = tuple(scales_list[2:])
    else:
        return [x]
    mlx_mode = _onnx_resize_mode(mode)
    # ONNX: NCHW -> MLX Upsample expects NHWC
    if ndim == 1:
        x = mx.transpose(x, (0, 2, 1))
    elif ndim == 2:
        x = mx.transpose(x, (0, 2, 3, 1))
    elif ndim == 3:
        x = mx.transpose(x, (0, 2, 3, 4, 1))
    scale_factors = (
        scale_factors[0]
        if len(set(scale_factors)) == 1
        else scale_factors
    )
    up = nn.Upsample(
        scale_factor=scale_factors,
        mode=mlx_mode,
        align_corners=align_corners
    )
    y = up(x)
    if ndim == 1:
        y = mx.transpose(y, (0, 2, 1))
    elif ndim == 2:
        y = mx.transpose(y, (0, 3, 1, 2))
    elif ndim == 3:
        y = mx.transpose(y, (0, 4, 1, 2, 3))
    return [y]

@register("GridSample")
def grid_sample(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    x = inputs[0]        # (N, C, H_in, W_in) -- NCHW
    grid = inputs[1]     # (N, H_out, W_out, 2)
    mode = attrs.get("mode", "bilinear")
    padding_mode = attrs.get("padding_mode", "zeros")
    align_corners = bool(attrs.get("align_corners", 0))
    if mode == "bilinear" and padding_mode == "zeros" and not align_corners:
        # NCHW -> NHWC
        x_nhwc = mx.transpose(x, (0, 2, 3, 1))
        y_nhwc = _grid_sample_metal(x_nhwc, grid)
        # NHWC -> NCHW
        return [mx.transpose(y_nhwc, (0, 3, 1, 2))]
    raise NotImplementedError(
        f"GridSample with mode='{mode}', padding_mode='{padding_mode}', "
        f"align_corners={align_corners} is not supported by the Metal kernel"
    )

def _onnx_resize_mode(mode: str) -> str:
    match mode:
        case "nearest":     return "nearest"
        case "linear":      return "linear"
        case "bilinear":    return "linear"
        case "cubic":       return "cubic"
        case "bicubic":     return "cubic"
        case _:             raise NotImplementedError(f"Resize mode '{mode}' is not supported")

def _grid_sample_metal(x: mx.array, grid: mx.array):
    """
    Bilinear grid sample with zeros padding and align_corners=False.

    Parameters:
        x (mx.array): Input tensor in NHWC layout `(B, H, W, C)`.
        grid (mx.array): Sampling grid `(B, gH, gW, 2)` with coords in `[-1, 1]`.
    """
    B, _, _, C = x.shape
    _, gN, gM, _ = grid.shape
    out_shape = (B, gN, gM, C)
    kernel = mx.fast.metal_kernel(
        name="grid_sample",
        input_names=["x", "grid"],
        output_names=["out"],
        source=_GRID_SAMPLE_SOURCE,
    )
    num_elements = 1
    for s in out_shape:
        num_elements *= s
    outputs = kernel(
        inputs=[x, grid],
        output_shapes=[out_shape],
        output_dtypes=[x.dtype],
        grid=(num_elements, 1, 1),
        threadgroup=(256, 1, 1),
        template=[("T", x.dtype)],
    )
    return outputs[0]

_GRID_SAMPLE_SOURCE = """
uint elem = thread_position_in_grid.x;
int H = x_shape[1];
int W = x_shape[2];
int C = x_shape[3];
int w_stride = C;
int h_stride = W * w_stride;
int b_stride = H * h_stride;
int gH = grid_shape[1];
int gW = grid_shape[2];
uint grid_idx = elem / C * 2;
float ix = ((grid[grid_idx] + 1) * W - 1) / 2;
float iy = ((grid[grid_idx + 1] + 1) * H - 1) / 2;
int ix_nw = floor(ix);
int iy_nw = floor(iy);
int ix_ne = ix_nw + 1;
int iy_ne = iy_nw;
int ix_sw = ix_nw;
int iy_sw = iy_nw + 1;
int ix_se = ix_nw + 1;
int iy_se = iy_nw + 1;
T nw = (ix_se - ix)    * (iy_se - iy);
T ne = (ix    - ix_sw) * (iy_sw - iy);
T sw = (ix_ne - ix)    * (iy    - iy_ne);
T se = (ix    - ix_nw) * (iy    - iy_nw);
int batch_idx = elem / C / gH / gW * b_stride;
int channel_idx = elem % C;
int base_idx = batch_idx + channel_idx;
T I_nw = x[base_idx + iy_nw * h_stride + ix_nw * w_stride];
T I_ne = x[base_idx + iy_ne * h_stride + ix_ne * w_stride];
T I_sw = x[base_idx + iy_sw * h_stride + ix_sw * w_stride];
T I_se = x[base_idx + iy_se * h_stride + ix_se * w_stride];
I_nw = iy_nw >= 0 && iy_nw <= H - 1 && ix_nw >= 0 && ix_nw <= W - 1 ? I_nw : 0;
I_ne = iy_ne >= 0 && iy_ne <= H - 1 && ix_ne >= 0 && ix_ne <= W - 1 ? I_ne : 0;
I_sw = iy_sw >= 0 && iy_sw <= H - 1 && ix_sw >= 0 && ix_sw <= W - 1 ? I_sw : 0;
I_se = iy_se >= 0 && iy_se <= H - 1 && ix_se >= 0 && ix_se <= W - 1 ? I_se : 0;
out[elem] = nw * I_nw + ne * I_ne + sw * I_sw + se * I_se;
"""