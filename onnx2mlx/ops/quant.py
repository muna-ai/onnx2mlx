#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import mlx.core as mx
from . import register

@register("DynamicQuantizeLinear")
def dynamic_quantize_linear(inputs, attrs) -> list[mx.array]:
    x = inputs[0]
    x_min = mx.minimum(mx.array(0, dtype=x.dtype), mx.min(x))
    x_max = mx.maximum(mx.array(0, dtype=x.dtype), mx.max(x))
    scale = (x_max - x_min) / mx.array(255.0, dtype=x.dtype)
    scale = mx.maximum(scale, mx.array(1e-8, dtype=x.dtype))
    zero_point = mx.clip(mx.round(-x_min / scale), 0, 255).astype(mx.uint8)
    y = mx.clip(mx.round(x / scale) + zero_point.astype(x.dtype), 0, 255).astype(mx.uint8)
    return [y, scale, zero_point]

@register("MatMulInteger")
def matmul_integer(inputs, attrs) -> list[mx.array]:
    a = inputs[0]
    b = inputs[1]
    a_zp = inputs[2] if len(inputs) > 2 and inputs[2] is not None else mx.array(0, dtype=a.dtype)
    b_zp = inputs[3] if len(inputs) > 3 and inputs[3] is not None else mx.array(0, dtype=b.dtype)
    a_f = a.astype(mx.float32) - a_zp.astype(mx.float32)
    b_f = b.astype(mx.float32) - b_zp.astype(mx.float32)
    return [(a_f @ b_f).astype(mx.int32)]

@register("ConvInteger")
def conv_integer(inputs, attrs) -> list[mx.array]:
    x = inputs[0]
    w = inputs[1]
    x_zp = inputs[2] if len(inputs) > 2 and inputs[2] is not None else mx.array(0, dtype=x.dtype)
    w_zp = inputs[3] if len(inputs) > 3 and inputs[3] is not None else mx.array(0, dtype=w.dtype)
    x_f = x.astype(mx.float32) - x_zp.astype(mx.float32)
    w_f = w.astype(mx.float32) - w_zp.astype(mx.float32)
    # Delegate to the Conv handler via NCHW->NHWC conv
    from .conv import _conv_impl
    result = _conv_impl(x_f, w_f, None, attrs)
    return [result.astype(mx.int32)]

@register("DynamicQuantizeLSTM")
def dynamic_quantize_lstm(inputs, attrs) -> list[mx.array]:
    x = inputs[0]                # (seq_len, batch, input_size)
    w_quant = inputs[1]          # (num_directions, input_size, 4*hidden) uint8
    r_quant = inputs[2]          # (num_directions, hidden, 4*hidden) uint8
    b = inputs[3]                # (num_directions, 8*hidden)
    seq_lens = inputs[4]         # (batch,) or None
    init_h = inputs[5]           # (num_directions, batch, hidden)
    init_c = inputs[6]           # (num_directions, batch, hidden)
    # input[7] is empty (peepholes, unused)
    w_scale = inputs[8]          # (num_directions,)
    w_zp = inputs[9]             # (num_directions,) uint8
    r_scale = inputs[10]         # (num_directions,)
    r_zp = inputs[11]            # (num_directions,) uint8
    hidden_size = int(attrs.get("hidden_size"))
    direction = attrs.get("direction", "forward")
    num_directions = 2 if direction == "bidirectional" else 1
    seq_len = x.shape[0]
    batch = x.shape[1]
    all_h_states = []
    final_h = []
    final_c = []
    for d in range(num_directions):
        # Pre-compute integer weight matrices (zero-point subtracted)
        w_int = w_quant[d].astype(mx.float32) - w_zp[d].astype(mx.float32)
        r_int = r_quant[d].astype(mx.float32) - r_zp[d].astype(mx.float32)
        b_d = b[d] if b is not None else mx.zeros((8 * hidden_size,))
        bias = b_d[:4 * hidden_size] + b_d[4 * hidden_size:]
        h_t = init_h[d] if init_h is not None else mx.zeros((batch, hidden_size))
        c_t = init_c[d] if init_c is not None else mx.zeros((batch, hidden_size))
        h_seq = []
        indices = range(seq_len) if d == 0 else range(seq_len - 1, -1, -1)
        for t in indices:
            # Quantize x_t and h_t to uint8, then integer matmul + dequantize
            x_scale, x_int = _dynamic_quantize(x[t])
            h_scale, h_int = _dynamic_quantize(h_t)
            gates = (
                (x_int @ w_int) * (x_scale * w_scale[d])
                + (h_int @ r_int) * (h_scale * r_scale[d])
                + bias
            )
            i = mx.sigmoid(gates[:, :hidden_size])
            o = mx.sigmoid(gates[:, hidden_size:2 * hidden_size])
            f = mx.sigmoid(gates[:, 2 * hidden_size:3 * hidden_size])
            c_cand = mx.tanh(gates[:, 3 * hidden_size:])
            c_t = f * c_t + i * c_cand
            h_t = o * mx.tanh(c_t)
            h_seq.append(h_t)
        if d == 1:
            h_seq.reverse()
        all_h_states.append(mx.stack(h_seq))
        final_h.append(h_t)
        final_c.append(c_t)
    y = mx.stack(all_h_states, axis=1)
    y_h = mx.stack(final_h, axis=0)
    y_c = mx.stack(final_c, axis=0)
    return [y, y_h, y_c]

def _dynamic_quantize(x):
    """
    Quantize float tensor to uint8, returning (scale, zero_point, x_int).

    x_int contains the zero-point-subtracted integer values in float32,
    ready for matmul. This matches the DynamicQuantizeLinear + subtract
    pipeline that ORT uses internally.
    """
    x_min = mx.minimum(mx.array(0.0), mx.min(x))
    x_max = mx.maximum(mx.array(0.0), mx.max(x))
    scale = mx.maximum((x_max - x_min) / 255.0, mx.array(1e-8))
    zp = mx.clip(mx.round(-x_min / scale), 0, 255)
    x_q = mx.clip(mx.round(x / scale) + zp, 0, 255)
    return scale, x_q - zp