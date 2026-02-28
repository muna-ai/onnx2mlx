#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import mlx.core as mx
from . import register

@register("MatMul")
def matmul(inputs, attrs):
    return [mx.matmul(inputs[0], inputs[1])]

@register("Gemm")
def gemm(inputs, attrs):
    A, B = inputs[0], inputs[1]
    C = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
    alpha = attrs.get("alpha", 1.0)
    beta = attrs.get("beta", 1.0)
    transA = attrs.get("transA", 0)
    transB = attrs.get("transB", 0)
    if transA:
        A = mx.transpose(A)
    if transB:
        B = mx.transpose(B)
    result = mx.matmul(A, B)
    if alpha != 1.0:
        result = mx.multiply(result, alpha)
    if C is not None:
        if beta != 1.0:
            C = mx.multiply(C, beta)
        result = mx.add(result, C)
    return [result]

@register("Einsum")
def einsum(inputs, attrs):
    equation = attrs.get("equation", "")
    return [mx.einsum(equation, *inputs)]