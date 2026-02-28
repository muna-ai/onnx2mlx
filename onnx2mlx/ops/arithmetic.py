#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import mlx.core as mx
from . import register

@register("Add")
def add(inputs, attrs):
    return [mx.add(inputs[0], inputs[1])]

@register("Sub")
def sub(inputs, attrs):
    return [mx.subtract(inputs[0], inputs[1])]

@register("Mul")
def mul(inputs, attrs):
    return [mx.multiply(inputs[0], inputs[1])]

@register("Div")
def div(inputs, attrs):
    return [mx.divide(inputs[0], inputs[1])]

@register("Pow")
def pow_(inputs, attrs):
    return [mx.power(inputs[0], inputs[1])]

@register("Sqrt")
def sqrt(inputs, attrs):
    return [mx.sqrt(inputs[0])]

@register("Exp")
def exp(inputs, attrs):
    return [mx.exp(inputs[0])]

@register("Log")
def log(inputs, attrs):
    return [mx.log(inputs[0])]

@register("Neg")
def neg(inputs, attrs):
    return [mx.negative(inputs[0])]

@register("Abs")
def abs_(inputs, attrs):
    return [mx.abs(inputs[0])]

@register("Reciprocal")
def reciprocal(inputs, attrs):
    return [mx.reciprocal(inputs[0])]

@register("Floor")
def floor(inputs, attrs):
    return [mx.floor(inputs[0])]

@register("Ceil")
def ceil(inputs, attrs):
    return [mx.ceil(inputs[0])]

@register("Sign")
def sign(inputs, attrs):
    return [mx.sign(inputs[0])]

@register("Mod")
def mod(inputs, attrs):
    return [mx.remainder(inputs[0], inputs[1])]

@register("Sum")
def sum_(inputs, attrs):
    result = inputs[0]
    for inp in inputs[1:]:
        if inp is not None:
            result = mx.add(result, inp)
    return [result]

@register("Min")
def min_(inputs, attrs):
    result = inputs[0]
    for inp in inputs[1:]:
        if inp is not None:
            result = mx.minimum(result, inp)
    return [result]

@register("Max")
def max_(inputs, attrs):
    result = inputs[0]
    for inp in inputs[1:]:
        if inp is not None:
            result = mx.maximum(result, inp)
    return [result]

@register("Erf")
def erf(inputs, attrs):
    return [mx.erf(inputs[0])]

@register("Sin")
def sin(inputs, attrs):
    return [mx.sin(inputs[0])]

@register("Cos")
def cos(inputs, attrs):
    return [mx.cos(inputs[0])]

@register("Tan")
def tan(inputs, attrs):
    return [mx.tan(inputs[0])]