#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import mlx.core as mx

from ..context import ConvertContext
from . import register

@register("Add")
def add(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.add(inputs[0], inputs[1])]

@register("Sub")
def sub(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.subtract(inputs[0], inputs[1])]

@register("Mul")
def mul(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.multiply(inputs[0], inputs[1])]

@register("Div")
def div(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    a, b = inputs[0], inputs[1]
    if (
        mx.issubdtype(a.dtype, mx.integer) and
        mx.issubdtype(b.dtype, mx.integer)
    ):
        return [mx.floor_divide(a, b)]
    return [mx.divide(a, b)]

@register("Pow")
def pow_(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.power(inputs[0], inputs[1])]

@register("Sqrt")
def sqrt(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.sqrt(inputs[0])]

@register("Exp")
def exp(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.exp(inputs[0])]

@register("Log")
def log(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.log(inputs[0])]

@register("Neg")
def neg(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.negative(inputs[0])]

@register("Abs")
def abs_(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.abs(inputs[0])]

@register("Reciprocal")
def reciprocal(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.reciprocal(inputs[0])]

@register("Floor")
def floor(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.floor(inputs[0])]

@register("Ceil")
def ceil(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.ceil(inputs[0])]

@register("Sign")
def sign(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.sign(inputs[0])]

@register("Mod")
def mod(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.remainder(inputs[0], inputs[1])]

@register("Sum")
def sum_(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    result = inputs[0]
    for inp in inputs[1:]:
        if inp is not None:
            result = mx.add(result, inp)
    return [result]

@register("Min")
def min_(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    result = inputs[0]
    for inp in inputs[1:]:
        if inp is not None:
            result = mx.minimum(result, inp)
    return [result]

@register("Max")
def max_(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    result = inputs[0]
    for inp in inputs[1:]:
        if inp is not None:
            result = mx.maximum(result, inp)
    return [result]

@register("Erf")
def erf(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.erf(inputs[0])]

@register("Sin")
def sin(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.sin(inputs[0])]

@register("Cos")
def cos(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.cos(inputs[0])]

@register("Tan")
def tan(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.tan(inputs[0])]

@register("Atan")
def atan(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.arctan(inputs[0])]

@register("Round")
def round_(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.round(inputs[0])]