#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import mlx.core as mx

from ..context import ConvertContext
from . import register

@register("Equal")
def equal(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.equal(inputs[0], inputs[1])]

@register("Greater")
def greater(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.greater(inputs[0], inputs[1])]

@register("Less")
def less(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.less(inputs[0], inputs[1])]

@register("GreaterOrEqual")
def greater_or_equal(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.greater_equal(inputs[0], inputs[1])]

@register("LessOrEqual")
def less_or_equal(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.less_equal(inputs[0], inputs[1])]

@register("Not")
def not_(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.logical_not(inputs[0])]

@register("And")
def and_(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.logical_and(inputs[0], inputs[1])]

@register("Or")
def or_(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.logical_or(inputs[0], inputs[1])]

@register("Where")
def where(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.where(inputs[0], inputs[1], inputs[2])]

@register("IsNaN")
def isnan(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    return [mx.isnan(inputs[0])]