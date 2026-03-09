#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
import mlx.core as mx
from typing import Callable, TYPE_CHECKING

from ..converter import ConvertContext

Handler = Callable[[
    list[mx.array | None],
    dict[str, object],
    ConvertContext
], list[mx.array]]

OP_REGISTRY: dict[str, Handler] = {}

def register(*op_types: str):
    """
    Register one or more ONNX op types to a handler function.
    """
    def decorator(fn: Handler) -> Handler:
        for op_type in op_types:
            OP_REGISTRY[op_type] = fn
        return fn
    return decorator

from . import (
    arithmetic,
    activation,
    comparison,
    linalg,
    conv,
    pool,
    normalization,
    shape,
    reduction,
    tensor,
    image,
    quant,
    sequence,
    control,
)