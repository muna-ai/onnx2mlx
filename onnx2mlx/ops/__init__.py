#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from typing import Callable
import mlx.core as mx

Handler = Callable[[list[mx.array | None], dict[str, object]], list[mx.array]]

OP_REGISTRY: dict[str, Handler] = {}

def register(*op_types: str):
    """Register one or more ONNX op types to a handler function."""
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