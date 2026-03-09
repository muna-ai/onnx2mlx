#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import mlx.core as mx

from ..context import ConvertContext
from . import register

@register("If")
def if_(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    cond = inputs[0]
    scope = attrs.get("_scope", {})
    then_branch = attrs["then_branch"]
    else_branch = attrs["else_branch"]
    if cond.item():
        return then_branch(scope, ctx)
    return else_branch(scope, ctx)

@register("Loop")
def loop(
    inputs: list[mx.array | None],
    attrs: dict[str, object],
    ctx: ConvertContext,
) -> list[mx.array]:
    max_trip = inputs[0]
    cond = inputs[1]
    body = attrs["body"]
    scope = attrs.get("_scope", {})
    carried = list(inputs[2:])
    max_trip_val = max_trip.item() if max_trip is not None else float("inf")
    cond_val = cond.item() if cond is not None else True
    num_carried = len(carried)
    num_scan = len(body._output_names) - 1 - num_carried
    scan_outputs = [[] for _ in range(num_scan)]
    for i in range(int(max_trip_val)):
        if not cond_val:
            break
        iter_val = mx.array([i], dtype=mx.int64)
        cond_arr = mx.array(cond_val, dtype=mx.bool_)
        results = body(scope, ctx, iter_val, cond_arr, *carried)
        cond_val = results[0].item() if isinstance(results[0], mx.array) else bool(results[0])
        carried = list(results[1:1 + num_carried])
        for j in range(num_scan):
            scan_outputs[j].append(results[1 + num_carried + j])
    final_scan = []
    for scan in scan_outputs:
        if len(scan) > 0 and isinstance(scan[0], list):
            final_scan.append(scan[-1] if scan else [])
        elif len(scan) > 0:
            final_scan.append(mx.concatenate(scan, axis=0) if scan else mx.array([]))
        else:
            final_scan.append([])
    return list(carried) + final_scan