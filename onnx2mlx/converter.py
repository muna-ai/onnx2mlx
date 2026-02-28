#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from onnx import ModelProto
from typing import Callable

import mlx.core as mx

from ._utils import get_attrs, onnx_tensor_to_mlx
from .ops import OP_REGISTRY

def onnx2mlx(model: ModelProto) -> Callable:
    """
    Convert an ONNX model to an MLX callable.

    The returned function accepts `mx.array` inputs in the same order as the
    ONNX graph's runtime inputs (graph inputs minus initializers). It returns
    a single `mx.array` when the graph has one output, or a list otherwise.

    The callable is compatible with `mx.compile` and `mx.export_function`
    because all control flow is data-independent.
    """
    # Gather graph I/O names
    graph = model.graph
    initializers: dict[str, mx.array] = {
        init.name: onnx_tensor_to_mlx(init)
        for init in graph.initializer
    }
    initializer_names = set(initializers.keys())
    input_names = [
        inp.name
        for inp in graph.input
        if inp.name not in initializer_names
    ]
    output_names = [out.name for out in graph.output]
    # Gather node specs
    nodes = list(graph.node)
    node_specs = []
    for node in nodes:
        handler = OP_REGISTRY.get(node.op_type)
        if handler is None:
            raise NotImplementedError(f"ONNX op '{node.op_type}' is not supported")
        attrs = get_attrs(node)
        node_specs.append((
            list(node.input),
            list(node.output),
            handler,
            attrs,
        ))
    def forward(*args: mx.array):
        values = dict(initializers)
        for name, arg in zip(input_names, args):
            values[name] = arg
        for inp_names, out_names, handler, attrs in node_specs:
            inputs = [values.get(n) if n != "" else None for n in inp_names]
            outputs = handler(inputs, attrs)
            for name, out in zip(out_names, outputs):
                values[name] = out
        results = [values[name] for name in output_names]
        return results[0] if len(results) == 1 else results
    return forward
