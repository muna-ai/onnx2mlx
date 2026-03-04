#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from collections import defaultdict, deque
import mlx.core as mx
from onnx import ModelProto, AttributeProto
from typing import Callable

from ._utils import get_attrs, onnx_tensor_to_mlx
from .ops import OP_REGISTRY

def onnx2mlx(model: ModelProto) -> Callable[..., mx.array | list[mx.array]]:
    """
    Convert an ONNX model to an MLX callable.

    The returned function accepts `mx.array` inputs in the same order as the
    ONNX graph's runtime inputs (graph inputs minus initializers). It returns
    a single `mx.array` when the graph has one output, or a list otherwise.

    The callable is compatible with `mx.compile` and `mx.export_function`
    because all control flow is data-independent.
    """
    initializers, input_names, output_names, node_specs = _compile_graph(model.graph)
    def forward(*args: mx.array):
        return _run_graph(initializers, input_names, output_names, node_specs, args)
    return forward

def _compile_graph(graph) -> tuple[dict[str, mx.array], list[str], list[str], list]:
    """
    Compile an ONNX GraphProto into initializers, I/O names, and node specs.
    """
    initializers = {
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
    available = initializer_names | {inp.name for inp in graph.input}
    sorted_nodes = _toposort(list(graph.node), available)
    node_specs = []
    for node in sorted_nodes:
        handler = OP_REGISTRY.get(node.op_type)
        if handler is None:
            raise NotImplementedError(f"ONNX op '{node.op_type}' is not supported")
        attrs = get_attrs(node)
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH:
                attrs[attr.name] = _compile_subgraph(attr.g)
        node_specs.append((
            list(node.input),
            list(node.output),
            handler,
            attrs,
        ))
    return initializers, input_names, output_names, node_specs

def _compile_subgraph(graph) -> Callable:
    """
    Compile a subgraph into a callable that takes a values dict.
    """
    initializers, input_names, output_names, node_specs = _compile_graph(graph)
    def run(parent_values: dict, *args):
        values = dict(parent_values)
        values.update(initializers)
        for name, arg in zip(input_names, args):
            values[name] = arg
        for inp_names, out_names, handler, attrs in node_specs:
            inputs = [values.get(n) if n != "" else None for n in inp_names]
            outputs = handler(inputs, attrs)
            for name, out in zip(out_names, outputs):
                values[name] = out
        return [values[name] for name in output_names]
    run._input_names = input_names
    run._output_names = output_names
    return run

def _run_graph(initializers, input_names, output_names, node_specs, args):
    """
    Execute a compiled graph.
    """
    values = dict(initializers)
    for name, arg in zip(input_names, args):
        values[name] = arg
    for inp_names, out_names, handler, attrs in node_specs:
        inputs = [values.get(n) if n != "" else None for n in inp_names]
        outputs = handler(inputs, {**attrs, "_scope": values})
        for name, out in zip(out_names, outputs):
            values[name] = out
    results = [values[name] for name in output_names]
    return results[0] if len(results) == 1 else results

def _toposort(nodes, available: set[str]) -> list:
    """
    Topologically sort nodes considering subgraph external references.
    """
    n = len(nodes)
    # Map output name -> node index
    producer = {}
    for i, node in enumerate(nodes):
        for out in node.output:
            if out:
                producer[out] = i
    # Build adjacency: node i depends on node j if j produces a name i needs
    in_degree = [0] * n
    dependents = defaultdict(list)
    for i, node in enumerate(nodes):
        deps = set()
        for inp in node.input:
            if inp and inp not in available and inp in producer:
                deps.add(producer[inp])
        for ext in _subgraph_external_refs(node):
            if ext not in available and ext in producer:
                deps.add(producer[ext])
        deps.discard(i)
        in_degree[i] = len(deps)
        for j in deps:
            dependents[j].append(i)
    queue = deque(i for i in range(n) if in_degree[i] == 0)
    order = []
    while queue:
        i = queue.popleft()
        order.append(i)
        for j in dependents[i]:
            in_degree[j] -= 1
            if in_degree[j] == 0:
                queue.append(j)
    if len(order) != n:
        remaining = [i for i in range(n) if in_degree[i] > 0]
        raise ValueError(f"Circular dependency among {len(remaining)} nodes")
    return [nodes[i] for i in order]

def _subgraph_external_refs(node) -> set[str]:
    """
    Collect names referenced inside subgraph attributes but not defined there.
    """
    refs = set()
    for attr in node.attribute:
        if attr.type != AttributeProto.GRAPH:
            continue
        g = attr.g
        defined = {i.name for i in g.initializer}
        defined |= {i.name for i in g.input}
        for n in g.node:
            defined.update(n.output)
        for n in g.node:
            for inp in n.input:
                if inp and inp not in defined:
                    refs.add(inp)
    return refs