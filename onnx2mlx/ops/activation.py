#
#   onnx2mlx
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import mlx.core as mx
import mlx.nn as nn
from . import register

@register("Relu")
def relu(inputs, attrs):
    return [nn.relu(inputs[0])]

@register("Sigmoid")
def sigmoid(inputs, attrs):
    return [mx.sigmoid(inputs[0])]

@register("Tanh")
def tanh(inputs, attrs):
    return [mx.tanh(inputs[0])]

@register("LeakyRelu")
def leaky_relu(inputs, attrs):
    alpha = attrs.get("alpha", 0.01)
    return [nn.leaky_relu(inputs[0], negative_slope=alpha)]

@register("Elu")
def elu(inputs, attrs):
    alpha = attrs.get("alpha", 1.0)
    return [nn.elu(inputs[0], alpha=alpha)]

@register("Selu")
def selu(inputs, attrs):
    return [nn.selu(inputs[0])]

@register("Gelu")
def gelu(inputs, attrs):
    approximate = attrs.get("approximate", "none")
    if approximate == "tanh":
        return [nn.gelu_approx(inputs[0])]
    return [nn.gelu(inputs[0])]

@register("Softmax")
def softmax(inputs, attrs):
    axis = attrs.get("axis", -1)
    return [mx.softmax(inputs[0], axis=axis)]

@register("LogSoftmax")
def log_softmax(inputs, attrs):
    axis = attrs.get("axis", -1)
    return [mx.log(mx.softmax(inputs[0], axis=axis))]

@register("HardSigmoid")
def hard_sigmoid(inputs, attrs):
    alpha = attrs.get("alpha", 0.2)
    beta = attrs.get("beta", 0.5)
    return [mx.clip(mx.add(mx.multiply(inputs[0], alpha), beta), 0.0, 1.0)]

@register("Hardswish")
def hardswish(inputs, attrs):
    return [nn.hardswish(inputs[0])]

@register("Clip")
def clip(inputs, attrs):
    x = inputs[0]
    a_min = inputs[1] if len(inputs) > 1 and inputs[1] is not None else None
    a_max = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
    if a_min is not None and a_max is not None:
        return [mx.clip(x, a_min, a_max)]
    elif a_min is not None:
        return [mx.maximum(x, a_min)]
    elif a_max is not None:
        return [mx.minimum(x, a_max)]
    return [x]

@register("Softplus")
def softplus(inputs, attrs):
    return [nn.softplus(inputs[0])]

@register("Softsign")
def softsign(inputs, attrs):
    return [nn.softsign(inputs[0])]

@register("Mish")
def mish(inputs, attrs):
    return [nn.mish(inputs[0])]

@register("PRelu")
def prelu(inputs, attrs):
    return [nn.prelu(inputs[0], inputs[1])]

@register("Celu")
def celu(inputs, attrs):
    alpha = attrs.get("alpha", 1.0)
    return [nn.celu(inputs[0], alpha=alpha)]