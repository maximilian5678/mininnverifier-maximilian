# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
import math

from .core import avgpool, conv, exp, log, pad, reduce_sum, relu, reshape, square
from .eval import Array, zeros
from .nested_containers import flatten, map_structure
from .random import rand_uniform, split_rng_key


def linear(x, weight, bias):
    return x @ weight + bias


def mlp(x, params: list[dict[str, Array]]):
    x = reshape(x, (-1,))
    for layer_params in params[:-1]:
        x = linear(x, layer_params["weight"], layer_params["bias"])
        x = relu(x)
    return linear(x, params[-1]["weight"], params[-1]["bias"])


def softmax(x, axis: int):
    x_mean = reduce_sum(x, axis, keepaxes=True) / Array(x.shape[axis])
    exp_x = exp(x - x_mean)  # more numerically stable softmax
    return exp_x / reduce_sum(exp_x, axis, keepaxes=True)


# ======================================================================================================================


def reduce_mean(x):
    return reduce_sum(x) / Array(math.prod(x.shape))


def cross_entropy(y_pred, y_true):
    y_pred = softmax(y_pred, axis=-1)
    return -reduce_mean(reduce_sum(y_true * log(y_pred), axes=-1))


def weight_decay(params):
    param_norms = map_structure(lambda p: reduce_mean(square(p)), params)
    return sum(flatten(param_norms)[0], start=Array(0))


# ======================================================================================================================


def init_mlp(in_size, layer_sizes, rng_key):  # layer_sizes[-1] is output size
    in_sizes = (in_size,) + tuple(layer_sizes[:-1])
    rng_keys = split_rng_key(rng_key, len(layer_sizes))
    return [
        {"weight": kaiming_uniform((in_, out), in_, key), "bias": zeros((out,))}
        for in_, out, key in zip(in_sizes, layer_sizes, rng_keys)
    ]


def kaiming_uniform(shape, fan_in, rng_key):
    # kaiming_uniform initialization for ReLU
    bound = math.sqrt(2) * math.sqrt(3 / fan_in)
    return rand_uniform(shape, -bound, bound, rng_key=rng_key)


# ======================================================================================================================
# ConvNet — operates on full batch (N, 784), no vmap needed
# arch is a list of static dicts (not differentiable), params has only Arrays
# ======================================================================================================================


def conv_net(x, params, arch):
    """Forward pass of a ConvNet on a batch x of shape (N, in_size).

    params: list of {"kernel","bias"} (conv) or {"weight","bias"} (dense) dicts
    arch:   list of {"type":"conv","stride":int,"pad":int,"pool":int}
                  or {"type":"dense"} — parallel to params
    """
    n = x.shape[0]
    h = reshape(x, (n, 1, 28, 28))
    flattened = False
    for p, a in zip(params, arch):
        if a["type"] == "conv":
            if a["pad"] > 0:
                h = pad(h, config=(a["pad"], a["pad"], 0), axes=(2, 3), value=0.0)
            h = conv(h, p["kernel"], stride=a["stride"])
            b = reshape(p["bias"], (1, p["bias"].shape[0], 1, 1))
            h = h + b
            h = relu(h)
            if a["pool"] > 1:
                pl = a["pool"]
                h = avgpool(h, window_size=(1, 1, pl, pl), stride=(1, 1, pl, pl))
        else:
            if not flattened:
                h = reshape(h, (n, math.prod(h.shape[1:])))
                flattened = True
            h = linear(h, p["weight"], p["bias"])
            if a.get("relu", False):
                h = relu(h)
    return h


def init_conv_net(conv_specs, dense_sizes, num_classes, rng_key, in_hw=28):
    """Initialise a ConvNet and return (params, arch).

    conv_specs: list of [Cout, kH, kW, stride, pad, pool]
    dense_sizes: list of hidden dense sizes before the output layer
    num_classes: number of output classes

    params contains only differentiable Arrays (no ints).
    arch contains the static layout dicts — pass both to conv_net().
    """
    n_layers = len(conv_specs) + len(dense_sizes) + 1
    keys = split_rng_key(rng_key, n_layers)
    key_iter = iter(keys)

    params, arch = [], []
    cin, h, w = 1, in_hw, in_hw

    for (cout, kh, kw, stride, pad_amt, pool) in conv_specs:
        params.append({
            "kernel": kaiming_uniform((cout, cin, kh, kw), cin * kh * kw, next(key_iter)),
            "bias": zeros((cout,)),
        })
        arch.append({"type": "conv", "stride": stride, "pad": pad_amt, "pool": pool})
        h = (h + 2 * pad_amt - kh) // stride + 1
        w = (w + 2 * pad_amt - kw) // stride + 1
        if pool > 1:
            h = (h - pool) // pool + 1
            w = (w - pool) // pool + 1
        cin = cout

    flat = cin * h * w
    for out in list(dense_sizes) + [num_classes]:
        params.append({
            "weight": kaiming_uniform((flat, out), flat, next(key_iter)),
            "bias": zeros((out,)),
        })
        arch.append({"type": "dense", "relu": out != num_classes})
        flat = out

    return params, arch
