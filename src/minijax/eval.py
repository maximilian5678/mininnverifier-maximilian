# Copyright (c) 2025 by David Boetius
# Licensed under the MIT License.
import numpy as np
import scipy.special as special

from . import core


class Array(core.Value):
    def __init__(self, array_like):
        self.array = np.asarray(array_like, dtype=np.float64)
        super().__init__(EvalInterpreter(), self.array.shape)

    def item(self):
        return self.array.item()

    def __repr__(self):
        data_str = str(self.array).replace("\n", "\n" + " " * len("Array("))
        return f"Array({data_str})"


def full(shape, fill_value):
    return Array(np.full(shape, fill_value, dtype=np.float64))


def zeros(shape):
    return full(shape, 0.0)


def ones(shape):
    return full(shape, 1.0)


class EvalInterpreter(core.Interpreter[Array]):
    def __init__(self):
        super().__init__(0)

    def wrap(self, value):
        if not isinstance(value, core.Value):
            return Array(value)
        elif not isinstance(value, Array):
            raise ValueError("EvalInterpreter must be the bottom interpreter")
        return value

    def process(self, primitive, values: list[Array], options: dict):
        np_vals = [v.array for v in values]
        np_out = eval_rules[primitive](*np_vals, **options)
        return Array(np_out)


def np_dot(x, y):  # np.dot doesn't broadcast
    if y.ndim <= 1:
        return np.dot(x, y)
    return np.einsum("...j,...jk", x, y)

def np_pad(x, config, axes, value):
    # Left, right, and middle padding
    l, r, m = config
    ndim = x.ndim 
    axes = [a % ndim for a in axes]

    # If interior padding is needed
    if m > 0:
        for ax in axes:
            new_shape = list(x.shape)
            new_shape[ax] = x.shape[ax] + (x.shape[ax] - 1) * m
            result = np.full(new_shape, value, dtype=x.dtype)
            # Fill only every (m+1)-th index
            idx = [slice(None)] * ndim
            idx[ax] = slice(None, None, m + 1)
            result[tuple(idx)] = x
            x = result
        # Left Right padding
    pad_width = [(0, 0)] * ndim
    for ax in axes:
        pad_width[ax] = (l, r)
    return np.pad(x, pad_width, constant_values=value)


eval_rules = {
    core.expand_dims: lambda x, axes: np.expand_dims(x, axes),
    core.moveaxis: np.moveaxis,
    core.reshape: lambda x, new_shape: np.reshape(x, new_shape),
    core.neg: lambda x: -x,
    core.add: lambda x, y: x + y,
    core.reduce_sum: lambda x, axes: x.sum(axes),
    core.dot: np_dot,
    core.mul: lambda x, y: x * y,
    core.reciprocal: lambda x: 1 / x,
    core.relu: lambda x: np.maximum(x, 0.0),
    core.square: np.square,
    core.sqrt: np.sqrt,
    core.exp: np.exp,
    core.log: np.log,
    core.where: np.where,
    # Activation Functions
    core.leaky_relu: lambda x, slope: np.where(x >= 0, x, slope * x),
    core.elu: lambda x: np.where(x >= 0, x, np.exp(x) - 1),
    core.gelu: lambda x: x * 0.5 * (1 + special.erf(x / np.sqrt(2))),
    core.normalcdf: lambda x: 0.5 * (1 + special.erf(x / np.sqrt(2))),
    core.ge: lambda x, y: x >= y,
    core.pad: lambda x, config, axes, value: np_pad(x, config, axes, value),
}
