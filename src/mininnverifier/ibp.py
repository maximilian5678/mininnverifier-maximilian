# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
from dataclasses import dataclass

import numpy as np
import scipy.special as special

from minijax import core
from minijax.core import Value, abs, where, relu
from minijax.nested_containers import map_structure
from minijax.eval import Array, zeros


@dataclass
class Box:
    lb: core.Value
    ub: core.Value


def box_or_val(obj):
    return isinstance(obj, (Box, core.Value))


def ibp(fn):
    def ibp_fn(*args: Box | core.Value, **kwargs):
        with core.new_interpreter(IBPInterpreter()) as interpreter:
            vals = map_structure(interpreter.wrap, args, is_leaf=box_or_val)
            out_bounds = fn(*vals, **kwargs)

        return map_structure(lambda ibp_val: Box(ibp_val.lb, ibp_val.ub), out_bounds)

    return ibp_fn

class IBPValue(core.Value):
    def __init__(self, interpreter, lb, ub, is_point=False):
        super().__init__(interpreter, lb.shape)
        self.lb = lb  # lower bound
        self.ub = ub  # upper bound
        self.is_point = is_point  # whether lb == ub


class IBPInterpreter(core.Interpreter[IBPValue]):
    def wrap(self, value):
        if isinstance(value, IBPValue):
            return value
        elif isinstance(value, Box):
            return IBPValue(self, value.lb, value.ub)
        if not isinstance(value, core.Value):
            value = Array(value)
        return IBPValue(self, value, value, is_point=True)

    def process(self, primitive, values, options):
        if all(v.is_point for v in values):
            res = primitive(*[v.lb for v in values], **options)
            return IBPValue(self, res, res, is_point=True)

        if primitive in custom_primitives:
            out_lb, out_ub = custom_primitives[primitive](*values, **options)
        elif primitive in mono_non_dec_primitives:
            out_lb, out_ub = ibp_monotonic_non_decreasing(primitive, *values, **options)
        elif primitive in mono_non_inc_primitives:
            out_lb, out_ub = ibp_monotonic_non_increasing(primitive, *values, **options)
        elif primitive in linear_primitives:
            out_lb, out_ub = ibp_linear(primitive, *values, **options)
        elif primitive is core.square:
            out_lb, out_ub = ibp_square(*values, **options)
        else:
            raise NotImplementedError(f"No IBP rule for primitive {primitive}")
        return IBPValue(self, out_lb, out_ub)


def ibp_monotonic_non_decreasing(fn, *args, **options):
    in_lbs, in_ubs = [box.lb for box in args], [box.ub for box in args]
    out_lb = fn(*in_lbs, **options)
    out_ub = fn(*in_ubs, **options)
    return out_lb, out_ub


def ibp_monotonic_non_increasing(fn, *args, **options):
    out_ub, out_lb = ibp_monotonic_non_decreasing(fn, *args, **options)
    return out_lb, out_ub


def ibp_linear(fn, x, y, **options):
    if not x.is_point and not y.is_point:
        if fn is core.mul:
            return ibp_mul_box_box(x, y)
        if fn is core.dot:
            return ibp_dot_box_box(x, y)
        raise NotImplementedError(f"No IBP rule for bilinear application of primitive {fn}")
    elif x.is_point:
        x = x.lb
        y_mid = (y.ub + y.lb) * 0.5
        y_ran = (y.ub - y.lb) * 0.5
        out_mid = fn(x, y_mid, **options)
        out_ran = fn(abs(x), y_ran, **options)
        return out_mid - out_ran, out_mid + out_ran
    elif y.is_point:
        return ibp_linear(lambda y, x: fn(x, y, **options), y, x)

def ibp_square(x):
    lb, ub = x.lb.array, x.ub.array
    sq_lb, sq_ub = lb * lb, ub * ub
    out_ub = np.maximum(sq_lb, sq_ub)
    straddles = (lb <= 0.0) & (ub >= 0.0)
    out_lb = np.where(straddles, 0.0, np.minimum(sq_lb, sq_ub))
    return Array(out_lb), Array(out_ub)


def ibp_where(cond, x, y):
    cl, cu = cond.lb.array, cond.ub.array
    xl, xu = x.lb.array, x.ub.array
    yl, yu = y.lb.array, y.ub.array

    sure_true  = (cl > 0.0) | (cu < 0.0)
    sure_false = (cl == 0.0) & (cu == 0.0)

    out_lb = np.where(sure_true, xl, np.where(sure_false, yl, np.minimum(xl, yl)))
    out_ub = np.where(sure_true, xu, np.where(sure_false, yu, np.maximum(xu, yu)))
    return Array(out_lb), Array(out_ub)

GELU_ARGMIN = -0.7517913647329811
GELU_MIN = -0.1699712074798982

def _gelu_np(x):
    return x * 0.5 * (1.0 + special.erf(x / np.sqrt(2.0)))

def ibp_gelu(x):
    lb, ub = x.lb.array, x.ub.array
    g_lb, g_ub = _gelu_np(lb), _gelu_np(ub)
    out_ub = np.maximum(g_lb, g_ub)
    contains_min = (lb <= GELU_ARGMIN) & (ub >= GELU_ARGMIN)
    out_lb = np.where(contains_min, GELU_MIN, np.minimum(g_lb, g_ub))
    return Array(out_lb), Array(out_ub)

def ibp_mul_box_box(x, y):
    xl, xu = x.lb.array, x.ub.array
    yl, yu = y.lb.array, y.ub.array
    p1, p2, p3, p4 = xl * yl, xl * yu, xu * yl, xu * yu
    out_lb = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
    out_ub = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
    return Array(out_lb), Array(out_ub)

def ibp_conv(x, k, **options):
    if not k.is_point:
        raise NotImplementedError("conv with non-constant kernel not supported")
    x_mid = (x.ub + x.lb) * Array(0.5)
    x_ran = (x.ub - x.lb) * Array(0.5)
    kernel = k.lb
    out_mid = core.conv(x_mid, kernel, **options)
    out_ran = core.conv(x_ran, abs(kernel), **options)
    return out_mid - out_ran, out_mid + out_ran

def ibp_reciprocal(x):
    lb, ub = x.lb.array, x.ub.array
    straddles = (lb <= 0.0) & (ub >= 0.0)
    with np.errstate(divide="ignore"):
        r_lb, r_ub = 1.0 / ub, 1.0 / lb 
    out_lb = np.where(straddles, -np.inf, np.minimum(r_lb, r_ub))
    out_ub = np.where(straddles,  np.inf, np.maximum(r_lb, r_ub))
    return Array(out_lb), Array(out_ub)

# used in the transformer
def ibp_dot_box_box(x, y):
    xl, xu = x.lb.array, x.ub.array
    yl, yu = y.lb.array, y.ub.array
    if yl.ndim <= 1:
        c = [xl * yl, xl * yu, xu * yl, xu * yu]
        lo = np.minimum.reduce(c)
        hi = np.maximum.reduce(c)
        return Array(lo.sum(-1)), Array(hi.sum(-1))
    else:
        xl_e, xu_e = xl[..., :, None], xu[..., :, None]
        c = [xl_e * yl, xl_e * yu, xu_e * yl, xu_e * yu]
        lo = np.minimum.reduce(c)
        hi = np.maximum.reduce(c)
        return Array(lo.sum(-2)), Array(hi.sum(-2))

custom_primitives = {
    core.square: ibp_square,
    core.where: ibp_where,
    core.gelu: ibp_gelu,
    core.conv: ibp_conv,
    core.reciprocal: ibp_reciprocal,
}

def ibp_square(x):
    y_l, y_r = core.square(x.lb), core.square(x.ub)
    # x.lb >= 0 => monotonic increasing
    # x.ub <= 0 => monotonic decreasing
    # x.lb < 0 < x.ub => lb = 0.0, ub = max(x.lb^2 , x.ub^2)
    y_lb = where(x.lb >= 0.0, y_l, where(x.ub < 0.0, y_r, zeros(x.shape)))
    # x.ub > -x.lb => x.ub + x.lb > 0
    y_ub = where(x.lb >= 0.0, y_r, where(x.ub < 0.0, y_l, where(-x.lb >= x.ub, y_l, y_r)))
    return y_lb, y_ub


mono_non_dec_primitives = {
    core.expand_dims,
    core.moveaxis,
    core.reshape,
    core.concat,
    core.head,
    core.tail,
    core.add,
    core.reduce_sum,
    core.relu,
    core.exp,
    core.sqrt, 
    core.log,
    core.leaky_relu, 
    core.elu, 
    core.normalcdf, 
    core.avgpool, 
    core.sumpool,
    core.pad,
}

mono_non_inc_primitives = {core.neg}
linear_primitives = {core.dot, core.mul}
