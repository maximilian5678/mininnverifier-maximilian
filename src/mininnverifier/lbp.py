# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
from dataclasses import dataclass

from minijax import core
from minijax.core import relu
from minijax.eval import Array, zeros
from minijax.grad import unbroadcast, vjp_rules

from .ibp import Box


def lbp_inner(w, x):  # multiply an LBP weight with an input
    return core.reduce_sum(w * x)


pos_part = relu


def neg_part(x):
    return -relu(-x)


@dataclass
class AffineBound:
    weights: tuple[core.Value, ...]
    bias: core.Value

    def concrete(self, *args: Box | core.Value):  # affine bound -> const bound
        res = self.bias
        for w, a in zip(self.weights, args, strict=True):
            a_lb, a_ub = (a.lb, a.ub) if isinstance(a, Box) else (a, a)
            res = res + lbp_inner(pos_part(w), a_lb) + lbp_inner(neg_part(w), a_ub)
        return res


def get_in_bounds(in_atoms, var_bounds):
    return [a.value if a.is_const else var_bounds[a] for a in in_atoms]


def linear_lower_bound(cg, var_bounds, params, rules):
    if len(cg.outvars) != 1:
        raise NotImplementedError("LBP only supports functions with a single return value.")
    if cg.outvars[0].shape not in ((), (1,)):
        raise NotImplementedError("LBP only supports functions with a scalar output.")

    weights = {cg.outvars[0]: Array(1.0)}
    bias = Array(0.0)

    def get_w(var):
        return weights.get(var, zeros(var.shape))

    for eqn in reversed(cg.equations):
        in_bounds = get_in_bounds(eqn.inputs, var_bounds)
        out_w = get_w(eqn.outvar)

        if eqn.primitive in rules:
            in_ws, in_b = rules[eqn.primitive](params[eqn.outvar], out_w, *in_bounds, **eqn.options)
            bias = bias + in_b
        elif eqn.primitive in linear_primitives:
            in_ws = transpose_weights(eqn.primitive, out_w, *in_bounds, **eqn.options)
        elif eqn.primitive in bilinear_primitives:
            in_ws = lbp_linear(eqn.primitive, out_w, *in_bounds, **eqn.options)
        else:
            raise NotImplementedError(f"No rule for primitive {eqn.primitive}")

        in_ws = (in_ws,) if not isinstance(in_ws, tuple) else in_ws
        for v, in_w in zip(eqn.inputs, in_ws, strict=True):
            in_w = unbroadcast(in_w, v.shape)
            if v.is_const or not isinstance(var_bounds[v], Box):
                # "Early concretization": our current bounds are linear in v,
                # but v has a fixed value ==> make it part of the bias
                # Example: c = a + 2.0. If we are here, we have a weight for the "2.0" constant.
                val = v.value if v.is_const else var_bounds[v]
                bias = bias + lbp_inner(in_w, val)
            else:
                weights[v] = get_w(v) + in_w

    return AffineBound(tuple(get_w(iv) for iv in cg.invars), bias)


def transpose_weights(primitive, out_w, *in_bounds, **options):
    xs = [ib.lb if isinstance(ib, Box) else ib for ib in in_bounds]
    # the relevant vjp transpose rules don't read the out argument => can pass None
    return vjp_rules[primitive](out_w, None, *xs, **options)


def lbp_linear(primitive, out_w, x, y, **options):
    if isinstance(x, Box) and isinstance(y, Box):
        raise NotImplementedError(f"No LBP rule for bilinear application of primitive {primitive}.")
    x_w, y_w = transpose_weights(primitive, out_w, x, y, **options)
    if not isinstance(y, Box):
        return x_w, zeros(y.shape)
    else:
        return zeros(x.shape), y_w


linear_primitives = (
    core.expand_dims,
    core.moveaxis,
    core.reshape,
    core.neg,
    core.add,
    core.reduce_sum,
)
bilinear_primitives = (core.dot, core.mul)
