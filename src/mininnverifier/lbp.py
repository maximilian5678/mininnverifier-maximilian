# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
from dataclasses import dataclass

import numpy as np

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


def linear_lower_bound(cg, var_bounds, params, rules, seed=None, equations=None, record=None):
    """Back-substitute a linear lower bound to the graph inputs.

    ``seed`` starts the backward pass from an arbitrary var and weight instead of
    the graph output (used to bound intermediate neurons); ``equations`` limits
    the pass to a prefix of the graph; ``record``, if given, collects the
    outgoing weight of every equation, which the branch-and-bound split
    heuristic scores.
    """
    if seed is None:
        if len(cg.outvars) != 1:
            raise NotImplementedError("LBP only supports functions with a single return value.")
        if cg.outvars[0].shape not in ((), (1,)):
            raise NotImplementedError("LBP only supports functions with a scalar output.")
        seed = {cg.outvars[0]: Array(1.0)}

    weights = dict(seed)
    bias = Array(0.0)
    if equations is None:
        equations = cg.equations

    def get_w(var):
        return weights.get(var, zeros(var.shape))

    for eqn in reversed(equations):
        in_bounds = get_in_bounds(eqn.inputs, var_bounds)
        out_w = get_w(eqn.outvar)
        if record is not None:
            record[eqn.outvar] = out_w

        if eqn.primitive in rules:
            in_ws, in_b = rules[eqn.primitive](params.get(eqn.outvar), out_w, *in_bounds, **eqn.options)
            bias = bias + in_b
        elif eqn.primitive in linear_primitives:
            in_ws = transpose_weights(eqn.primitive, out_w, *in_bounds, **eqn.options)
        elif eqn.primitive in affine_primitives:
            in_ws, in_b = lbp_affine(eqn.primitive, out_w, *in_bounds, **eqn.options)
            bias = bias + in_b
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
    core.concat_two,
    core.head,
    core.tail,
)

# Affine in their first argument, constant in the remaining ones (conv kernel,
# pad value/config, pooling window).
affine_primitives = (core.pad, core.conv, core.avgpool, core.sumpool)

bilinear_primitives = (core.dot, core.mul)


def _concrete(a):
    """The value of an atom, ignoring its interval width (linear maps don't care)."""
    return a.lb if isinstance(a, Box) else a


_adjoint_cache = {}


def _adjoint_matrix(primitive, out_shape, in_shape, primals, options):
    """Materialise the transpose of a constant linear map as a matrix.

    The vjp rules for pad/conv/avgpool are written in numpy and read ``.array``,
    so they cannot be applied to the traced weights of the alpha optimisation.
    Their transpose only depends on the shapes and the (constant) extra
    arguments though, so we build it once by pushing basis vectors through the
    numpy rule and reuse it afterwards.
    """
    key = (
        primitive.name,
        tuple(out_shape),
        tuple(in_shape),
        repr(sorted(options.items())),
        tuple(id(p) for p in primals),
    )
    if key in _adjoint_cache:
        return _adjoint_cache[key][0]

    n_out, n_in = int(np.prod(out_shape)), int(np.prod(in_shape))
    args = [Array(np.asarray(p.array)) for p in primals]
    basis = np.zeros(out_shape)
    flat = basis.reshape(-1)
    matrix = np.zeros((n_out, n_in))
    for i in range(n_out):
        flat[i] = 1.0
        res = vjp_rules[primitive](Array(basis.copy()), None, *args, **options)
        res = res if isinstance(res, tuple) else (res,)
        matrix[i] = np.asarray(res[0].array).reshape(-1)
        flat[i] = 0.0
    _adjoint_cache[key] = (matrix, primals)
    return matrix


def lbp_affine(primitive, out_w, x, *rest, **options):
    """Transpose a primitive that is affine in ``x`` and constant in ``rest``.

    Splitting these out of ``linear_primitives`` also fixes conv: conv is
    bilinear in (x, kernel), so the vjp's kernel gradient must not be folded
    into the bias by the early-concretization branch.
    """
    if any(isinstance(a, Box) for a in rest):
        raise NotImplementedError(f"No LBP rule for {primitive} with a non-constant argument.")
    x_val = _concrete(x)
    in_shape = x_val.shape
    matrix = _adjoint_matrix(primitive, out_w.shape, in_shape, (x_val,) + rest, options)
    flat_w = core.reshape(out_w, new_shape=(int(np.prod(out_w.shape)),))
    in_w = core.reshape(flat_w @ Array(matrix), new_shape=in_shape)
    # constant offset of the affine map (a pad with a non-zero fill value)
    offset = primitive(zeros(in_shape), *rest, **options)
    in_bias = lbp_inner(out_w, offset)
    return (in_w,) + tuple(zeros(a.shape) for a in rest), in_bias