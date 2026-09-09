"""CROWN-based intermediate bound calculation for neural network verification.

Calculates linear relaxation bounds (CROWN) for intermediate activation 
nodes to replace weak interval bounds (IBP). Computes and caches affine 
forms across the root input box so sub-boxes can be concretized instantly 
during branch-and-bound without re-running backward passes.
"""

import numpy as np
import scipy.special as special

from minijax import core
from minijax.core import relu, elu, gelu, leaky_relu
from minijax.eval import Array

from .ibp import Box, IBPInterpreter
from .lbp import get_in_bounds, linear_lower_bound

ACTIVATIONS = (relu, elu, gelu, leaky_relu)


class AffineForms:
    """Per-element affine lower/upper bounds of one var, as a function of the input.
    """

    def __init__(self, shape, w_lo, b_lo, w_hi, b_hi):
        self.shape = shape
        self.w_lo, self.b_lo = w_lo, b_lo
        self.w_hi, self.b_hi = w_hi, b_hi

    def concretize(self, x_box):
        lo = np.asarray(x_box.lb.array).reshape(-1)
        hi = np.asarray(x_box.ub.array).reshape(-1)
        lb = np.maximum(self.w_lo, 0.0) @ lo + np.minimum(self.w_lo, 0.0) @ hi + self.b_lo
        ub = np.maximum(self.w_hi, 0.0) @ hi + np.minimum(self.w_hi, 0.0) @ lo + self.b_hi
        return lb.reshape(self.shape), ub.reshape(self.shape)


def _refine_targets(cg):
    targets = {}
    for i, eqn in enumerate(cg.equations):
        if eqn.primitive in ACTIVATIONS and not eqn.inputs[0].is_const:
            targets.setdefault(eqn.inputs[0], i)
    return targets


def ibp_step(eqn, arg_boxes):
    with core.new_interpreter(IBPInterpreter()) as interp:
        vals = [interp.wrap(a) for a in arg_boxes]
        out = eqn.primitive(*vals, **eqn.options)
    return Box(out.lb, out.ub)


def crown_var_bounds(cg, x_bounds, rules, max_neurons=4000):
    """Interval bounds for every var, refined by CROWN on the activation inputs.
    Returns the bounds and the affine forms used to tighten them.
    """
    targets = _refine_targets(cg)
    var_bounds = {iv: xb for iv, xb in zip(cg.invars, x_bounds)}
    forms = {}
    budget = max_neurons

    for i, eqn in enumerate(cg.equations):
        args = [a.value if a.is_const else var_bounds[a] for a in eqn.inputs]
        box = ibp_step(eqn, args)
        var_bounds[eqn.outvar] = box

        v = eqn.outvar
        if v not in targets:
            continue
        n = int(np.prod(v.shape)) if v.shape else 1
        if n > budget:
            continue
        budget -= n
        prefix = cg.equations[: i + 1]
        form = batched_affine_forms(cg, var_bounds, v, prefix)
        if form is None:
            form = _affine_forms(cg, var_bounds, rules, v, prefix)
        forms[v] = form
        lb, ub = form.concretize(x_bounds[0])
        var_bounds[v] = _intersect(box, lb, ub)
    return var_bounds, forms


def _intersect(box, lb, ub):
    return Box(
        Array(np.maximum(np.asarray(box.lb.array), lb)),
        Array(np.minimum(np.asarray(box.ub.array), ub)),
    )


def _affine_forms(cg, var_bounds, rules, v, prefix):
    shape = v.shape if v.shape else (1,)
    n = int(np.prod(shape))
    d = int(np.prod(cg.invars[0].shape)) if cg.invars[0].shape else 1
    w_lo, b_lo = np.zeros((n, d)), np.zeros(n)
    w_hi, b_hi = np.zeros((n, d)), np.zeros(n)
    onehot = np.zeros(shape)
    flat = onehot.reshape(-1)
    for i in range(n):
        flat[i] = 1.0
        low = linear_lower_bound(cg, var_bounds, {}, rules,
                                 seed={v: Array(onehot.copy())}, equations=prefix)
        flat[i] = -1.0
        high = linear_lower_bound(cg, var_bounds, {}, rules,
                                  seed={v: Array(onehot.copy())}, equations=prefix)
        flat[i] = 0.0
        w_lo[i] = np.asarray(low.weights[0].array).reshape(-1)
        b_lo[i] = np.asarray(low.bias.array).reshape(-1)[0]
        # the upper bound came out as a lower bound on -v
        w_hi[i] = -np.asarray(high.weights[0].array).reshape(-1)
        b_hi[i] = -np.asarray(high.bias.array).reshape(-1)[0]
    return AffineForms(v.shape, w_lo, b_lo, w_hi, b_hi)



# Batched backward pass
BATCHED_PRIMITIVES = (
    core.dot, core.add, core.mul, core.neg, core.reshape, core.expand_dims,
    core.moveaxis, core.reduce_sum, core.concat_two, core.head, core.tail,
    core.relu, core.leaky_relu, core.elu, core.gelu,
)

_SQRT2 = np.sqrt(2.0)
_GELU_ARGMIN = -0.7517913647329811
_GELU_MIN = -0.1699712074798982


def _elu_np(t):
    return np.where(t >= 0.0, t, np.exp(np.minimum(t, 0.0)) - 1.0)


def _elu_deriv_np(t):
    return np.where(t >= 0.0, 1.0, np.exp(np.minimum(t, 0.0)))


def _ncdf_np(t):
    return 0.5 * (1.0 + special.erf(t / _SQRT2))


def _gelu_np(t):
    return t * _ncdf_np(t)


def _gelu_deriv_np(t):
    return _ncdf_np(t) + t * np.exp(-0.5 * t * t) / np.sqrt(2.0 * np.pi)


def _act_planes(primitive, options, lb, ub):
    denom = ub - lb
    same = denom <= 0.0
    safe = np.where(same, 1.0, denom)
    pos, neg = lb >= 0.0, ub <= 0.0
    zero = np.zeros_like(lb)

    if primitive is relu:
        unstable = ~(pos | neg)
        u_slope = np.where(unstable, ub / safe, np.where(pos, 1.0, 0.0))
        u_off = np.where(unstable, -ub * lb / safe, 0.0)
        l_slope = np.where(unstable, np.where(-lb >= ub, 0.0, 1.0), u_slope)
        return l_slope, zero, u_slope, u_off

    if primitive is leaky_relu:
        s = options.get("slope", 0.01)
        u_slope = np.where(pos, 1.0, np.where(neg, s, (ub - s * lb) / safe))
        u_off = np.where(pos | neg, 0.0, -(1.0 - s) * ub * lb / safe)
        l_slope = np.where(pos, 1.0, np.where(neg, s, np.where(-lb >= ub, s, 1.0)))
        return l_slope, zero, u_slope, u_off

    if primitive is elu:
        f_lb, f_ub = _elu_np(lb), _elu_np(ub)
        u_slope = np.where(same, _elu_deriv_np(lb), (f_ub - f_lb) / safe)
        u_off = f_ub - u_slope * ub
        t = 0.5 * (lb + ub)
        l_slope = _elu_deriv_np(t)
        return l_slope, _elu_np(t) - l_slope * t, u_slope, u_off

    if primitive is gelu:
        g_lb, g_ub = _gelu_np(lb), _gelu_np(ub)
        secant = np.where(same, _gelu_deriv_np(lb), (g_ub - g_lb) / safe)
        secant_off = g_lb - secant * lb
        t = 0.5 * (lb + ub)
        tangent = _gelu_deriv_np(t)
        tangent_off = _gelu_np(t) - tangent * t
        convex = (lb >= -_SQRT2) & (ub <= _SQRT2)
        concave = (lb >= _SQRT2) | (ub <= -_SQRT2)
        mixed = ~(convex | concave)
        l_slope = np.where(convex, tangent, np.where(concave, secant, 0.0))
        l_off = np.where(convex, tangent_off, np.where(concave, secant_off, 0.0))
        u_slope = np.where(convex, secant, np.where(concave, tangent, 0.0))
        u_off = np.where(convex, secant_off, np.where(concave, tangent_off, 0.0))
        contains_min = (lb <= _GELU_ARGMIN) & (ub >= _GELU_ARGMIN)
        l_off = np.where(
            mixed, np.where(contains_min, _GELU_MIN, np.minimum(g_lb, g_ub)), l_off
        )
        u_off = np.where(mixed, np.maximum(g_lb, g_ub), u_off)
        return l_slope, l_off, u_slope, u_off

    return None


def _slice_at(ndim, axis, sl):
    idx = [slice(None)] * ndim
    idx[1 + axis] = sl
    return tuple(idx)


def _unbcast(w, shape):
    extra = w.ndim - 1 - len(shape)
    if extra > 0:
        w = w.sum(axis=tuple(range(1, 1 + extra)))
    for i, s in enumerate(shape):
        if w.shape[1 + i] != s:
            w = w.sum(axis=1 + i, keepdims=True)
    return w


def batched_affine_forms(cg, var_bounds, v, prefix):
    if any(eqn.primitive not in BATCHED_PRIMITIVES for eqn in prefix):
        return None
    shape = v.shape if v.shape else (1,)
    n = int(np.prod(shape))
    invar = cg.invars[0]

    seed = np.concatenate([np.eye(n), -np.eye(n)]).reshape((2 * n,) + shape)
    weights = {v: seed}
    bias = np.zeros(2 * n)

    for eqn in reversed(prefix):
        w = weights.pop(eqn.outvar, None)
        if w is None:
            continue
        in_ws, add_bias = _batched_rule(eqn, w, var_bounds)
        if in_ws is None:
            return None
        if add_bias is not None:
            bias = bias + add_bias
        for atom, in_w in zip(eqn.inputs, in_ws):
            if in_w is None:
                continue
            in_w = _unbcast(in_w, atom.shape)
            if atom.is_const:
                bias = bias + (in_w * np.asarray(atom.value.array)).reshape(2 * n, -1).sum(-1)
            else:
                prev = weights.get(atom)
                weights[atom] = in_w if prev is None else prev + in_w

    w_out = weights.get(invar)
    if w_out is None:
        w_out = np.zeros((2 * n,) + invar.shape)
    w_out = w_out.reshape(2 * n, -1)
    return AffineForms(v.shape, w_out[:n], bias[:n], -w_out[n:], -bias[n:])


def _batched_rule(eqn, w, var_bounds):
    p = eqn.primitive
    if p is core.neg:
        return (-w,), None
    if p is core.add:
        return tuple(w for _ in eqn.inputs), None
    if p is core.reshape:
        return (w.reshape((w.shape[0],) + eqn.inputs[0].shape),), None
    if p is core.expand_dims:
        axes = tuple(a % len(eqn.outvar.shape) for a in eqn.options["axes"])
        return (w.sum(axis=tuple(1 + a for a in axes)),), None
    if p is core.moveaxis:
        src, dst = eqn.options["source"], eqn.options["destination"]
        nd = len(eqn.outvar.shape)
        return (np.moveaxis(w, 1 + (dst % nd), 1 + (src % nd)),), None
    if p is core.reduce_sum:
        axes = tuple(a % len(eqn.inputs[0].shape) for a in eqn.options["axes"])
        wt = np.expand_dims(w, tuple(1 + a for a in axes))
        return (np.broadcast_to(wt, (w.shape[0],) + eqn.inputs[0].shape).copy(),), None
    if p is core.mul:
        x, y = eqn.inputs
        xc = np.asarray(x.value.array) if x.is_const else None
        yc = np.asarray(y.value.array) if y.is_const else None
        if yc is not None:
            return (w * yc, None), None
        if xc is not None:
            return (None, w * xc), None
        return None, None
    if p is core.dot:
        x, y = eqn.inputs
        xc = np.asarray(x.value.array) if x.is_const else None
        yc = np.asarray(y.value.array) if y.is_const else None

        if yc is not None:
            if yc.ndim == 1:
                return (w[..., None] * yc, None), None
            if yc.ndim == 2:
                return (w @ yc.T, None), None
            return None, None
        if xc is not None:
            yd = len(y.shape)
            if xc.ndim == 1 and yd == 1:
                return (None, w[..., None] * xc), None
            if xc.ndim == 1 and yd == 2:
                return (None, xc[:, None] * w[:, None, :]), None
            if xc.ndim == 2 and yd == 1:
                return (None, w @ xc), None
            if xc.ndim == 2 and yd == 2:
                return (None, xc.T @ w), None
            return None, None
        return None, None
    if p is core.head:
        ax = eqn.options["axis"] % len(eqn.inputs[0].shape)
        in_w = np.zeros((w.shape[0],) + eqn.inputs[0].shape)
        in_w[_slice_at(in_w.ndim, ax, slice(0, eqn.options["index"]))] = w
        return (in_w,), None
    if p is core.tail:
        ax = eqn.options["axis"] % len(eqn.inputs[0].shape)
        in_w = np.zeros((w.shape[0],) + eqn.inputs[0].shape)
        in_w[_slice_at(in_w.ndim, ax, slice(eqn.options["index"], None))] = w
        return (in_w,), None
    if p is core.concat_two:
        ax = eqn.options["axis"] % len(eqn.outvar.shape)
        n0 = eqn.inputs[0].shape[ax]
        return (
            w[_slice_at(w.ndim, ax, slice(0, n0))].copy(),
            w[_slice_at(w.ndim, ax, slice(n0, None))].copy(),
        ), None
    if p in (core.relu, core.leaky_relu, core.elu, core.gelu):
        box = var_bounds[eqn.inputs[0]]
        if not isinstance(box, Box):
            return None, None
        lb, ub = np.asarray(box.lb.array), np.asarray(box.ub.array)
        planes = _act_planes(p, eqn.options, lb, ub)
        if planes is None:
            return None, None
        l_slope, l_off, u_slope, u_off = planes
        pos = w >= 0.0
        slope = np.where(pos, l_slope, u_slope)
        off = np.where(pos, l_off, u_off)
        bias_add = (w * off).reshape(w.shape[0], -1).sum(-1)
        return (w * slope,), bias_add
    return None, None