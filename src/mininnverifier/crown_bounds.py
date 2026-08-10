# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Intermediate bounds via CROWN instead of interval propagation.

IBP bounds degrade exponentially with depth: on a 6x50 ReLU network every single
neuron comes out unstable, which makes node-splitting branch and bound hopeless.
Back-substituting a linear bound for each neuron instead keeps the intermediate
boxes usable, so there is something left to branch on.

The backward pass is seeded per neuron because the LBP machinery carries one
weight per variable element, not a weight matrix. We keep the resulting affine
forms around: they are valid on the whole root box, so concretising them over a
sub-box gives tightened intermediate bounds for an input split for free, without
redoing any backward pass.
"""

import numpy as np

from minijax import core
from minijax.core import relu, elu, gelu, leaky_relu
from minijax.eval import Array

from .ibp import Box, IBPInterpreter
from .lbp import get_in_bounds, linear_lower_bound

ACTIVATIONS = (relu, elu, gelu, leaky_relu)


class AffineForms:
    """Per-element affine lower/upper bounds of one var, as a function of the input.

    ``w_lo``/``w_hi`` have shape (n_elements, n_input_elements).
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
    """Vars worth spending a backward pass on: the activation inputs."""
    targets = {}
    for i, eqn in enumerate(cg.equations):
        if eqn.primitive in ACTIVATIONS and not eqn.inputs[0].is_const:
            targets.setdefault(eqn.inputs[0], i)
    return targets


def ibp_step(eqn, arg_boxes):
    """One interval-propagation step for a single equation."""
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
    in_bounds = list(x_bounds)
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
        if form is None:  # primitive outside the fast path
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


# ---------------------------------------------------------------------------
# Batched backward pass
#
# The per-neuron loop above costs one Python-level backward pass per neuron,
# which is the single most expensive step in the verifier. Intermediate bounds
# need no gradients, so for the primitives that make up dense/ReLU networks we
# can run the whole layer in one shot in plain numpy: the weights simply carry a
# leading "which neuron" axis. Anything else falls back to the generic path.
# ---------------------------------------------------------------------------

BATCHED_PRIMITIVES = (
    core.dot, core.add, core.mul, core.neg, core.reshape, core.expand_dims,
    core.moveaxis, core.reduce_sum, core.relu,
)


def _unbcast(w, shape):
    """Reduce a batched weight down to (batch,) + shape."""
    extra = w.ndim - 1 - len(shape)
    if extra > 0:
        w = w.sum(axis=tuple(range(1, 1 + extra)))
    for i, s in enumerate(shape):
        if w.shape[1 + i] != s:
            w = w.sum(axis=1 + i, keepdims=True)
    return w


def _np_of(atom, var_bounds):
    if atom.is_const:
        return np.asarray(atom.value.array)
    box = var_bounds[atom]
    return None if isinstance(box, Box) else np.asarray(box.array)


def batched_affine_forms(cg, var_bounds, v, prefix):
    """Affine bounds for every element of ``v`` in a single backward sweep.

    Returns None if the prefix uses a primitive this fast path doesn't cover.
    """
    if any(eqn.primitive not in BATCHED_PRIMITIVES for eqn in prefix):
        return None
    shape = v.shape if v.shape else (1,)
    n = int(np.prod(shape))
    invar = cg.invars[0]

    # Two sweeps in one batch: the first n rows bound v from below, the next n
    # bound -v (i.e. v from above).
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
        # np.dot has a different contraction for every ndim combination, so key
        # on both operands rather than on the constant one alone.
        if yc is not None:
            if yc.ndim == 1:  # (..., m) @ (m,) -> (...)
                return (w[..., None] * yc, None), None
            if yc.ndim == 2:  # (..., n) @ (n, m) -> (..., m)
                return (w @ yc.T, None), None
            return None, None
        if xc is not None:
            yd = len(y.shape)
            if xc.ndim == 1 and yd == 1:  # (m,) @ (m,) -> ()
                return (None, w[..., None] * xc), None
            if xc.ndim == 1 and yd == 2:  # (m,) @ (m, k) -> (k,)
                return (None, xc[:, None] * w[:, None, :]), None
            if xc.ndim == 2 and yd == 1:  # (n, m) @ (m,) -> (n,)
                return (None, w @ xc), None
            if xc.ndim == 2 and yd == 2:  # (n, m) @ (m, k) -> (n, k)
                return (None, xc.T @ w), None
            return None, None
        return None, None
    if p is core.relu:
        box = var_bounds[eqn.inputs[0]]
        lb, ub = np.asarray(box.lb.array), np.asarray(box.ub.array)
        unstable = (lb < 0.0) & (ub > 0.0)
        denom = np.where(unstable, ub - lb, 1.0)
        u_slope = np.where(unstable, ub / denom, np.where(lb >= 0.0, 1.0, 0.0))
        u_off = np.where(unstable, -ub * lb / denom, 0.0)
        # match crown_relu's adaptive choice exactly, ties included
        l_slope = np.where(unstable, np.where(-lb >= ub, 0.0, 1.0), u_slope)
        pos = w >= 0.0
        slope = np.where(pos, l_slope, u_slope)
        off = np.where(pos, 0.0, u_off)
        bias_add = (w * off).reshape(w.shape[0], -1).sum(-1)
        return (w * slope,), bias_add
    return None, None