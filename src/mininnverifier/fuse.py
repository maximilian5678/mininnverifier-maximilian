# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Fuse decomposed activation subgraphs back into single primitives.

The milestone 3 test networks do not use the fused ``elu``/``gelu`` primitives,
they spell the activations out:

    elu(t)  = where(relu(t), t, exp(t) - 1)
    gelu(t) = mul(t, normalcdf(t))

Per-primitive CROWN on those decompositions is hopeless: ``where`` has no useful
linear relaxation, and ``mul(t, normalcdf(t))`` needs McCormick envelopes over
two perfectly correlated operands, which throws away most of the tightness.
Rewriting the subgraph back into the fused primitive lets the dedicated
``crown_elu``/``crown_gelu`` rules do the work.
"""

import numpy as np

from minijax import core
from minijax.compute_graph import ComputeGraph, Equation
from minijax.eval import Array
from .crown_bounds import ibp_step


def _producers(cg):
    return {eqn.outvar: eqn for eqn in cg.equations}


def _const_close(atom, value):
    if not atom.is_const:
        return False
    arr = np.asarray(atom.value.array)
    return bool(arr.size > 0 and np.all(np.isclose(arr, value)))


def _unary_of(prod, atom, primitive, arg):
    """Check that ``atom`` is produced by ``primitive(arg)``."""
    eqn = prod.get(atom)
    return eqn is not None and eqn.primitive is primitive and eqn.inputs[0] is arg


def _is_positivity_test(prod, cond, t):
    """``cond`` is non-zero exactly where ``t > 0`` (or ``t >= 0``)."""
    eqn = prod.get(cond)
    if eqn is None:
        return False
    if eqn.primitive is core.relu:  # relu(t) != 0  <=>  t > 0
        return eqn.inputs[0] is t
    if eqn.primitive in (core.ge, core.greater_equal):
        a, b = eqn.inputs
        return a is t and _const_close(b, 0.0)
    return False


def _match_elu(eqn, prod):
    """where(relu(t), t, exp(t) - 1)  ->  elu(t)"""
    if eqn.primitive is not core.where:
        return None
    cond, t, false_val = eqn.inputs
    if t.is_const or t.shape != eqn.outvar.shape:
        return None
    if not _is_positivity_test(prod, cond, t):
        return None
    # the false branch is exp(t) - 1
    add_eqn = prod.get(false_val)
    if add_eqn is None or add_eqn.primitive is not core.add:
        return None
    a, b = add_eqn.inputs
    if _const_close(b, -1.0):
        exp_atom = a
    elif _const_close(a, -1.0):
        exp_atom = b
    else:
        return None
    if not _unary_of(prod, exp_atom, core.exp, t):
        return None
    return Equation(core.elu, (t,), eqn.outvar, {})


def _match_gelu(eqn, prod):
    """mul(t, normalcdf(t))  ->  gelu(t)"""
    if eqn.primitive is not core.mul:
        return None
    x, y = eqn.inputs
    for t, other in ((x, y), (y, x)):
        if t.is_const or t.shape != eqn.outvar.shape:
            continue
        if _unary_of(prod, other, core.normalcdf, t):
            return Equation(core.gelu, (t,), eqn.outvar, {})
    return None


MATCHERS = (_match_elu, _match_gelu)


def fuse_activations(cg: ComputeGraph) -> ComputeGraph:
    """Rewrite decomposed elu/gelu subgraphs into their fused primitives."""
    prod = _producers(cg)
    new_eqns, changed = [], False
    for eqn in cg.equations:
        fused = None
        for matcher in MATCHERS:
            fused = matcher(eqn, prod)
            if fused is not None:
                break
        new_eqns.append(fused if fused is not None else eqn)
        changed = changed or fused is not None
    if not changed:
        return cg
    return prune(ComputeGraph(cg.invars, cg.outvars, tuple(new_eqns)))


def prune(cg: ComputeGraph) -> ComputeGraph:
    """Drop equations whose result is no longer needed for the graph outputs."""
    needed = set(cg.outvars)
    keep = []
    for eqn in reversed(cg.equations):
        if eqn.outvar in needed:
            keep.append(eqn)
            needed.update(a for a in eqn.inputs if not a.is_const)
    return ComputeGraph(cg.invars, cg.outvars, tuple(reversed(keep)))

# ---------------------------------------------------------------------------
# Softmax
#
# Per-primitive interval propagation over exp -> reduce_sum -> reciprocal -> mul
# treats the numerator and the denominator as independent, which they are not:
# both are built from the same scores. Writing a softmax coordinate as
#
#     softmax_i = 1 / (1 + sum_{j != i} exp(g_j - g_i))
#
# makes it monotone in every g, so the exact box follows by evaluating each
# coordinate at the corner that extremises it. That box is the tightest sound
# one, and it feeds the McCormick envelopes downstream.
# ---------------------------------------------------------------------------

_EXP_CLIP = 700.0  # np.exp overflows just above this


def _exp(z):
    return np.exp(np.minimum(z, _EXP_CLIP))


def exact_softmax_box(g_lb, g_ub, axis):
    """Tightest interval enclosure of softmax(g) over the box [g_lb, g_ub]."""
    shift = np.max(g_ub, axis=axis, keepdims=True)
    e_up = _exp(g_ub - shift)
    e_lo = _exp(g_lb - shift)
    others_up = np.sum(e_up, axis=axis, keepdims=True) - e_up
    others_lo = np.sum(e_lo, axis=axis, keepdims=True) - e_lo
    with np.errstate(over="ignore", invalid="ignore"):
        lb = 1.0 / (1.0 + others_up * _exp(shift - g_lb))
        ub = 1.0 / (1.0 + others_lo * _exp(shift - g_ub))
    lb = np.nan_to_num(lb, nan=0.0, posinf=1.0, neginf=0.0)
    ub = np.nan_to_num(ub, nan=1.0, posinf=1.0, neginf=0.0)
    return np.clip(lb, 0.0, 1.0), np.clip(ub, 0.0, 1.0)


def find_softmax(cg):
    """Locate softmax subgraphs: mul(exp(g), reciprocal(sum(exp(g)))).

    Returns a map from the softmax output var to (scores var, axis).
    """
    prod = _producers(cg)
    found = {}
    for eqn in cg.equations:
        if eqn.primitive is not core.mul:
            continue
        for num, den in ((eqn.inputs[0], eqn.inputs[1]), (eqn.inputs[1], eqn.inputs[0])):
            exp_eqn = prod.get(num)
            if exp_eqn is None or exp_eqn.primitive is not core.exp:
                continue
            g = exp_eqn.inputs[0]
            rec_eqn = prod.get(den)
            if rec_eqn is None or rec_eqn.primitive is not core.reciprocal:
                continue
            atom = rec_eqn.inputs[0]
            expand = prod.get(atom)
            if expand is not None and expand.primitive is core.expand_dims:
                atom = expand.inputs[0]
            sum_eqn = prod.get(atom)
            if sum_eqn is None or sum_eqn.primitive is not core.reduce_sum:
                continue
            if sum_eqn.inputs[0] is not num or len(sum_eqn.options["axes"]) != 1:
                continue
            found[eqn.outvar] = (g, int(sum_eqn.options["axes"][0]))
            break
    return found


# ---------------------------------------------------------------------------
# Layer norm
#
# The normalised value c_i / sqrt(var) obeys a bound that does not depend on the
# input box at all: with var = alpha * sum_j c_j^2 (+ eps) and c_i^2 <= sum_j
# c_j^2, we get |c_i| / sqrt(var) <= 1 / sqrt(alpha), i.e. sqrt(H) for the usual
# mean-of-squares. Interval propagation cannot see this, because it treats the
# numerator and the variance as unrelated, and a variance box that reaches down
# to zero sends the reciprocal to infinity.
# ---------------------------------------------------------------------------


def _peel_scale(prod, atom):
    """Walk back through +const / *const / expand_dims, collecting the scale.

    Returns (reduce_sum equation, scale) or None.
    """
    scale = 1.0
    for _ in range(8):
        eqn = prod.get(atom)
        if eqn is None:
            return None
        if eqn.primitive is core.reduce_sum:
            return eqn, scale
        if eqn.primitive is core.expand_dims:
            atom = eqn.inputs[0]
        elif eqn.primitive is core.add:
            a, b = eqn.inputs
            const, other = (a, b) if a.is_const else (b, a)
            if not const.is_const:
                return None
            # only a non-negative offset (the epsilon) keeps the bound sound
            if np.min(np.asarray(const.value.array)) < 0.0:
                return None
            atom = other
        elif eqn.primitive is core.mul:
            a, b = eqn.inputs
            const, other = (a, b) if a.is_const else (b, a)
            if not const.is_const:
                return None
            value = np.asarray(const.value.array)
            if value.size != 1 or float(value.reshape(-1)[0]) <= 0.0:
                return None
            scale *= float(value.reshape(-1)[0])
            atom = other
        else:
            return None
    return None


def find_layer_norms(cg):
    """Locate mul(c, reciprocal(sqrt(alpha * sum(c^2) + eps))) subgraphs.

    Returns a map from the normalised output var to its magnitude bound.
    """
    prod = _producers(cg)
    found = {}
    for eqn in cg.equations:
        if eqn.primitive is not core.mul:
            continue
        for c, den in ((eqn.inputs[0], eqn.inputs[1]), (eqn.inputs[1], eqn.inputs[0])):
            rec = prod.get(den)
            if rec is None or rec.primitive is not core.reciprocal:
                continue
            root = prod.get(rec.inputs[0])
            if root is None or root.primitive is not core.sqrt:
                continue
            peeled = _peel_scale(prod, root.inputs[0])
            if peeled is None:
                continue
            sum_eqn, scale = peeled
            sq = prod.get(sum_eqn.inputs[0])
            if sq is None or sq.primitive is not core.square or sq.inputs[0] is not c:
                continue
            if scale <= 0.0:
                continue
            found[eqn.outvar] = 1.0 / np.sqrt(scale)
            break
    return found


def refine_softmax_bounds(cg, var_bounds, box_cls):
    """Re-propagate the interval bounds, clamping softmax and layer norm.

    The clamp has to happen *during* propagation: replacing a softmax output
    box afterwards leaves everything downstream computed from the unclamped
    value, which is where the blow-up actually lands.
    """

    softmaxes = find_softmax(cg)
    layer_norms = find_layer_norms(cg)
    if not softmaxes and not layer_norms:
        return var_bounds

    refined = {iv: var_bounds[iv] for iv in cg.invars if iv in var_bounds}
    for eqn in cg.equations:
        args = [a.value if a.is_const else refined[a] for a in eqn.inputs]
        box = ibp_step(eqn, args)
        entry = softmaxes.get(eqn.outvar)
        if entry is not None:
            g, axis = entry
            g_box = refined.get(g)
            if g_box is not None and hasattr(g_box, "lb"):
                lb, ub = exact_softmax_box(
                    np.asarray(g_box.lb.array), np.asarray(g_box.ub.array), axis
                )
                box = box_cls(
                    Array(np.maximum(np.asarray(box.lb.array), lb)),
                    Array(np.minimum(np.asarray(box.ub.array), ub)),
                )
        limit = layer_norms.get(eqn.outvar)
        if limit is not None:
            box = box_cls(
                Array(np.maximum(np.asarray(box.lb.array), -limit)),
                Array(np.minimum(np.asarray(box.ub.array), limit)),
            )
        refined[eqn.outvar] = box
    return refined