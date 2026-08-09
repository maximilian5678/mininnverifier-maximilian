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