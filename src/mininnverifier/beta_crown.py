# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.

import numpy as np

from minijax import core
from minijax.core import relu
from minijax.nested_containers import map_structure
from minijax.eval import zeros, Array
from minijax.compute_graph import make_graph
from minijax.grad import _grad_backwards
from minijax.nested_containers import flatten, unflatten

from .lbp import get_in_bounds, linear_lower_bound
from .alpha_crown import crown_relu, crown_rules, init_params as init_alpha_params


def beta_crown_lb(cg, var_bounds, splits, warm_start=None, lr=0.05, iters=8, record=None):
    """beta-CROWN lower bound for one branch.

    ``splits`` maps a ReLU output var to an int array with +1 (forced active),
    -1 (forced inactive) or 0 (not split). Returns the affine bound together
    with the optimised parameters, so the caller can warm-start the children
    with them - re-optimising from scratch in every branch is what makes naive
    node splitting unusable.
    """
    params = init_params(cg, var_bounds, splits) if warm_start is None else _copy(warm_start)
    split_arrays = {ov: Array(np.asarray(s, dtype=np.float64)) for ov, s in splits.items()}

    def apply_splits(ps):
        # split = 1 => relu split so that input x >= 0 => Lagrange term has negative sign
        # split = -1 => relu split so that input x < 0 => Lagrange term has positive sign
        return {
            ov: (p[0], -split_arrays[ov] * p[1]) if ov in split_arrays else p
            for ov, p in ps.items()
        }

    def loss(ps):
        affine_lb = linear_lower_bound(cg, var_bounds, apply_splits(ps), beta_crown_rules)
        return affine_lb.concrete(*get_in_bounds(cg.invars, var_bounds))

    def project(ov, node):
        if ov not in split_arrays:
            return core.clip(node, 0.0, 1.0)
        return (core.clip(node[0], 0.0, 1.0), core.maximum(node[1], 0.0))

    if len(params) > 0 and iters > 0:
        # Trace the loss once and reuse the graph for every ascent step. minijax's
        # grad() re-traces on each call, which dominates the per-branch cost.
        flat, structure = flatten(params)
        loss_cg = make_graph(lambda *ps: loss(unflatten(structure, ps)))(*flat)
        for _ in range(iters):
            flat, _ = flatten(params)
            primals = loss_cg(*flat)
            gs = _grad_backwards(loss_cg, primals, [Array(1.0)])
            gs = unflatten(structure, gs)
            params = map_structure(lambda p, g: p + lr * g, params, gs)
            params = {ov: project(ov, node) for ov, node in params.items()}

    affine = linear_lower_bound(
        cg, var_bounds, apply_splits(params), beta_crown_rules, record=record
    )
    return affine, params


def _copy(params):
    def cp(v):
        return tuple(cp(e) for e in v) if isinstance(v, tuple) else Array(np.asarray(v.array).copy())

    return {ov: cp(p) for ov, p in params.items()}


def init_params(cg, var_bounds, splits=()):
    """alpha for every activation, plus a beta multiplier for every split ReLU."""
    params = {}
    for outvar, alpha in init_alpha_params(cg, var_bounds).items():
        params[outvar] = (alpha, zeros(outvar.shape)) if outvar in splits else alpha
    return params


def beta_crown_relu(params, out_w, x):
    alpha, beta = params
    in_w, in_bias = crown_relu(alpha, out_w, x)
    return in_w + beta, in_bias


beta_crown_rules = dict(crown_rules)
beta_crown_rules[relu] = beta_crown_relu