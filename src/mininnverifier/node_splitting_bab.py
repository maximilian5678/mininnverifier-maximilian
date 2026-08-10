# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Branch and bound over activation phases and, where it pays, over the input box."""

import time

import numpy as np

from minijax.compute_graph import make_graph
from minijax.core import relu, leaky_relu, elu, gelu
from minijax.eval import Array

from .ibp import Box
from .lbp import get_in_bounds
from .beta_crown import beta_crown_lb
from .crown_bounds import crown_var_bounds
from .alpha_crown import crown_rules
from .attack import pgd_attack, concretization_minimizer, evaluate
from .fuse import fuse_activations
from .input_splitting_bab import split_longest_edge

TOL = 1e-9
MIN_WIDTH = 1e-6  # don't keep bisecting an interval that is already a point
ACTIVATIONS = (relu, leaky_relu, elu, gelu)


def activation_eqns(cg):
    return [eqn for eqn in cg.equations if eqn.primitive in ACTIVATIONS]


def init_splits(act_eqns):
    """No split yet: every activation input keeps its unconstrained interval.

    A split is stored as a pair of clips rather than a ReLU phase, so that the
    same mechanism covers smooth activations: for those a split bisects the
    input interval instead of fixing a phase (there is no phase to fix).
    """
    return {
        eqn.outvar: (
            np.full(eqn.inputs[0].shape, -np.inf),
            np.full(eqn.inputs[0].shape, np.inf),
        )
        for eqn in act_eqns
    }


def apply_node_splits(var_bounds, act_eqns, splits):
    """Intersect each activation's pre-activation box with its branch's clips.

    Fixing a ReLU active means its input is non-negative on this branch, so the
    relaxation for that neuron becomes exact; bisecting a GELU's input halves
    the relaxation gap. This, rather than the beta multipliers alone, is what
    makes node splitting pay off.
    """
    var_bounds = dict(var_bounds)
    for eqn in act_eqns:
        lo_clip, hi_clip = splits[eqn.outvar]
        if not (np.isfinite(lo_clip).any() or np.isfinite(hi_clip).any()):
            continue
        box = var_bounds[eqn.inputs[0]]
        lb = np.maximum(np.asarray(box.lb.array), lo_clip)
        ub = np.minimum(np.asarray(box.ub.array), hi_clip)
        var_bounds[eqn.inputs[0]] = Box(Array(lb), Array(np.maximum(ub, lb)))
    return var_bounds


def beta_signs(act_eqns, splits):
    """Lagrange sign per split ReLU: +1 forced active, -1 forced inactive, 0 free.

    Only ReLU nodes get beta multipliers; for the smooth activations the clipped
    interval is the whole constraint.
    """
    signs = {}
    for eqn in act_eqns:
        if eqn.primitive is not relu:
            continue
        lo_clip, hi_clip = splits[eqn.outvar]
        signs[eqn.outvar] = np.where(lo_clip >= 0.0, 1.0, np.where(hi_clip <= 0.0, -1.0, 0.0))
    return signs


def _relaxation_gap(primitive, lb, ub, options):
    """How much the linear relaxation of this neuron can be off, per element."""
    width = ub - lb
    straddles = (lb < 0.0) & (ub > 0.0)
    if primitive is relu:
        return np.where(straddles, -lb * ub / np.maximum(width, 1e-12), 0.0)
    if primitive is leaky_relu:
        slope = options.get("slope", 0.01)
        return (1.0 - slope) * np.where(straddles, -lb * ub / np.maximum(width, 1e-12), 0.0)
    # Smooth activations: a chord deviates from a convex function by at most
    # |f''|/8 * width^2, and |f''| <= 1 for both elu and gelu.
    return 0.125 * width * width


def input_split_gain(affine, box, total_slack):
    """Bound improvement an input split can buy, in output units.

    Two parts. The direct one: halving the widest weighted input edge removes
    half of that edge's own concretisation slack. The indirect one matters far
    more: a smaller box makes CROWN recompute *every* intermediate bound
    tighter, shrinking all relaxation gaps at once. Halving one of n input
    edges shrinks the box along a 1/n share, so we credit the input split with
    that share of the total relaxation slack. Same units as the node scores,
    which is what lets us choose between the two split kinds per branch instead
    of alternating on a fixed schedule.
    """
    w = np.abs(np.asarray(affine.weights[0].array).reshape(-1))
    lb = np.asarray(box.lb.array).reshape(-1)
    ub = np.asarray(box.ub.array).reshape(-1)
    widths = ub - lb
    if w.size == 0 or not (widths > 0.0).any():
        return -np.inf
    n_open = int((widths > 0.0).sum())
    direct = 0.5 * float(np.max(w * widths))
    return direct + 0.5 * total_slack / n_open


def pick_node_split(var_bounds, act_eqns, splits, record):
    """BaBSR-style heuristic: branch where the relaxation loses the most.

    Score = relaxation gap times the weight the backward pass gave the neuron,
    i.e. an estimate of how much the output bound improves once the neuron is
    split. Same units as ``input_split_gain``.
    """
    best, best_score, total_slack = None, -np.inf, 0.0
    for eqn in act_eqns:
        box = var_bounds[eqn.inputs[0]]
        lb, ub = np.asarray(box.lb.array), np.asarray(box.ub.array)
        splittable = (ub - lb) > MIN_WIDTH
        if not splittable.any():
            continue
        out_w = record.get(eqn.outvar)
        w = np.abs(np.asarray(out_w.array)) if out_w is not None else np.ones(lb.shape)
        gap = _relaxation_gap(eqn.primitive, lb, ub, eqn.options)
        weighted = np.where(splittable, np.broadcast_to(w, lb.shape) * gap, 0.0)
        total_slack += float(weighted.sum())
        score = np.where(splittable, weighted, -np.inf)
        i = int(np.argmax(score))
        if score.reshape(-1)[i] > best_score:
            # split at zero where the activation bends, otherwise bisect
            lo, hi = lb.reshape(-1)[i], ub.reshape(-1)[i]
            point = 0.0 if lo < 0.0 < hi else 0.5 * (lo + hi)
            best_score, best = score.reshape(-1)[i], (eqn.outvar, i, point)
    return best, best_score, total_slack


def child_splits(splits, choice, upper_half):
    outvar, index, point = choice
    lo_clip, hi_clip = splits[outvar]
    lo_clip, hi_clip = lo_clip.copy(), hi_clip.copy()
    if upper_half:
        lo_clip.reshape(-1)[index] = max(lo_clip.reshape(-1)[index], point)
    else:
        hi_clip.reshape(-1)[index] = min(hi_clip.reshape(-1)[index], point)
    child = dict(splits)
    child[outvar] = (lo_clip, hi_clip)
    return child


def node_splitting_bab(
    fn,
    time_limit=540.0,
    beta_iters=8,
    refine_fraction=0.5,
    input_split_bias=1.0,
):
    """Branch and bound on top of beta-CROWN, over neuron phases and input box.

    Which kind of split to take is decided per branch by comparing the estimated
    bound improvement of each, not by a fixed schedule. Splitting the input box
    lets CROWN recompute every intermediate bound at once, which node splitting
    cannot do, and on low-dimensional inputs that dominates; on a 784-pixel box
    it is nearly worthless and node splitting wins. ``input_split_bias`` scales
    the input-split estimate if you want to force the balance either way.
    """

    def bab_fn(x_bounds: Box):
        deadline = time.monotonic() + time_limit
        cg = fuse_activations(make_graph(fn)(x_bounds.lb))
        act_eqns = activation_eqns(cg)

        counterexample = pgd_attack(fn, x_bounds, restarts=8, steps=40)
        if counterexample is not None:
            return counterexample

        base_splits = init_splits(act_eqns)
        root_bounds = crown_var_bounds(cg, (x_bounds,), crown_rules)[0]
        root_affine, _ = beta_crown_lb(
            cg, root_bounds, beta_signs(act_eqns, base_splits), iters=0
        )
        root_lb = np.asarray(
            root_affine.concrete(*get_in_bounds(cg.invars, root_bounds)).array
        ).reshape(-1)[0]
        refine_margin = max(abs(root_lb) * refine_fraction, 1e-6)

        branches = [(-np.inf, x_bounds, base_splits, None, 0)]
        visited = 0
        while branches:
            if time.monotonic() > deadline:
                raise RuntimeError("verification budget exhausted")

            worst = int(np.argmin([b[0] for b in branches]))
            _, box, splits, warm_start, depth = branches.pop(worst)

            var_bounds = crown_var_bounds(cg, (box,), crown_rules)[0]
            var_bounds = apply_node_splits(var_bounds, act_eqns, splits)
            in_bounds = get_in_bounds(cg.invars, var_bounds)
            signs = beta_signs(act_eqns, splits)

            # Cheap bound first. Optimising alpha/beta costs about 30x a plain
            # backward pass, so only spend it on branches close enough to zero
            # for the extra tightness to actually prune them.
            record = {}
            affine, params = beta_crown_lb(
                cg, var_bounds, signs, warm_start=warm_start, iters=0, record=record
            )
            child_lb = np.asarray(affine.concrete(*in_bounds).array).reshape(-1)[0]
            visited += 1
            if child_lb < -TOL and child_lb > -refine_margin:
                record = {}
                affine, params = beta_crown_lb(
                    cg, var_bounds, signs, warm_start=params,
                    iters=beta_iters, record=record,
                )
                child_lb = np.asarray(affine.concrete(*in_bounds).array).reshape(-1)[0]
            if child_lb >= -TOL:  # branch verified
                continue

            # the minimiser of the affine bound is a free counterexample candidate
            candidate = concretization_minimizer(affine, box)
            if evaluate(fn, candidate) < 0.0:
                return candidate

            choice, node_gain, total_slack = pick_node_split(
                var_bounds, act_eqns, splits, record
            )
            input_gain = input_split_bias * input_split_gain(affine, box, total_slack)
            if choice is None or input_gain > node_gain:
                for child_box in split_longest_edge(box):
                    branches.append((child_lb, child_box, splits, params, depth + 1))
                continue
            for upper_half in (True, False):
                branches.append(
                    (child_lb, box, child_splits(splits, choice, upper_half),
                     params, depth + 1)
                )

        return None  # Verified

    return bab_fn