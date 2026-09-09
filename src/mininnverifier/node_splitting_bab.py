# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Branch and bound over activation phases and, where it pays, over the input box."""

import heapq
import itertools
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
MIN_WIDTH = 1e-6
ACTIVATIONS = (relu, leaky_relu, elu, gelu)


def activation_eqns(cg):
    return [eqn for eqn in cg.equations if eqn.primitive in ACTIVATIONS]


def init_splits(act_eqns):
    return {
        eqn.outvar: (
            np.full(eqn.inputs[0].shape, -np.inf),
            np.full(eqn.inputs[0].shape, np.inf),
        )
        for eqn in act_eqns
    }


def apply_node_splits(var_bounds, act_eqns, splits):
    """Intersect each activation's pre-activation box with its branch's clips.
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
    refine_warmup=25,
    refine_hit_rate=0.05,
    refine_probe_every=400,
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

        counterexample = pgd_attack(fn, x_bounds, restarts=4, steps=25)
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

        # A heap instead of a list scan: best-first popped the worst branch with
        # an argmin over the whole queue, which is O(n) per step once the queue
        # runs into the thousands. The counter only breaks ties - boxes and
        # split dicts are not ordered, so they must never reach a comparison.
        # Refinement pays only if it actually prunes. It costs ~40x a plain
        # backward pass, so we watch its hit rate and stop paying for it once it
        # stops converting branches, re-probing occasionally in case the deeper
        # part of the tree behaves differently.
        tries = prunes = 0
        probe_at = 0
        tick = itertools.count()
        branches = [(-np.inf, next(tick), x_bounds, base_splits, root_bounds, None, 0)]
        while branches:
            if time.monotonic() > deadline:
                raise RuntimeError("verification budget exhausted")

            _, _, box, splits, base, warm_start, depth = heapq.heappop(branches)

            # A node split leaves the input box untouched, so the parent's CROWN
            # intermediate bounds are still exactly the right ones. Only an input
            # split invalidates them, and then they are recomputed lazily, on pop
            # rather than on push, so children that are never explored cost
            # nothing.
            if base is None:
                base = crown_var_bounds(cg, (box,), crown_rules)[0]
            var_bounds = apply_node_splits(base, act_eqns, splits)
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
            pops = next(tick) - 1
            worth_it = tries < refine_warmup or prunes >= refine_hit_rate * tries
            if pops >= probe_at:
                worth_it, probe_at = True, pops + refine_probe_every
            if beta_iters > 0 and worth_it and -refine_margin < child_lb < -TOL:
                tries += 1
                record = {}
                affine, params = beta_crown_lb(
                    cg, var_bounds, signs, warm_start=params,
                    iters=beta_iters, record=record,
                )
                child_lb = np.asarray(affine.concrete(*in_bounds).array).reshape(-1)[0]
                prunes += child_lb >= -TOL
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
                    heapq.heappush(branches, (
                        child_lb, next(tick), child_box, splits, None, params, depth + 1
                    ))
                continue
            for upper_half in (True, False):
                heapq.heappush(branches, (
                    child_lb, next(tick), box,
                    child_splits(splits, choice, upper_half), base, params, depth + 1
                ))

        return None # Verified

    return bab_fn