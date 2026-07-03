#Copyright (c) 2026 by David Boetius
#Licensed under the MIT License.
"""Input-splitting branch and bound with a falsifier and a heap-ordered queue."""
import heapq
import itertools
import sys

import numpy as np

from minijax.eval import Array
from minijax.grad import value_and_grad

from .ibp import ibp, Box


def falsify(f, box: Box, n_random=4096, n_restarts=20, n_steps=60, seed=0):
    """Search for x in the box with f(x) < 0. Returns an Array or None."""
    lb = box.lb.array.reshape(-1)
    ub = box.ub.array.reshape(-1)
    shape = box.lb.array.shape
    rng = np.random.default_rng(seed)

    def val(x_flat):
        return float(np.asarray(f(Array(x_flat.reshape(shape))).array).reshape(()))

    best_x, best_v = None, np.inf
    samples = [lb, ub, 0.5 * (lb + ub)]
    samples += [lb + rng.random(lb.shape) * (ub - lb) for _ in range(n_random)]
    for x in samples:
        v = val(x)
        if v < best_v:
            best_v, best_x = v, x.copy()
        if v < 0.0:
            return Array(x.reshape(shape))

    vg = value_and_grad(lambda x: f(x))
    starts = [best_x] + [lb + rng.random(lb.shape) * (ub - lb)
                         for _ in range(n_restarts - 1)]
    span = np.maximum(ub - lb, 1e-12)
    for x0 in starts:
        x = x0.copy()
        step = 0.1 * span
        for _ in range(n_steps):
            v, g = vg(Array(x.reshape(shape)))
            v = float(np.asarray(v.array).reshape(()))
            if v < 0.0:
                return Array(x.reshape(shape))
            g = g[0] if isinstance(g, tuple) else g
            g = np.asarray(g.array).reshape(-1)
            x = np.clip(x - step * np.sign(g) * span, lb, ub)
            step *= 0.92
        v = val(x)
        if v < best_v:
            best_v, best_x = v, x.copy()
    return None


def split_longest_edge(branch: Box):
    lb = branch.lb.array.reshape(-1)
    ub = branch.ub.array.reshape(-1)
    index = int((ub - lb).argmax())
    mid = 0.5 * (lb[index] + ub[index])
    shape = branch.lb.array.shape
    left_ub = ub.copy()  
    left_ub[index] = mid
    right_lb = lb.copy()
    right_lb[index] = mid
    left = Box(branch.lb, Array(left_ub.reshape(shape)))
    right = Box(Array(right_lb.reshape(shape)), branch.ub)
    return left, right

def input_splitting_bab(fn, split=split_longest_edge, compute_bounds=ibp):
    bounds_fn = compute_bounds(fn)

    def bab_fn(x_bounds: Box):
        # Disprove first: most hard instances here are violations.
        ce = falsify(fn, x_bounds)
        if ce is not None:
            return ce

        counter = itertools.count()
        root = bounds_fn(x_bounds)
        heap = [(root.lb.item(), next(counter), x_bounds)]
        steps = 0
        while heap:
            steps += 1
            lb, _, branch = heapq.heappop(heap)
            if lb >= 0.0:
                return None  # every remaining branch is verified -> sat
            for child in split(branch):
                cb = bounds_fn(child)
                c_lb, c_ub = cb.lb.item(), cb.ub.item()
                if c_ub < 0.0:
                    ce = falsify(fn, child, n_random=256, n_restarts=8)
                    mid = 0.5 * (child.lb.array + child.ub.array)
                    return ce if ce is not None else Array(mid)
                if c_lb < 0.0:
                    heapq.heappush(heap, (c_lb, next(counter), child))
        return None  # verified -> sat
    return bab_fn
