"""Check if a neural network is safe on a box of inputs.

We are given a function f. The property we want is: f(x) >= 0 for every
input x inside a box.

There are two possible answers:
  - "sat"  : f(x) >= 0 everywhere in the box. The network is safe.
  - "viol" : there is some x in the box with f(x) < 0. We return that x
             as a counterexample.

I use three tools:

1. find_counterexample(f, box)
   Try to find a bad point (f(x) < 0) by guessing. It checks the corners,
   the middle, and many random points. If it finds f(x) < 0, that x is a counterexample.

2. IBP is called through bounds_fn.
   IF:
     - lb >= 0  -> whole box is safe. Done with this box.
     - ub < 0   -> whole box is bad. Any point in it is a counterexample.
     - lb < 0 <= ub -> unclear. Box too big for IBP. Split it.

3. split (longest-edge heuristic)
   Cut a box in half along its widest side.
"""

import heapq
import itertools

import numpy as np

from minijax.eval import Array
from minijax.vmap import vmap

from .ibp import ibp, Box


def find_counterexample(f, box: Box, n_random=4096, seed=0):
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
    return None


def split_longest_edge_batch(lb, ub):
    shape = lb.shape[1:]
    K = lb.shape[0]
    flat_lb = lb.reshape(K, -1)
    flat_ub = ub.reshape(K, -1)
    rows = np.arange(K)
    idx = (flat_ub - flat_lb).argmax(axis=1)
    mid = 0.5 * (flat_lb[rows, idx] + flat_ub[rows, idx])

    left_ub = flat_ub.copy()
    left_ub[rows, idx] = mid
    right_lb = flat_lb.copy() 
    right_lb[rows, idx] = mid 

    D = flat_lb.shape[1]
    child_lb = np.empty((2 * K, D), dtype=flat_lb.dtype)
    child_ub = np.empty((2 * K, D), dtype=flat_ub.dtype)
    child_lb[0::2] = flat_lb
    child_ub[0::2] = left_ub
    child_lb[1::2] = right_lb
    child_ub[1::2] = flat_ub
    return child_lb.reshape((2 * K,) + shape), child_ub.reshape((2 * K,) + shape)

def input_splitting_bab(fn, split=split_longest_edge_batch, compute_bounds=ibp, batch_size=256):
    single_bounds = compute_bounds(fn) # ibp(fn): one box
    batched_bounds = compute_bounds(vmap(fn)) # ibp(vmap(fn)): a batch of boxes

    def bab_fn(x_bounds: Box):
        # Disprove first: most hard instances here are violations.
        ce = find_counterexample(fn, x_bounds)
        if ce is not None:
            return ce

        root = single_bounds(x_bounds)
        if root.lb.item() >= 0.0:
            return None # verified at the root -> sat

        probe_lb = np.stack([x_bounds.lb.array, x_bounds.lb.array], axis=0)
        probe_ub = np.stack([x_bounds.ub.array, x_bounds.ub.array], axis=0)
        batched_bounds(Box(Array(probe_lb), Array(probe_ub)))

        counter = itertools.count()
        heap = [(root.lb.item(), next(counter),
                 x_bounds.lb.array, x_bounds.ub.array)]
        steps = 0
        while heap:
            steps += 1
            k = min(batch_size, len(heap))
            popped = [heapq.heappop(heap) for _ in range(k)]
            if popped[0][0] >= 0.0:
                return None  # global-min lb >= 0 -> all branches verified

            lb_stack = np.stack([p[2] for p in popped], axis=0)
            ub_stack = np.stack([p[3] for p in popped], axis=0)
            child_lb, child_ub = split(lb_stack, ub_stack)

            cb = batched_bounds(Box(Array(child_lb), Array(child_ub)))
            n_children = child_lb.shape[0]
            out_lb = np.asarray(cb.lb.array).reshape(n_children, -1)[:, 0]
            out_ub = np.asarray(cb.ub.array).reshape(n_children, -1)[:, 0]

            for i in range(n_children):
                c_ub = float(out_ub[i])
                if c_ub < 0.0:
                    child = Box(Array(child_lb[i]), Array(child_ub[i]))
                    ce = find_counterexample(fn, child, n_random=256, n_restarts=8)
                    mid = 0.5 * (child.lb.array + child.ub.array)
                    return ce if ce is not None else Array(mid)
                c_lb = float(out_lb[i])
                if c_lb < 0.0:
                    heapq.heappush(heap, (c_lb, next(counter), child_lb[i], child_ub[i]))
        return None # queue exhausted -> every branch verified
    return bab_fn