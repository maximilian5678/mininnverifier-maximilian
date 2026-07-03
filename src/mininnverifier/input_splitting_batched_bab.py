"""Check if a neural network is safe on a box of inputs.

We are given a function f. The property we want is: f(x) >= 0 for every
input x inside a box.

There are two possible answers:
  - "sat"  : f(x) >= 0 everywhere in the box. The network is safe.
  - "viol" : there is some x in the box with f(x) < 0. We return that x
             as a counterexample.

We use three tools:

1. falsify(f, box)
   Try to find a bad point (f(x) < 0) by guessing. It checks the corners,
   the middle, and many random points, then walks downhill using the
   gradient. If it finds f(x) < 0, that x is a counterexample. If it finds
   nothing, that does NOT prove the network is safe. It can only disprove.

2. IBP (interval bound propagation), called through bounds_fn.
   Given a box, it returns a range [lb, ub] that f is guaranteed to stay
   inside over the WHOLE box: lb <= f(x) <= ub for every x in the box.
   lb is the lower bound (smallest possible value), ub the upper bound
   (largest possible value). Sound but often loose.
     - lb >= 0  -> whole box is safe. Done with this box.
     - ub < 0   -> whole box is bad. Any point in it is a counterexample.
     - lb < 0 <= ub -> unclear. Box too big for IBP. Split it.

3. split (longest-edge)
   Cut a box in half along its widest side. This makes two smaller boxes
   that together cover the old one. Smaller boxes give tighter IBP ranges,
   so unclear boxes become clear after enough splits.

Main loop (branch and bound):
   Keep a queue of unclear boxes, ordered by their lower bound (worst first).
   Take boxes out, split them, run IBP on the children:
     - child safe    (lb >= 0)  -> throw it away.
     - child bad     (ub < 0)   -> return a counterexample.
     - child unclear (lb < 0 <= ub) -> put it back in the queue.
   If the worst box in the queue has lb >= 0, then every box is safe -> sat.
   If the queue empties with no bad box found -> sat.

Speed trick (batching):
   Running IBP on one box at a time is slow, because the Python overhead per
   call is large. Instead we pop many boxes at once, stack them into one big
   array, and run IBP on all of them in a single call using vmap. This does
   the same work but with far fewer, larger operations.
"""

import heapq
import itertools
import sys

import numpy as np

from minijax.eval import Array
from minijax.grad import value_and_grad
from minijax.vmap import vmap

from .ibp import ibp, Box


def falsify(f, box: Box, n_random=4096, n_restarts=20, n_steps=60, seed=0):
    """Search for x in the box with f(x) < 0. Returns an Array or None."""
    lb = box.lb.array.reshape(-1)
    ub = box.ub.array.reshape(-1)
    shape = box.lb.array.shape
    rng = np.random.default_rng(seed)

    def val(x_flat):
        return float(np.asarray(f(Array(x_flat.reshape(shape))).array).reshape(()))

    # Phase 1: corners + midpoint + uniform random samples
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

    # Phase 2: projected sign-gradient descent from the best random starts
    # vg = value_and_grad(lambda x: f(x))
    # starts = [best_x] + [lb + rng.random(lb.shape) * (ub - lb)
    #                      for _ in range(n_restarts - 1)]
    # span = np.maximum(ub - lb, 1e-12)
    # for x0 in starts:
    #     x = x0.copy()
    #     step = 0.1 * span
    #     for _ in range(n_steps):
    #         v, g = vg(Array(x.reshape(shape)))
    #         v = float(np.asarray(v.array).reshape(()))
    #         if v < 0.0:
    #             return Array(x.reshape(shape))
    #         g = g[0] if isinstance(g, tuple) else g
    #         g = np.asarray(g.array).reshape(-1)
    #         x = np.clip(x - step * np.sign(g) * span, lb, ub)
    #         step *= 0.92
    #     v = val(x)
    #     if v < best_v:
    #         best_v, best_x = v, x.copy()
    # return None


# def split_longest_edge(branch: Box):
#     """Scalar split of one box along its widest edge (used by the fallback)."""
#     lb = branch.lb.array.reshape(-1)
#     ub = branch.ub.array.reshape(-1)
#     index = int((ub - lb).argmax())
#     mid = 0.5 * (lb[index] + ub[index])
#     shape = branch.lb.array.shape
#     left_ub = ub.copy()
#     left_ub[index] = mid
#     right_lb = lb.copy()
#     right_lb[index] = mid
#     left = Box(branch.lb, Array(left_ub.reshape(shape)))
#     right = Box(Array(right_lb.reshape(shape)), branch.ub)
#     return left, right


def split_longest_edge_batch(lb, ub):
    """Vectorised widest-edge split.

    lb, ub: arrays of shape (K, *input_shape). Returns (child_lb, child_ub),
    each of shape (2K, *input_shape), with the two children of branch i at
    positions 2i (left) and 2i+1 (right).
    """
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


# def _serial_bab(fn, x_bounds, bounds_fn, split):
#     """Original one-box-at-a-time loop. Used for networks vmap cannot batch."""
#     counter = itertools.count()
#     root = bounds_fn(x_bounds)
#     heap = [(root.lb.item(), next(counter), x_bounds)]
#     steps = 0
#     while heap:
#         steps += 1
#         lb, _, branch = heapq.heappop(heap)
#         if lb >= 0.0:
#             return None
#         for child in split(branch):
#             cb = bounds_fn(child)
#             c_lb, c_ub = cb.lb.item(), cb.ub.item()
#             if c_ub < 0.0:
#                 ce = falsify(fn, child, n_random=256, n_restarts=8)
#                 mid = 0.5 * (child.lb.array + child.ub.array)
#                 return ce if ce is not None else Array(mid)
#             if c_lb < 0.0:
#                 heapq.heappush(heap, (c_lb, next(counter), child))
#         if steps % 5000 == 0:
#             print(f"[bab] steps={steps} queue={len(heap)} "
#                   f"worst_lb={heap[0][0]:.4f}", file=sys.stderr)
#     return None


def input_splitting_bab(fn, split=split_longest_edge_batch, compute_bounds=ibp, batch_size=256):
    single_bounds = compute_bounds(fn)          # ibp(fn)      : one box
    batched_bounds = compute_bounds(vmap(fn))   # ibp(vmap(fn)): a batch of boxes

    def bab_fn(x_bounds: Box):
        # Disprove first: most hard instances here are violations.
        ce = falsify(fn, x_bounds)
        if ce is not None:
            return ce

        root = single_bounds(x_bounds)
        if root.lb.item() >= 0.0:
            return None  # verified at the root -> sat

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
                    ce = falsify(fn, child, n_random=256, n_restarts=8)
                    mid = 0.5 * (child.lb.array + child.ub.array)
                    return ce if ce is not None else Array(mid)
                c_lb = float(out_lb[i])
                if c_lb < 0.0:
                    heapq.heappush(heap, (c_lb, next(counter), child_lb[i], child_ub[i]))
        return None # queue exhausted -> every branch verified
    return bab_fn