# # Copyright (c) 2026 by David Boetius
# # Licensed under the MIT License.
# import numpy as np

# from minijax.eval import Array

# from .ibp import ibp, Box


# def split_longest_edge(branch: Box):
#     """Split the box along its widest dimension into two halves."""
#     lb = branch.lb.array.reshape(-1)
#     ub = branch.ub.array.reshape(-1)
#     index = int((ub - lb).argmax()) # dimension with the widest edge
#     mid = 0.5 * (lb[index] + ub[index])

#     left_ub = ub.copy()
#     left_ub[index] = mid
#     right_lb = lb.copy()
#     right_lb[index] = mid

#     shape = branch.lb.array.shape
#     left = Box(branch.lb, Array(left_ub.reshape(shape)))
#     right = Box(Array(right_lb.reshape(shape)), branch.ub)
#     return left, right

# # def split_smart(branch: Box, compute_bounds):
# #     """Split the dimension whose bisection raises the worse child lower bound most."""
# #     lb = branch.lb.array.reshape(-1)
# #     ub = branch.ub.array.reshape(-1)
# #     shape = branch.lb.array.shape

# #     widths = ub - lb
# #     candidates = np.where(widths > 1e-12)[0]
# #     if len(candidates) == 0:
# #         candidates = [int(widths.argmax())]

# #     best_index, best_score = candidates[0], -np.inf
# #     best_children = None
# #     for index in candidates:
# #         mid = 0.5 * (lb[index] + ub[index])
# #         left_ub = ub.copy();  left_ub[index]  = mid
# #         right_lb = lb.copy(); right_lb[index] = mid
# #         left = Box(branch.lb, Array(left_ub.reshape(shape)))
# #         right = Box(Array(right_lb.reshape(shape)), branch.ub)

# #         l_lb = compute_bounds(left).lb.item()
# #         r_lb = compute_bounds(right).lb.item()
# #         score = min(l_lb, r_lb)
# #         if score > best_score:
# #             best_score, best_index = score, index
# #             best_children = (left, right)

# #     return best_children


# def pick_worst_lb(branches):
#     """Index of the branch with the smallest (most negative) lower bound."""
#     min_lb, selected = branches[0][0], 0
#     for i in range(1, len(branches)):
#         lb, _ = branches[i]
#         if lb < min_lb:
#             min_lb, selected = lb, i
#     return selected


# def input_splitting_bab(fn, split=split_longest_edge, compute_bounds=ibp):
#     compute_bounds = compute_bounds(fn)

#     def bab_fn(x_bounds: Box):
#         branches = [(-np.inf, x_bounds)]
#         steps = 0
#         while branches:
#             if steps % 2000 == 0:
#                 best = min(b[0] for b in branches)
#                 print(f"[bab] steps={steps} queue={len(branches)} worst_lb={best:.4f}", file=__import__("sys").stderr)
#             branch = branches.pop(pick_worst_lb(branches))[1]
#             for child in split(branch):
#                 cb = compute_bounds(child)
#                 child_lb, child_ub = cb.lb.item(), cb.ub.item()
#                 if child_ub < 0:
#                     mid = 0.5 * (child.lb.array + child.ub.array)
#                     return Array(mid)
#                 if child_lb < 0:
#                     branches.append((child_lb, child))
#         return None  # verified
#     return bab_fn

# # def input_splitting_bab(fn, compute_bounds=ibp):
# #     compute_bounds = compute_bounds(fn)

# #     def bab_fn(x_bounds: Box):
# #         branches = [(-np.inf, x_bounds)]
# #         steps = 0
# #         while branches:
# #             if steps % 2000 == 0:
# #                 best = min(b[0] for b in branches)
# #                 print(f"[bab] steps={steps} queue={len(branches)} worst_lb={best:.4f}", file=__import__("sys").stderr)
# #             branch = branches.pop(pick_worst_lb(branches))[1]
# #             for child in split_smart(branch, compute_bounds):
# #                 cb = compute_bounds(child)
# #                 child_lb, child_ub = cb.lb.item(), cb.ub.item()
# #                 if child_ub < 0:
# #                     mid = 0.5 * (child.lb.array + child.ub.array)
# #                     return Array(mid)
# #                 if child_lb < 0:
# #                     branches.append((child_lb, child))
# #         return None
# #     return bab_fn

# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
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

    # 1) corners + midpoint + uniform random samples
    best_x, best_v = None, np.inf
    samples = [lb, ub, 0.5 * (lb + ub)]
    samples += [lb + rng.random(lb.shape) * (ub - lb) for _ in range(n_random)]
    for x in samples:
        v = val(x)
        if v < best_v:
            best_v, best_x = v, x.copy()
        if v < 0.0:
            return Array(x.reshape(shape))

    # 2) projected sign-gradient descent from the best random starts
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
    left_ub = ub.copy();  left_ub[index] = mid
    right_lb = lb.copy(); right_lb[index] = mid
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
            if steps % 5000 == 0:
                print(f"[bab] steps={steps} queue={len(heap)} "
                      f"worst_lb={heap[0][0]:.4f}", file=sys.stderr)
        return None  # verified -> sat
    return bab_fn