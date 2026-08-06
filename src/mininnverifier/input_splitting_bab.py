# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
import numpy as np

from minijax.core import reshape, where
from minijax.eval import Array

from .ibp import ibp, Box


def split_longest_edge(branch: Box):
    lb, ub = branch.lb, branch.ub
    lb_flat = reshape(lb, new_shape=(-1,))
    ub_flat = reshape(ub, new_shape=(-1,))
    numel = lb_flat.shape[0]  # number of elements

    # Find argument with longest edge
    ranges = ub_flat - lb_flat
    longest_edge, index = ranges[0].item(), 0
    for i in range(1, numel):
        ran = ranges[i].item()
        if ran > longest_edge:
            longest_edge, index = ran, i

    mid = (ub_flat + lb_flat) / 2.0
    mask = Array(np.arange(numel) == index)
    left_ub = where(mask, mid, ub_flat)
    right_lb = where(mask, mid, lb_flat)

    left_ub = reshape(left_ub, ub.shape)
    right_lb = reshape(right_lb, ub.shape)
    return Box(lb, left_ub), Box(right_lb, ub)


def pick_worst_lb(branches):
    min_lb, selected = branches[0][0], 0
    for i in range(1, len(branches)):
        lb, _ = branches[i]
        if lb < min_lb:
            min_lb, selected = lb, i
    return selected


def input_splitting_bab(fn, split=split_longest_edge, compute_bounds=ibp):
    # Verify whether fn(x) >= 0 for all x in x_bounds
    # Compare with https://jmlr.org/papers/v21/19-468.html
    compute_bounds = compute_bounds(fn)

    def bab_fn(x_bounds: Box):
        branches = [(-np.inf, x_bounds)]
        while len(branches) > 0:
            branch_i = pick_worst_lb(branches)
            _, branch = branches.pop(branch_i)
            children = split(branch)
            for child_branch in children:
                child_bounds = compute_bounds(child_branch)
                child_lb, child_ub = child_bounds.lb.item(), child_bounds.ub.item()
                if child_ub < 0:
                    return (child_branch.ub + child_branch.lb) / 2
                if child_lb < 0:
                    branches.append((child_lb, child_branch))
        return None  # Verified

    return bab_fn