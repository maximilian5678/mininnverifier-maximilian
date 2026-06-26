# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
import numpy as np

from minijax.eval import Array

from .ibp import ibp, Box


def split_longest_edge(branch: Box):
    """Split the box along its widest dimension into two halves."""
    lb = branch.lb.array.reshape(-1)
    ub = branch.ub.array.reshape(-1)
    index = int((ub - lb).argmax()) # dimension with the widest edge
    mid = 0.5 * (lb[index] + ub[index])

    left_ub = ub.copy()
    left_ub[index] = mid
    right_lb = lb.copy()
    right_lb[index] = mid

    shape = branch.lb.array.shape
    left = Box(branch.lb, Array(left_ub.reshape(shape)))
    right = Box(Array(right_lb.reshape(shape)), branch.ub)
    return left, right


def pick_worst_lb(branches):
    """Index of the branch with the smallest (most negative) lower bound."""
    min_lb, selected = branches[0][0], 0
    for i in range(1, len(branches)):
        lb, _ = branches[i]
        if lb < min_lb:
            min_lb, selected = lb, i
    return selected


def input_splitting_bab(fn, split=split_longest_edge, compute_bounds=ibp):
    # Verify whether fn(x) >= 0 for all x in x_bounds.
    compute_bounds = compute_bounds(fn)

    def bab_fn(x_bounds: Box):
        branches = [(-np.inf, x_bounds)]
        while branches:
            branch = branches.pop(pick_worst_lb(branches))[1]
            for child in split(branch):
                cb = compute_bounds(child)
                child_lb, child_ub = cb.lb.item(), cb.ub.item()
                if child_ub < 0:
                    mid = 0.5 * (child.lb.array + child.ub.array)
                    return Array(mid)
                if child_lb < 0:
                    branches.append((child_lb, child))
        return None  # verified
    return bab_fn