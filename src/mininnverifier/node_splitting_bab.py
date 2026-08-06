# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
import numpy as np

from minijax.compute_graph import make_graph
from minijax.core import relu, concat, reshape, where, split
from minijax.eval import zeros, Array

from .ibp import ibp, Box
from .lbp import get_in_bounds
from .beta_crown import beta_crown_lb
from .input_splitting_bab import pick_worst_lb


def flatten_splits(splits):
    flat_splits = [reshape(s, new_shape=(-1,)) for s in splits.values()]
    sizes = [s.shape[0] for s in flat_splits]
    return concat(*flat_splits), sizes


def unflatten_splits(split_vector, reference, split_sizes):
    split_points = list(np.cumsum(split_sizes)[:-1])
    flat_splits = split(split_vector, split_points)
    return {ov: reshape(f, s.shape) for f, (ov, s) in zip(flat_splits, reference.items())}


def split_last(splits):
    # assumes splits dict is sorted by variable order
    split_vector, split_sizes = flatten_splits(splits)
    num_relus = split_vector.shape[0]

    last_unsplit = None
    for i in reversed(range(num_relus)):
        if split_vector[i].item() == 0.0:
            last_unsplit = i
            break

    if last_unsplit is None:
        return (splits,)

    mask = Array(np.arange(num_relus) == last_unsplit)
    left = where(mask, 1.0, split_vector)  # relu >= 0
    right = where(mask, -1.0, split_vector)  # relu == 0
    left = unflatten_splits(left, splits, split_sizes)
    right = unflatten_splits(right, splits, split_sizes)
    return left, right


def init_splits(cg):
    """An empty split assignment (no node split yet) for every ReLU node."""
    splits = {}
    for eqn in cg.equations:
        if eqn.primitive is relu:
            splits[eqn.outvar] = zeros(eqn.inputs[0].shape)
    return splits


def node_splitting_bab(fn, split=split_last, init_bounds=ibp):
    def bab_fn(x_bounds: Box):
        cg = make_graph(fn)(x_bounds.lb)
        var_bounds = init_bounds(cg)(x_bounds)
        base_splits = init_splits(cg)

        branches = [(-np.inf, base_splits)]
        while len(branches) > 0:
            branch_i = pick_worst_lb(branches)
            _, splits = branches.pop(branch_i)
            for child_splits in split(splits):
                affine_lb = beta_crown_lb(cg, var_bounds, child_splits)
                child_lb = affine_lb.concrete(*get_in_bounds(cg.invars, var_bounds)).item()
                # TODO: upper bounds & counterexamples
                # TODO: prune infeasible branches
                if child_lb < -1e-8:  # treats floating-point errors
                    branches.append((child_lb, child_splits))
        return None  # Verified

    return bab_fn
