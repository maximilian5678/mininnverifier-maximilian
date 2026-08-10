# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Projected gradient descent, used to look for counterexamples.

Branch and bound only ever proves properties; a violated property is found much
faster by attacking it directly (Madry et al., ICLR 2018). Every branch also
hands us a candidate for free: the point where its affine lower bound attains
its minimum over the box.
"""

import numpy as np

from minijax.eval import Array
from minijax.grad import grad


def concretization_minimizer(affine_bound, box):
    """The corner of ``box`` where the affine lower bound is smallest."""
    w = np.asarray(affine_bound.weights[0].array)
    lb, ub = np.asarray(box.lb.array), np.asarray(box.ub.array)
    return Array(np.where(w > 0.0, lb, ub))


def evaluate(fn, x):
    return float(np.asarray(fn(x).array).reshape(-1)[0])


def pgd_attack(fn, box, restarts=8, steps=40, seed=0):
    """Search for x in the box with fn(x) < 0. Returns the point or None."""
    lb, ub = np.asarray(box.lb.array), np.asarray(box.ub.array)
    width = ub - lb
    rng = np.random.default_rng(seed)
    fn_grad = grad(fn)

    for r in range(restarts):
        x = 0.5 * (lb + ub) if r == 0 else rng.uniform(lb, ub)
        step = 0.25 * width
        for _ in range(steps):
            xa = Array(x)
            if evaluate(fn, xa) < 0.0:
                return xa
            g = np.asarray(fn_grad(xa)[0].array)
            x = np.clip(x - step * np.sign(g), lb, ub)
            step = step * 0.9
    return None