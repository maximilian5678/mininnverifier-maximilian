"""Projected gradient descent to find counterexamples.

Branch and bound only proves that a property holds. To find a  violation we use PGD. Each branch also provides a free 
candidate point at the minimum of its lower bound.
"""

import numpy as np

from minijax.compute_graph import make_graph
from minijax.eval import Array
from minijax.grad import _grad_backwards


def concretization_minimizer(affine_bound, box):
    w = np.asarray(affine_bound.weights[0].array)
    lb, ub = np.asarray(box.lb.array), np.asarray(box.ub.array)
    return Array(np.where(w > 0.0, lb, ub))


def evaluate(fn, x):
    return float(np.asarray(fn(x).array).reshape(-1)[0])


def pgd_attack(fn, box, restarts=4, steps=25, seed=0):
    lb, ub = np.asarray(box.lb.array), np.asarray(box.ub.array)
    width = ub - lb
    rng = np.random.default_rng(seed)

    cg = make_graph(fn)(Array(0.5 * (lb + ub)))
    outvar = cg.outvars[0]

    def value_and_grad(x):
        primals = cg(Array(x))
        value = float(np.asarray(primals[outvar].array).reshape(-1)[0])
        g = _grad_backwards(cg, primals, [Array(1.0)])[0]
        return value, np.asarray(g.array)

    for r in range(restarts):
        x = 0.5 * (lb + ub) if r == 0 else rng.uniform(lb, ub)
        step = 0.25 * width
        for _ in range(steps):
            value, g = value_and_grad(x)
            if value < 0.0:
                return Array(x)
            x = np.clip(x - step * np.sign(g), lb, ub)
            step = step * 0.9
    return None