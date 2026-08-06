# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.

from minijax import core
from minijax.core import relu
from minijax.nested_containers import map_structure
from minijax.eval import zeros
from minijax.grad import grad

from .lbp import get_in_bounds, linear_lower_bound
from .alpha_crown import crown_relu, init_params as init_alpha_params


def beta_crown_lb(cg, var_bounds, splits, lr=0.01, iters=100):
    params = init_params(cg, var_bounds)

    def apply_splits(params):
        # split = 1 => relu split so that input x >= 0 => Lagrange term has negative sign
        # split = -1 => relu split so that input x < 0 => Lagrange term has positive sign
        return {ov: (alpha, -splits[ov] * beta) for ov, (alpha, beta) in params.items()}

    def loss(params):
        params = apply_splits(params)
        affine_lb = linear_lower_bound(cg, var_bounds, params, beta_crown_rules)
        return affine_lb.concrete(*get_in_bounds(cg.invars, var_bounds))

    def project_params(node_params):
        alpha = core.clip(node_params[0], 0.0, 1.0)
        beta = core.maximum(node_params[1], 0.0)
        return (alpha, beta)

    p_grads = grad(loss)
    if len(params) > 0:
        for _ in range(iters):
            gs = p_grads(params)[0]
            params = map_structure(lambda p, g: p + lr * g, params, gs)
            params = {ov: project_params(node) for ov, node in params.items()}

    return linear_lower_bound(cg, var_bounds, apply_splits(params), beta_crown_rules)


def init_params(cg, var_bounds):
    params = {}
    for outvar, alpha in init_alpha_params(cg, var_bounds).items():
        beta = zeros(outvar.shape)
        params[outvar] = (alpha, beta)
    return params


def beta_crown_relu(params, out_w, x):
    alpha, beta = params
    in_w, in_bias = crown_relu(alpha, out_w, x)
    return in_w + beta, in_bias


beta_crown_rules = {relu: beta_crown_relu}
