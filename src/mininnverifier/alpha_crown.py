# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.

from minijax import core
from minijax.compute_graph import make_graph
from minijax.core import relu, where
from minijax.nested_containers import map_structure
from minijax.eval import zeros, ones
from minijax.grad import grad

from .ibp import ibp, box_or_val, Box
from .lbp import pos_part, neg_part, get_in_bounds, lbp_inner, linear_lower_bound, AffineBound


def alpha_crown(fn, init_bounds=ibp, lr=0.001, iters=10):
    def bounds_fn(*args: Box | core.Value, **kwargs):
        # -lb on -fn is ub on fn
        def neg_fn(*args, **kwargs):
            return -fn(*args, **kwargs)

        args_ = map_structure(lambda a: a.lb if isinstance(a, Box) else a, args, is_leaf=box_or_val)
        cg = make_graph(fn)(*args_, **kwargs)
        cg_neg = make_graph(neg_fn)(*args_, **kwargs)

        var_bounds = init_bounds(cg)(*args, **kwargs)
        var_bounds_neg = init_bounds(cg_neg)(*args, **kwargs)

        lb = alpha_crown_optim(cg, var_bounds, lr=lr, iters=iters)
        lb_neg = alpha_crown_optim(cg_neg, var_bounds_neg, lr=lr, iters=iters)
        ub = AffineBound(tuple(-w for w in lb_neg.weights), -lb_neg.bias)
        return lb, ub

    return bounds_fn


def alpha_crown_optim(cg, var_bounds, lr=0.001, iters=10):
    def loss(params):
        affine_lb = linear_lower_bound(cg, var_bounds, params, crown_rules)
        return affine_lb.concrete(*get_in_bounds(cg.invars, var_bounds))

    p_grads = grad(loss)
    params = init_params(cg, var_bounds)
    if len(params) > 0:  # no params => no need to optimize
        for _ in range(iters):  # gradient *ascent* on alpha => maximize the lower bound
            gs = p_grads(params)[0]
            params = map_structure(lambda p, g: p + lr * g, params, gs)
            params = map_structure(lambda p: core.clip(p, 0.0, 1.0), params)

    return linear_lower_bound(cg, var_bounds, params, crown_rules)


def init_params(cg, var_bounds):
    params = {}
    for eqn in cg.equations:
        x = get_in_bounds(eqn.inputs, var_bounds)
        if eqn.primitive is relu and isinstance(x[0], Box):
            # init alpha with adaptive bound
            x_lb, x_ub = x[0].lb, x[0].ub
            alpha = where(-x_lb >= x_ub, zeros(x_lb.shape), ones(x_lb.shape))
            params[eqn.outvar] = alpha
    return params


def crown_relu(alpha, out_w, x):
    x_lb, x_ub = (x.lb, x.ub) if isinstance(x, Box) else (x, x)
    zero, one = zeros(x_lb.shape), ones(x_lb.shape)

    # mixed phase weights used when x_lb <= 0 <= x_ub
    upper_slope = x_ub / (x_ub - x_lb)
    if alpha is None:  # regular CROWN with adaptive lower slope
        lower_slope = where(-x_lb >= x_ub, zero, one)
    else:  # alpha-CROWN
        lower_slope = alpha
    upper_offset = -x_ub * x_lb / (x_ub - x_lb)
    # lower_offset is 0.0

    upper_slope = where(x_lb >= zero, one, where(x_ub <= zero, zero, upper_slope))
    lower_slope = where(x_lb >= zero, one, where(x_ub <= zero, zero, lower_slope))
    upper_offset = where((x_lb >= zero) | (x_ub <= zero), zero, upper_offset)

    in_w = lower_slope * pos_part(out_w) + upper_slope * neg_part(out_w)
    in_bias = lbp_inner(upper_offset, neg_part(out_w))  # lower bound bias
    return in_w, in_bias


crown_rules = {relu: crown_relu}
