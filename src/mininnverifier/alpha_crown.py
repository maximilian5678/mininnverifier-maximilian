# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.

from minijax import core
from minijax.compute_graph import make_graph
from minijax.core import relu, where, leaky_relu, elu, gelu, exp, normalcdf, sqrt, reciprocal, square, mul
from minijax.nested_containers import map_structure
from minijax.eval import zeros, ones, Array
from minijax.grad import grad, unbroadcast

from .fuse import fuse_activations
from .ibp import ibp, box_or_val, Box
from .lbp import lbp_linear, pos_part, neg_part, get_in_bounds, lbp_inner, linear_lower_bound, AffineBound

import numpy as np
import scipy.special as special


def alpha_crown(fn, init_bounds=ibp, lr=0.1, iters=100):
    def bounds_fn(*args: Box | core.Value, **kwargs):
        # -lb on -fn is ub on fn
        def neg_fn(*args, **kwargs):
            return -fn(*args, **kwargs)

        args_ = map_structure(lambda a: a.lb if isinstance(a, Box) else a, args, is_leaf=box_or_val)
        cg = fuse_activations(make_graph(fn)(*args_, **kwargs))
        cg_neg = fuse_activations(make_graph(neg_fn)(*args_, **kwargs))

        var_bounds = init_bounds(cg)(*args, **kwargs)
        var_bounds_neg = init_bounds(cg_neg)(*args, **kwargs)

        lb = bounded_lower_bound(cg, var_bounds, lr=lr, iters=iters)
        lb_neg = bounded_lower_bound(cg_neg, var_bounds_neg, lr=lr, iters=iters)
        ub = AffineBound(tuple(-w for w in lb_neg.weights), -lb_neg.bias)
        return lb, ub

    return bounds_fn


def _finite(value):
    return bool(np.all(np.isfinite(np.asarray(value.array))))


def _ibp_bound(cg, var_bounds):
    """The IBP lower bound as a constant (zero-weight) affine bound."""
    out_lb = var_bounds[cg.outvars[0]].lb
    return AffineBound(tuple(zeros(iv.shape) for iv in cg.invars), out_lb)


def bounded_lower_bound(cg, var_bounds, lr=0.1, iters=100):
    """alpha-CROWN, guarded by IBP.

    CROWN's linear relaxations need finite intermediate bounds: an overflowing
    exp/mul chain makes the secants and tangents degenerate and the back-
    substituted result can silently turn into a finite but unsound number. In
    that case, and whenever the affine bound concretises to something worse
    than IBP, we return the IBP bound instead. That is never looser and always
    sound.
    """
    ibp_ab = _ibp_bound(cg, var_bounds)
    for b in var_bounds.values():
        if not (_finite(b.lb) and _finite(b.ub)):
            return ibp_ab

    affine = alpha_crown_optim(cg, var_bounds, lr=lr, iters=iters)
    if not (all(_finite(w) for w in affine.weights) and _finite(affine.bias)):
        return ibp_ab

    in_bounds = get_in_bounds(cg.invars, var_bounds)
    crown_conc = np.asarray(affine.concrete(*in_bounds).array)
    ibp_conc = np.asarray(ibp_ab.bias.array)
    if not np.all(np.isfinite(crown_conc)) or np.all(crown_conc <= ibp_conc):
        return ibp_ab
    return affine


def alpha_crown_optim(cg, var_bounds, lr=0.1, iters=100):
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


def _elu_np(t):
    return np.where(t >= 0.0, t, np.exp(np.minimum(t, 0.0)) - 1.0)


def _elu_deriv_np(t):
    return np.where(t >= 0.0, 1.0, np.exp(np.minimum(t, 0.0)))


def _ncdf_np(t):
    return 0.5 * (1.0 + special.erf(t / _SQRT2))


def _gelu_np(t):
    return t * _ncdf_np(t)


def _gelu_deriv_np(t):
    return _ncdf_np(t) + t * np.exp(-0.5 * t * t) / np.sqrt(2.0 * np.pi)


def _tangent_alpha(f, df, lb, ub, n=65):
    """Per-element tangent point minimising the worst-case gap to f on [lb, ub].

    For a convex (or concave) f the gap between f and a tangent line is maximal
    at an endpoint, so a grid search over the tangent point is enough. The
    midpoint tangent that CROWN uses by default is a poor choice on asymmetric
    boxes; this is the warm start that alpha then refines.
    """
    grid = np.linspace(0.0, 1.0, n)
    a = grid.reshape((n,) + (1,) * np.ndim(lb))
    t = lb + a * (ub - lb)
    slope = df(t)
    offset = f(t) - slope * t
    gap = np.maximum(
        np.abs(f(lb) - (slope * lb + offset)), np.abs(f(ub) - (slope * ub + offset))
    )
    return grid[np.argmin(gap, axis=0)]


_SMOOTH_ALPHA_INIT = {elu: (_elu_np, _elu_deriv_np), gelu: (_gelu_np, _gelu_deriv_np)}


def init_params(cg, var_bounds):
    params = {}
    for eqn in cg.equations:
        x = get_in_bounds(eqn.inputs, var_bounds)
        if eqn.primitive is relu and isinstance(x[0], Box):
            # init alpha with adaptive bound
            x_lb, x_ub = x[0].lb, x[0].ub
            alpha = where(-x_lb >= x_ub, zeros(x_lb.shape), ones(x_lb.shape))
            params[eqn.outvar] = alpha
        elif eqn.primitive in _SMOOTH_ALPHA_INIT and isinstance(x[0], Box):
            f, df = _SMOOTH_ALPHA_INIT[eqn.primitive]
            lb_np, ub_np = np.asarray(x[0].lb.array), np.asarray(x[0].ub.array)
            params[eqn.outvar] = Array(_tangent_alpha(f, df, lb_np, ub_np))
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

_SQRT2 = np.sqrt(2.0)
_GELU_ARGMIN = -0.7517913647329811 # argmin of gelu (gelu'(x) = 0)
_GELU_MIN = -0.1699712074798982 # gelu(_GELU_ARGMIN)


def _in_bounds(x):
    return (x.lb, x.ub) if isinstance(x, Box) else (x, x)


def _backsub(lower_slope, lower_offset, upper_slope, upper_offset, out_w):
    """CROWN lower-bound backward transform: lower line where out_w >= 0,
    upper line where out_w < 0."""
    in_w = lower_slope * pos_part(out_w) + upper_slope * neg_part(out_w)
    in_bias = lbp_inner(lower_offset, pos_part(out_w)) + lbp_inner(upper_offset, neg_part(out_w))
    return in_w, in_bias


def crown_leaky_relu(alpha, out_w, x, *, slope):
    """leaky_relu is convex for 0 < slope < 1: upper = chord, lower = tangent."""
    x_lb, x_ub = _in_bounds(x)
    zero, one = zeros(x_lb.shape), ones(x_lb.shape)
    s = slope * one
    denom = x_ub - x_lb
    safe = where(denom <= 0.0, one, denom)
    upper_slope = (x_ub - slope * x_lb) / safe
    upper_offset = -(one - s) * x_ub * x_lb / safe
    if alpha is None:
        lower_slope = where(-x_lb >= x_ub, s, one)
    else:
        lower_slope = s + alpha * (one - s)
    pos, neg = x_lb >= zero, x_ub <= zero
    upper_slope = where(pos, one, where(neg, s, upper_slope))
    lower_slope = where(pos, one, where(neg, s, lower_slope))
    upper_offset = where(pos | neg, zero, upper_offset)
    return _backsub(lower_slope, zero, upper_slope, upper_offset, out_w)


def _elu(t):
    return where(t >= 0.0, t, exp(t) - 1.0)


def _elu_deriv(t):
    return where(t >= 0.0, ones(t.shape), exp(t))


def crown_elu(alpha, out_w, x):
    """elu is convex and smooth: upper = secant over [lb, ub], lower = tangent."""
    x_lb, x_ub = _in_bounds(x)
    one = ones(x_lb.shape)
    denom = x_ub - x_lb
    same = denom <= 0.0
    safe = where(same, one, denom)
    f_lb, f_ub = _elu(x_lb), _elu(x_ub)
    upper_slope = where(same, _elu_deriv(x_lb), (f_ub - f_lb) / safe)
    upper_offset = f_ub - upper_slope * x_ub
    t = 0.5 * (x_lb + x_ub) if alpha is None else x_lb + alpha * denom
    lower_slope = _elu_deriv(t)
    lower_offset = _elu(t) - lower_slope * t
    return _backsub(lower_slope, lower_offset, upper_slope, upper_offset, out_w)


def _gelu(t):
    return t * normalcdf(t)


def _gelu_pdf(t):
    return exp(-0.5 * t * t) / np.sqrt(2.0 * np.pi)


def _gelu_deriv(t):
    return normalcdf(t) + t * _gelu_pdf(t)


def crown_gelu(alpha, out_w, x):
    x_lb, x_ub = _in_bounds(x)
    lb, ub = np.asarray(x_lb.array), np.asarray(x_ub.array)
    zero, one = zeros(x_lb.shape), ones(x_lb.shape)
    denom = x_ub - x_lb
    same = denom <= 0.0
    safe = where(same, one, denom)
    f_lb, f_ub = _gelu(x_lb), _gelu(x_ub)
    secant = where(same, _gelu_deriv(x_lb), (f_ub - f_lb) / safe)
    secant_offset = f_lb - secant * x_lb
    t = 0.5 * (x_lb + x_ub) if alpha is None else x_lb + alpha * denom
    tangent = _gelu_deriv(t)
    tangent_offset = _gelu(t) - tangent * t

    convex = (lb >= -_SQRT2) & (ub <= _SQRT2)
    concave = (lb >= _SQRT2) | (ub <= -_SQRT2)
    mixed = ~(convex | concave)

    lower_slope = where(convex, tangent, where(concave, secant, zero))
    lower_offset = where(convex, tangent_offset, where(concave, secant_offset, zero))
    upper_slope = where(convex, secant, where(concave, tangent, zero))
    upper_offset = where(convex, secant_offset, where(concave, tangent_offset, zero))

    g_lb = np.asarray(_gelu(x_lb).array)
    g_ub = np.asarray(_gelu(x_ub).array)
    contains_min = (lb <= _GELU_ARGMIN) & (ub >= _GELU_ARGMIN)
    const_min = np.where(contains_min, _GELU_MIN, np.minimum(g_lb, g_ub))
    const_max = np.maximum(g_lb, g_ub)
    lower_offset = where(mixed, Array(const_min), lower_offset)
    upper_offset = where(mixed, Array(const_max), upper_offset)
    return _backsub(lower_slope, lower_offset, upper_slope, upper_offset, out_w)

def _convex_coeffs(f, df, x_lb, x_ub):
    """Convex f: upper = secant over [lb, ub], lower = tangent at the midpoint."""
    one = ones(x_lb.shape)
    denom = x_ub - x_lb
    same = denom <= 0.0
    safe = where(same, one, denom)
    f_lb, f_ub = f(x_lb), f(x_ub)
    upper_slope = where(same, df(x_lb), (f_ub - f_lb) / safe)
    upper_offset = f_ub - upper_slope * x_ub
    t = 0.5 * (x_lb + x_ub)
    lower_slope = df(t)
    lower_offset = f(t) - lower_slope * t
    return lower_slope, lower_offset, upper_slope, upper_offset


def _concave_coeffs(f, df, x_lb, x_ub):
    """Concave f: mirror of the convex case (upper = tangent, lower = secant)."""
    ls, lo, us, uo = _convex_coeffs(lambda z: -f(z), lambda z: -df(z), x_lb, x_ub)
    return -us, -uo, -ls, -lo


def crown_exp(alpha, out_w, x):
    x_lb, x_ub = _in_bounds(x)
    return _backsub(*_convex_coeffs(exp, exp, x_lb, x_ub), out_w)


def crown_square(alpha, out_w, x):
    x_lb, x_ub = _in_bounds(x)
    return _backsub(*_convex_coeffs(square, lambda z: 2.0 * z, x_lb, x_ub), out_w)


def crown_sqrt(alpha, out_w, x):
    x_lb, x_ub = _in_bounds(x)
    return _backsub(*_concave_coeffs(sqrt, lambda z: 0.5 / sqrt(z), x_lb, x_ub), out_w)


def crown_reciprocal(alpha, out_w, x):
    x_lb, x_ub = _in_bounds(x)
    return _backsub(*_convex_coeffs(reciprocal, lambda z: -reciprocal(z * z), x_lb, x_ub), out_w)


def _normal_pdf(z):
    return exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def crown_normalcdf(alpha, out_w, x):
    """Phi is S-shaped: convex on x <= 0, concave on x >= 0, inflection at 0."""
    x_lb, x_ub = _in_bounds(x)
    lb, ub = np.asarray(x_lb.array), np.asarray(x_ub.array)
    zero = zeros(x_lb.shape)
    convex = ub <= 0.0
    concave = lb >= 0.0
    mixed = ~(convex | concave)
    cvx = _convex_coeffs(normalcdf, _normal_pdf, x_lb, x_ub)
    ccv = _concave_coeffs(normalcdf, _normal_pdf, x_lb, x_ub)
    ls = where(convex, cvx[0], where(concave, ccv[0], zero))
    lo = where(convex, cvx[1], where(concave, ccv[1], zero))
    us = where(convex, cvx[2], where(concave, ccv[2], zero))
    uo = where(convex, cvx[3], where(concave, ccv[3], zero))
    lo = where(mixed, normalcdf(x_lb), lo)
    uo = where(mixed, normalcdf(x_ub), uo)
    return _backsub(ls, lo, us, uo, out_w)


def crown_mul(alpha, out_w, x, y):
    """Bilinear z = x*y via McCormick envelopes (per-element tighter plane at the
    box centre). A constant operand is a degenerate box => the bounds are exact."""
    x_lb, x_ub = _in_bounds(x)
    y_lb, y_ub = _in_bounds(y)
    x_mid, y_mid = 0.5 * (x_lb + x_ub), 0.5 * (y_lb + y_ub)

    a_x, a_y, a_c = y_lb, x_lb, -x_lb * y_lb
    b_x, b_y, b_c = y_ub, x_ub, -x_ub * y_ub
    pick_a = (a_x * x_mid + a_y * y_mid + a_c) >= (b_x * x_mid + b_y * y_mid + b_c)
    lx = where(pick_a, a_x, b_x); ly = where(pick_a, a_y, b_y); lc = where(pick_a, a_c, b_c)

    c_x, c_y, c_c = y_lb, x_ub, -x_ub * y_lb
    d_x, d_y, d_c = y_ub, x_lb, -x_lb * y_ub
    pick_c = (c_x * x_mid + c_y * y_mid + c_c) <= (d_x * x_mid + d_y * y_mid + d_c)
    ux = where(pick_c, c_x, d_x); uy = where(pick_c, c_y, d_y); uc = where(pick_c, c_c, d_c)
    in_wx = lx * pos_part(out_w) + ux * neg_part(out_w)
    in_wy = ly * pos_part(out_w) + uy * neg_part(out_w)
    in_bias = lbp_inner(lc, pos_part(out_w)) + lbp_inner(uc, neg_part(out_w))
    return (in_wx, in_wy), in_bias


def crown_where(alpha, out_w, cond, x, y):
    """Sound fallback for a where that survived activation fusion.

    Exact on the branches that are decided by the condition bounds, constant
    enclosure on the undecided ones (a linear relaxation of an undecided
    where is not possible without knowing how x, y and cond are related).
    """
    c_lb, c_ub = _in_bounds(cond)
    x_lb, x_ub = _in_bounds(x)
    y_lb, y_ub = _in_bounds(y)
    cl, cu = np.asarray(c_lb.array), np.asarray(c_ub.array)
    sure_true = (cl > 0.0) | (cu < 0.0)  # where treats any non-zero as true
    sure_false = (cl == 0.0) & (cu == 0.0)
    mixed = ~(sure_true | sure_false)

    lo = where(x_lb <= y_lb, x_lb, y_lb)
    hi = where(x_ub >= y_ub, x_ub, y_ub)
    w_true = Array(sure_true.astype(np.float64)) * out_w
    w_false = Array(sure_false.astype(np.float64)) * out_w
    w_mixed = Array(mixed.astype(np.float64)) * out_w
    in_bias = lbp_inner(lo, pos_part(w_mixed)) + lbp_inner(hi, neg_part(w_mixed))
    return (zeros(c_lb.shape), w_true, w_false), in_bias


def _mccormick_planes(xl, xu, yl, yu):
    """Element-wise McCormick envelopes of the product x*y over [xl,xu]x[yl,yu].

    Each envelope is a pair of planes; we keep the one that is tighter at the
    centre of the box, which is what CROWN back-substitutes.
    """
    xm, ym = 0.5 * (xl + xu), 0.5 * (yl + yu)
    a, b = (yl, xl, -xl * yl), (yu, xu, -xu * yu)
    take_a = (a[0] * xm + a[1] * ym + a[2]) >= (b[0] * xm + b[1] * ym + b[2])
    low = tuple(np.where(take_a, p, q) for p, q in zip(a, b))
    c, d = (yl, xu, -xu * yl), (yu, xl, -xl * yu)
    take_c = (c[0] * xm + c[1] * ym + c[2]) <= (d[0] * xm + d[1] * ym + d[2])
    high = tuple(np.where(take_c, p, q) for p, q in zip(c, d))
    return low, high


def _align_dot(xl, yl, out_w):
    """Line the two dot operands and the output weight up on a common product
    tensor, so the contraction becomes an element-wise product plus a sum."""
    xd, yd = xl.ndim, yl.ndim
    if xd == 0 or yd == 0:  # scalar factor: dot degenerates to a product
        return xl.shape, yl.shape, out_w
    if yd == 1:  # (..., J) @ (J,) -> (...)
        return xl.shape, yl.shape, core.expand_dims(out_w, axes=(-1,))
    if xd == 1:  # (J,) @ (..., J, K) -> (..., K)
        return xl.shape + (1,), yl.shape, core.expand_dims(out_w, axes=(-2,))
    # (..., I, J) @ (..., J, K) -> (..., I, K)
    return xl.shape + (1,), yl.shape[:-2] + (1,) + yl.shape[-2:], core.expand_dims(
        out_w, axes=(-2,)
    )


def crown_dot(alpha, out_w, x, y):
    """dot with one constant operand stays exact; box x box uses McCormick."""
    if not (isinstance(x, Box) and isinstance(y, Box)):
        return lbp_linear(core.dot, out_w, x, y), Array(0.0)

    xl, xu = np.asarray(x.lb.array), np.asarray(x.ub.array)
    yl, yu = np.asarray(y.lb.array), np.asarray(y.ub.array)
    x_shape, y_shape, w = _align_dot(xl, yl, out_w)
    xl_a, xu_a = xl.reshape(x_shape), xu.reshape(x_shape)
    yl_a, yu_a = yl.reshape(y_shape), yu.reshape(y_shape)

    (lx, ly, lc), (ux, uy, uc) = _mccormick_planes(xl_a, xu_a, yl_a, yu_a)
    pos, neg = pos_part(w), neg_part(w)
    wx = Array(lx) * pos + Array(ux) * neg
    wy = Array(ly) * pos + Array(uy) * neg
    in_wx = core.reshape(unbroadcast(wx, x_shape), new_shape=xl.shape)
    in_wy = core.reshape(unbroadcast(wy, y_shape), new_shape=yl.shape)
    in_bias = lbp_inner(Array(lc), pos) + lbp_inner(Array(uc), neg)
    return (in_wx, in_wy), in_bias

crown_rules = {
    relu: crown_relu,
    where: crown_where,
    leaky_relu: crown_leaky_relu,
    elu: crown_elu,
    gelu: crown_gelu,
    exp: crown_exp,
    square: crown_square,
    sqrt: crown_sqrt,
    reciprocal: crown_reciprocal,
    normalcdf: crown_normalcdf,
    mul: crown_mul,
}