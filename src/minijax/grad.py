# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
from . import core
from .compute_graph import make_graph
from .eval import Array, zeros
from .nested_containers import flatten, unflatten
import scipy.special as special
import numpy as np


def grad(fn):
    v_and_g_fn = value_and_grad(fn)
    return lambda *args, **kwargs: v_and_g_fn(*args, **kwargs)[1]


def value_and_grad(fn):
    def v_and_g_fn(*primals, **kwargs):
        return vjp(fn, return_primals=True)(primals, Array(1.0), **kwargs)

    return v_and_g_fn


def vjp(fn, return_primals=False):
    def vjp_fn(in_primals, out_tangents, **kwargs):
        cg = make_graph(fn)(*in_primals, **kwargs)

        in_primals, in_structure = flatten(in_primals)
        out_tangents, out_structure = flatten(out_tangents)

        primals = _forward(cg, in_primals)
        in_tangents = _backwards(cg, primals, out_tangents)

        in_tangents = unflatten(in_structure, in_tangents)
        if return_primals:
            out_primals = unflatten(out_structure, [primals[v] for v in cg.outvars])
            return out_primals, in_tangents
        else:
            return in_tangents

    return vjp_fn


def _forward(cg, primals):
    primals = {iv: p for iv, p in zip(cg.invars, primals, strict=True)}
    for eqn in cg.equations:
        args = [v.value if v.is_const else primals[v] for v in eqn.inputs]
        out = eqn.primitive(*args, **eqn.options)
        primals[eqn.outvar] = out
    return primals


def _backwards(cg, primals, out_tangents):
    tangents = {ov: t for ov, t in zip(cg.outvars, out_tangents)}

    def update(var, tangent):
        if var in tangents:
            tangents[var] = core.add(tangents[var], unbroadcast(tangent, primals[var]))
        elif not var.is_const:
            tangents[var] = unbroadcast(tangent, primals[var])

    for eqn in reversed(cg.equations):
        in_primals = [a.value if a.is_const else primals[a] for a in eqn.inputs]
        out_tangent = tangents[eqn.outvar] if eqn.outvar in tangents else zeros(eqn.outvar.shape)
        out_primal = primals[eqn.outvar]

        in_tangents = vjp_rules[eqn.primitive](out_tangent, out_primal, *in_primals, **eqn.options)

        in_tangents = (in_tangents,) if not isinstance(in_tangents, tuple) else in_tangents
        for v, t in zip(eqn.inputs, in_tangents, strict=True):
            update(v, t)

    return [tangents.get(iv, zeros(iv.shape)) for iv in cg.invars]


def unbroadcast(tangent, primal):
    added = [i for i in range(len(tangent.shape) - len(primal.shape))]
    tangent = core.reduce_sum(tangent, tuple(added))
    # tangent and primal now have the same number of axes
    expanded = [i for i, (t, p) in enumerate(zip(tangent.shape, primal.shape)) if t != p]
    return core.reduce_sum(tangent, tuple(expanded), keepaxes=True)


def broadcast_to(tangent, primal):
    # exploiting that add does broadcasting
    return core.add(tangent, zeros(primal.shape))


def vjp_dot(t, _, x, y):
    if y.ndim == 1:
        dx = core.expand_dims(t, axes=-1) @ core.expand_dims(y, axes=0)
    else:
        dx = t @ core.transpose(y)

    if x.ndim == 1:
        dy = core.expand_dims(x, axes=-1) @ core.expand_dims(t, axes=0)
    else:
        dy = core.transpose(x) @ t
    return dx, dy


def vjp_where(tangent, out, cond, true_val, false_val):
    zero = zeros(cond.shape)
    return (zero, core.where(cond, tangent, zero), core.where(cond, zero, tangent))

def np_unpad(t, config, axes, original_shape):
    l, r, m = config
    ndim = t.ndim
    axes = [a % ndim for a in axes]
    
    result = t.array
    
    # left/right cutoff
    idx = [slice(None)] * ndim
    for ax in axes:
        s = result.shape[ax]
        idx[ax] = slice(l, s - r if r > 0 else None)
    result = result[tuple(idx)]
    
    # interior padding cutoff
    if m > 0:
        for ax in axes:
            idx = [slice(None)] * ndim
            idx[ax] = slice(None, None, m + 1)
            result = result[tuple(idx)]
    
    return broadcast_to(Array(result), Array(np.zeros(original_shape)))

def vjp_conv(t, _, x, k, stride):
    """
    t: (N, Cout, H_out, W_out) out_tangent (Array)
    x: (N, Cin, H, W) primal input (Array)
    k: (Cout, Cin, kH, kW) primal kernel (Array)
    """
    t_np = t.array
    x_np = x.array
    k_np = k.array

    N, _, H, W = x_np.shape
    Cout, _, kH, kW = k_np.shape
    _, _, H_out, W_out = t_np.shape

    x_windows = np.lib.stride_tricks.sliding_window_view(x_np, (kH, kW), axis=(2, 3))
    x_windows = x_windows[:, :, ::stride, ::stride, :, :]
    dk = np.einsum("nchwij,nohw->ocij", x_windows, t_np)

    if stride > 1:
        t_dil = np.zeros(
            (N, Cout, (H_out - 1) * stride + 1, (W_out - 1) * stride + 1),
            dtype=t_np.dtype,
        )
        t_dil[:, :, ::stride, ::stride] = t_np
    else:
        t_dil = t_np

    pad_top = kH - 1
    pad_left = kW - 1
    pad_bottom = H - (H_out - 1) * stride - 1 + (kH - 1) - pad_top
    pad_right = W - (W_out - 1) * stride - 1 + (kW - 1) - pad_left
    pad_bottom = H + kH - 2 - (H_out - 1) * stride - pad_top
    pad_right = W + kW - 2 - (W_out - 1) * stride - pad_left

    t_padded = np.pad(
        t_dil,
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
    )

    k_flipped = k_np[:, :, ::-1, ::-1]
    t_windows = np.lib.stride_tricks.sliding_window_view(t_padded, (kH, kW), axis=(2, 3))
    dx = np.einsum("nohwij,ocij->nchw", t_windows, k_flipped)
    return Array(dx), Array(dk)

def vjp_avgpool(t, _, x, window_size, stride):
    """
    t: out_tangent, shape = output shape (Array)
    x: primal input (Array)
    window_size
    stride
    """
    t_np = t.array
    x_np = x.array
    ndim = x_np.ndim

    window_volume = int(np.prod(window_size))
    dx = np.zeros_like(x_np)
    t_scaled = t_np / window_volume

    for offset in np.ndindex(*window_size):
        slicer = tuple(
            slice(offset[i], offset[i] + stride[i] * t_np.shape[i], stride[i])
            for i in range(ndim)
        )
        dx[slicer] += t_scaled

    return Array(dx)


vjp_rules = {
    core.expand_dims: lambda t, _, x, axes: core.reduce_sum(t, axes),
    core.moveaxis: lambda t, _, __, source, destination: core.moveaxis(t, destination, source),
    core.reshape: lambda t, _, x, new_shape: core.reshape(t, x.shape),
    core.neg: lambda t, *_: -t,
    core.add: lambda t, *_: (t, t),
    core.reduce_sum: lambda t, _, x, axes: broadcast_to(core.expand_dims(t, axes), x),
    core.dot: vjp_dot,
    core.mul: lambda t, _, x, y: (t * y, x * t),
    core.reciprocal: lambda t, _, x: -core.reciprocal(core.square(x)) * t,
    core.relu: lambda t, out, x: core.where(out, t, Array(0)),
    core.square: lambda t, _, x: t * Array(2) * x,
    core.sqrt: lambda t, _, x: t / (Array(2) * core.sqrt(x)),
    core.exp: lambda t, out, x: t * out,
    core.log: lambda t, _, x: t / x,
    core.where: vjp_where,
    # Activation Functions
    core.leaky_relu: lambda t, out, x, slope: core.where(core.ge(x, Array(0)), t, Array(slope) * t),
    core.elu: lambda t, out, x: core.where(core.ge(x, Array(0)), t, (out + Array(1)) * t),
    core.gelu: lambda t, _, x: t * Array(
        0.5 * (1 + special.erf(x.array / np.sqrt(2))) 
        + x.array * np.exp(-0.5 * x.array**2) / np.sqrt(2 * np.pi)
    ),
    core.normalcdf: lambda t, _, x: t * Array(np.exp(-0.5 * x.array**2) / np.sqrt(2 * np.pi)),
    core.pad: lambda t, _, x, config, axes, value: np_unpad(t, config, axes, x.shape),
    core.conv: vjp_conv,
    core.avgpool: vjp_avgpool,
}
