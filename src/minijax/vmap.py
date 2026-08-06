# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
from typing import Any

from . import core
from .eval import Array, broadcast_to
from .nested_containers import map_structure


def vmap(fn, in_axis: int | None | Any = 0, out_axis: int | None | Any = 0):
    def vmapped_fn(*args, **kwargs):
        with core.new_interpreter(VmapInterpreter()) as vmapper:
            vmap_vals = map_structure(lambda v, ax: Vmapped(vmapper, v, ax), args, in_axis)
            results = fn(*vmap_vals, **kwargs)

        results = map_structure(lambda vval, axis: vval.move_batch_axis(axis), results, out_axis)
        return map_structure(lambda vval: vval.base_value, results)

    return vmapped_fn


class Vmapped(core.Value):
    def __init__(self, interpreter, value, batch_axis: int | None):
        shape = value.shape
        if batch_axis is not None:
            batch_axis = batch_axis if batch_axis >= 0 else len(shape) + batch_axis
            shape = tuple(s for i, s in enumerate(shape) if i != batch_axis)
        super().__init__(interpreter, shape)
        self.interpreter = interpreter
        self.base_value = value
        self.batch_axis = batch_axis

    @property
    def full_shape(self):
        return self.base_value.shape

    def move_batch_axis(self, new_axis):
        new_axis = new_axis if new_axis >= 0 else len(self.full_shape) + new_axis
        if self.batch_axis is None or self.batch_axis == new_axis:
            return self

        if new_axis == len(self.base_value.shape):
            new_base = core.expand_dims(self.base_value, -1)
        else:
            new_base = core.moveaxis(self.base_value, self.batch_axis, new_axis)
        return Vmapped(self.interpreter, new_base, new_axis)


class VmapInterpreter(core.Interpreter):
    def wrap(self, value):
        if isinstance(value, Vmapped):
            return value
        if not isinstance(value, core.Value):
            value = Array(value)
        return Vmapped(self, value, None)

    def process(self, primitive, values, options):
        vvals = [vval.move_batch_axis(0) for vval in values]
        base_vals = [vval.base_value for vval in vvals]
        if primitive is core.dot:
            return vmap_dot(*vvals, **options)
        elif primitive is core.concat_two:
            return vmap_concat_two(*vvals, **options)
        elif primitive in vmap_rules:
            result = vmap_rules[primitive](*base_vals, **options)
        else:
            result = primitive(*base_vals, **options)
        return Vmapped(self, result, batch_axis=0)


def vmap_dot(x: Vmapped, y: Vmapped):
    if len(y.full_shape) <= 2 and y.batch_axis is not None:
        y = y.move_batch_axis(-1)
        out = core.dot(x.base_value, y.base_value)
        return Vmapped(x.interpreter, out, batch_axis=-1)

    out = core.dot(x.base_value, y.base_value)
    return Vmapped(x.interpreter, out, batch_axis=0)


def vmap_concat_two(x: Vmapped, y: Vmapped, axis):
    if x.batch_axis is None and y.batch_axis is None:
        out = core.concat_two(x.base_value, y.base_value, axis=axis)
        return Vmapped(x.interpreter, out, batch_axis=None)

    batch_size = x.base_value.shape[0] if x.batch_axis is not None else y.base_value.shape[0]
    x_base = _ensure_batched(x, batch_size)
    y_base = _ensure_batched(y, batch_size)
    out = core.concat_two(x_base, y_base, axis=_shift(axis))
    return Vmapped(x.interpreter, out, batch_axis=0)


def _ensure_batched(value: Vmapped, batch_size):
    # Already batched (axis 0) => use as-is; otherwise broadcast in a leading batch axis.
    if value.batch_axis is not None:
        return value.base_value
    return broadcast_to(value.base_value, (batch_size,) + tuple(value.base_value.shape))


def _shift(index):
    return index + 1 if index >= 0 else index


vmap_rules = {
    core.expand_dims: lambda x, axes: core.expand_dims(x, [_shift(ax) for ax in axes]),
    core.moveaxis: lambda x, **axes: core.moveaxis(x, **{k: _shift(v) for k, v in axes.items()}),
    core.reshape: lambda x, new_shape: core.reshape(x, x.shape[:1] + new_shape),
    core.reduce_sum: lambda x, axes: core.reduce_sum(x, [_shift(ax) for ax in axes]),
    core.head: lambda x, axis, index: core.head(x, _shift(axis), index),
    core.tail: lambda x, axis, index: core.tail(x, _shift(axis), index),
}
