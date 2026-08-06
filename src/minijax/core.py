# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass


class Value(ABC):
    def __init__(self, interpreter, shape) -> None:
        self.interpreter = interpreter
        self.shape = shape

    @property
    def ndim(self):
        """Return the number of dimensions of this value."""
        return len(self.shape)

    def __neg__(self):
        return neg(self)

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __matmul__(self, other):
        return dot(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __rsub__(self, other):
        return sub(other, self)

    def __rmul__(self, other):
        return mul(other, self)

    def __rtruediv__(self, other):
        return div(other, self)

    def __rmatmul__(self, other):
        return dot(other, self)

    def __ge__(self, other):
        return greater_equal(self, other)

    def __le__(self, other):
        return less_equal(self, other)

    def __gt__(self, other):
        return greater_than(self, other)

    def __lt__(self, other):
        return less_than(self, other)

    def __eq__(self, other):
        return equals(self, other)

    def __ne__(self, other):
        return elementwise_not(equals(self, other))

    def __invert__(self):
        return elementwise_not(self)

    def __and__(self, other):
        return elementwise_and(self, other)

    def __or__(self, other):
        return elementwise_or(self, other)

    def __getitem__(self, i):
        return getitem(self, i)


class Interpreter[V: Value](ABC):
    @abstractmethod
    def wrap(self, value: Value) -> V:
        raise NotImplementedError()

    @abstractmethod
    def process(self, primitive, values: list[V], options: dict) -> V:
        raise NotImplementedError()


_interpreter_stack: list[Interpreter] = []


@contextmanager
def new_interpreter(interpreter: Interpreter):
    _interpreter_stack.append(interpreter)
    try:
        yield interpreter
    finally:
        _interpreter_stack.pop()


def top_interpreter(values: Iterable[Value]):
    owners = {val.interpreter for val in values if isinstance(val, Value)}
    for interpreter in reversed(_interpreter_stack):
        if interpreter in owners:
            return interpreter
    from .eval import EvalInterpreter  # cannot import at top level (circular import)

    return EvalInterpreter()


@dataclass(frozen=True)
class Primitive:
    name: str
    num_args: int
    keyword_args: tuple[str, ...] = ()

    def __call__(self, *values, **options):
        assert len(values) >= self.num_args, "Wrong number of arguments."
        options |= dict(zip(self.keyword_args, values[self.num_args :], strict=False))
        values = values[: self.num_args]
        assert set(options.keys()) == set(self.keyword_args), "Wrong keyword arguments."
        interpreter = top_interpreter(values)
        values = [interpreter.wrap(val) for val in values]
        return interpreter.process(self, values, options)


class ReduceSumPrimitive(Primitive):
    def __init__(self):
        super().__init__("reduce_sum", 1, ("axes",))

    def __call__(self, x, axes: int | Sequence[int] | None = None, keepaxes: bool = False):
        if axes is None:
            axes = tuple(range(len(x.shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        res = super().__call__(x, axes=axes)
        return expand_dims(res, axes) if keepaxes else res


neg = Primitive("neg", 1)
add = Primitive("add", 2)
dot = Primitive("dot", 2)
mul = Primitive("mul", 2)
reciprocal = Primitive("reciprocal", 1)
relu = Primitive("relu", 1)
square = Primitive("square", 1)
sqrt = Primitive("sqrt", 1)
exp = Primitive("exp", 1)
log = Primitive("log", 1)
where = Primitive("where", 3)
expand_dims = Primitive("expand_dims", 1, ("axes",))
moveaxis = Primitive("moveaxis", 1, ("source", "destination"))
reshape = Primitive("reshape", 1, ("new_shape",))
reduce_sum = ReduceSumPrimitive()

# Primitive Activation Functions
leaky_relu = Primitive("leaky_relu", 1, ("slope",))
elu = Primitive("elu", 1)
gelu = Primitive("gelu", 1)
normalcdf = Primitive("normalcdf", 1)
ge = Primitive("ge", 2) # greater than or equal to
pad = Primitive("pad", 1, ("config", "axes", "value",))
conv = Primitive("conv", 2, ("stride",))
avgpool = Primitive("avgpool", 1, ("window_size", "stride"))
sumpool = Primitive("sumpool", 1, ("window_size", "stride"))
greater_equal = Primitive("greater_equal", 2)
less_equal = Primitive("less_equal", 2)
elementwise_not = Primitive("not", 1)
elementwise_and = Primitive("and", 2)

concat_two = Primitive("concat_two", 2, ("axis",))
head = Primitive("head", 1, ("axis", "index"))
tail = Primitive("tail", 1, ("axis", "index"))


def sub(x, y):
    return add(x, neg(y))


def div(x, y):
    return mul(x, reciprocal(y))


def abs(x):
    return add(relu(x), relu(neg(x)))


def maximum(x, y):
    return add(y, relu(x - y))


def minimum(x, y):
    return sub(y, relu(y - x))


def clip(x, lower, upper):
    return minimum(maximum(x, lower), upper)


def transpose(x):
    return moveaxis(x, -1, -2)


def less_than(x, y):
    return elementwise_not(greater_equal(x, y))


def greater_than(x, y):
    return elementwise_not(less_equal(x, y))


def elementwise_or(x, y):
    return elementwise_not(elementwise_and(elementwise_not(x), elementwise_not(y)))


def equals(x, y):
    return elementwise_and(greater_equal(x, y), less_equal(x, y))


def concat(arg0, *args, axis: int = 0):
    if len(args) == 0:
        return arg0

    res = arg0
    for arg in args:
        res = concat_two(res, arg, axis=axis)
    return res


def split(x, indices, axis: int = 0):
    pieces = []
    rest, start = x, 0
    for index in indices:
        pieces.append(head(rest, axis=axis, index=index - start))
        rest = tail(rest, axis=axis, index=index - start)
        start = index
    pieces.append(rest)
    return tuple(pieces)


def getitem(x, i, axis: int = 0):
    if i < 0:
        i += x.shape[axis]
    elem = head(tail(x, axis=axis, index=i), axis=axis, index=1)
    new_shape = x.shape[:axis] + x.shape[axis + 1 :]
    return reshape(elem, new_shape=new_shape)
