# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Compute affine output bounds on a scalar-output mininn network
using CROWN.

Usage:
    affine_bounds --output-dir <dir> <network.mininn> <input-spec> ...

Each input variable of the network is described by an inline marker followed
by one or two binary files (same convention as ``bounds``):

    box   <lb.bin> <ub.bin>     interval input (per-element lower/upper bound)
    point <value.bin>           fixed input (treated as a degenerate interval)

Unlike ``bounds`` (which writes constant interval bounds), CROWN produces an
*affine* relaxation of the network output as a function of the input. For a
scalar-output network ``f`` and input ``x`` it returns weight/bias pairs with

    lb_weight @ x + lb_bias  <=  f(x)  <=  ub_weight @ x + ub_bias

for every ``x`` in the input box. ``lb_weight``/``ub_weight`` have the same
shape as the (single) network input; the biases are scalars.

For each network output ``i`` four float64 binary files are written into
``--output-dir`` and their paths printed to stdout in this order:

    output_<i>_lb_weight.bin
    output_<i>_lb_bias.bin
    output_<i>_ub_weight.bin
    output_<i>_ub_bias.bin
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from minijax.serialize import load
from minijax.eval import Array
from minijax.jit import run_graph
from mininnverifier.ibp import Box
from mininnverifier.alpha_crown import alpha_crown


MARKERS = ("box", "point")


def _load_array(path, shape):
    return Array(np.fromfile(path, dtype=np.float64).reshape(shape))


def _parse_inputs(tokens, invars):
    """Walk ``tokens`` consuming one input spec per network input."""
    inputs = []
    i = 0
    for var in invars:
        if i >= len(tokens):
            print(
                f"Error: network has {len(invars)} input(s), "
                f"but only {len(inputs)} input spec(s) were provided.",
                file=sys.stderr,
            )
            sys.exit(1)
        marker = tokens[i]
        if marker == "box":
            if i + 2 >= len(tokens):
                print("Error: 'box' marker requires two file paths (lb, ub).", file=sys.stderr)
                sys.exit(1)
            lb = _load_array(tokens[i + 1], var.shape)
            ub = _load_array(tokens[i + 2], var.shape)
            inputs.append(Box(lb, ub))
            i += 3
        elif marker == "point":
            if i + 1 >= len(tokens):
                print("Error: 'point' marker requires one file path.", file=sys.stderr)
                sys.exit(1)
            inputs.append(_load_array(tokens[i + 1], var.shape))
            i += 2
        else:
            print(
                f"Error: unknown input marker {marker!r} (expected one of {MARKERS}).",
                file=sys.stderr,
            )
            sys.exit(1)

    if i != len(tokens):
        print(
            f"Error: trailing arguments after parsing {len(invars)} input(s): {tokens[i:]}",
            file=sys.stderr,
        )
        sys.exit(1)

    return inputs


def _to_numpy(value):
    """Concretize a minijax Value/Array to a flat-savable float64 numpy array."""
    arr = value.array if hasattr(value, "array") else np.asarray(value)
    return np.asarray(arr, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(
        description="Compute affine (CROWN) output bounds for a scalar-output mininn network."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("network", type=str)
    parser.add_argument("inputs", nargs="*", type=str)
    args = parser.parse_args()

    graph = load(args.network)
    inputs = _parse_inputs(args.inputs, graph.invars)
    if len(graph.invars) != 1:
        print(
            f"Error: affine_bounds expects a single-input network, "
            f"got {len(graph.invars)} inputs.",
            file=sys.stderr,
        )
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # CROWN bounds the single scalar output as an affine function of the input:
    #   lb_weight @ x + lb_bias  <=  f(x)  <=  ub_weight @ x + ub_bias.
    # ``alpha_crown`` returns a ``(lower, upper)`` pair of ``AffineBound``s, each
    # with a per-input ``weights`` tuple and a scalar ``bias``; for the single
    # network input the relevant coefficient is ``weights[0]`` (input-shaped).
    def fn(*xs):
        return run_graph(graph, xs)[0]

    lower, upper = alpha_crown(fn)(*inputs)

    out_idx = 0
    parts = [
        ("lb_weight", lower.weights[0]),
        ("lb_bias", lower.bias),
        ("ub_weight", upper.weights[0]),
        ("ub_bias", upper.bias),
    ]
    for name, value in parts:
        path = args.output_dir / f"output_{out_idx}_{name}.bin"
        _to_numpy(value).tofile(path)
        print(path)


if __name__ == "__main__":
    main()
