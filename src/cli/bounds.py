# Copyright (c) 2025 by David Boetius
# Licensed under the MIT License.
"""Compute bounds on a mininn network using interval bound propagation.

Usage:
    bounds --output-dir <dir> <network.mininn> <input-spec> ...

Each input variable of the network is described by an inline marker followed
by one or two binary files:

    box   <lb.bin> <ub.bin>     interval input (per-element lower/upper bound)
    point <value.bin>           fixed input (treated as a degenerate interval)

The order of input specs matches ``graph.invars``. Each ``.bin`` file holds
float64 values in row-major order matching that input variable's shape.

For each network output, two float64 binary files are written into
``--output-dir``:

    output_<i>_lb.bin    lower bound
    output_<i>_ub.bin    upper bound

Both paths are printed to stdout (lower bound first, then upper bound).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from minijax.serialize import load
from minijax.eval import Array
from minijax.jit import run_graph
from mininnverifier.ibp import Box, ibp


MARKERS = ("box", "point")


def _load_array(path, shape):
    return Array(np.fromfile(path, dtype=np.float64).reshape(shape))


def _parse_inputs(tokens, invars):
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
                print(
                    "Error: 'box' marker requires two file paths (lb, ub).",
                    file=sys.stderr,
                )
                sys.exit(1)
            lb = _load_array(tokens[i + 1], var.shape)
            ub = _load_array(tokens[i + 2], var.shape)
            inputs.append(Box(lb, ub))
            i += 3
        elif marker == "point":
            if i + 1 >= len(tokens):
                print(
                    "Error: 'point' marker requires one file path.",
                    file=sys.stderr,
                )
                sys.exit(1)
            inputs.append(_load_array(tokens[i + 1], var.shape))
            i += 2
        else:
            print(
                f"Error: unknown input marker {marker!r} "
                f"(expected one of {MARKERS}).",
                file=sys.stderr,
            )
            sys.exit(1)

    if i != len(tokens):
        print(
            f"Error: trailing arguments after parsing {len(invars)} input(s): "
            f"{tokens[i:]}",
            file=sys.stderr,
        )
        sys.exit(1)

    return inputs


def main():
    parser = argparse.ArgumentParser(
        description="Compute output bounds for a mininn network via IBP."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("network", type=str)
    parser.add_argument("inputs", nargs="*", type=str)
    args = parser.parse_args()

    graph = load(args.network)
    inputs = _parse_inputs(args.inputs, graph.invars)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_bounds = ibp(lambda *xs: run_graph(graph, xs))(*inputs)
    for i, box in enumerate(out_bounds):
        lb_path = args.output_dir / f"output_{i}_lb.bin"
        ub_path = args.output_dir / f"output_{i}_ub.bin"
        box.lb.array.tofile(lb_path)
        box.ub.array.tofile(ub_path)
        print(lb_path)
        print(ub_path)


if __name__ == "__main__":
    main()
