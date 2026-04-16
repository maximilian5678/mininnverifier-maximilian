# Copyright (c) 2025 by David Boetius
# Licensed under the MIT License.
"""Compute gradients of a mininn network with respect to its inputs.

Usage:
    grad --output-dir <dir> <network.mininn> <input1.bin> [<input2.bin> ...]

Each input file contains float64 values in row-major order matching the shape
of the corresponding network input variable. Gradients are computed as
d(sum of outputs)/d(inputs). Gradients are written as float64 binary files
to the output directory, one file per input variable.
Filenames are printed to stdout, one per line.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from minijax.serialize import load
from minijax.eval import Array
from minijax.grad import _forward, _backwards


def main():
    parser = argparse.ArgumentParser(description="Compute network gradients.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("network", type=str)
    parser.add_argument("inputs", nargs="*", type=str)
    args = parser.parse_args()

    graph = load(args.network)

    if len(args.inputs) != len(graph.invars):
        print(
            f"Error: network has {len(graph.invars)} input(s), "
            f"but {len(args.inputs)} input file(s) were provided.",
            file=sys.stderr,
        )
        sys.exit(1)

    inputs = [
        Array(np.fromfile(path, dtype=np.float64).reshape(var.shape))
        for var, path in zip(graph.invars, args.inputs)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    primals = _forward(graph, inputs)
    out_tangents = [Array(np.ones(ov.shape)) for ov in graph.outvars]
    in_tangents = _backwards(graph, primals, out_tangents)

    for i, grad in enumerate(in_tangents):
        out_path = args.output_dir / f"grad_{i}.bin"
        grad.array.tofile(out_path)
        print(out_path)


if __name__ == "__main__":
    main()
