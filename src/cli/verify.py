# Copyright (c) 2025 by David Boetius
# Licensed under the MIT License.
"""Verify whether a mininn network output stays non-negative over an input box.

    verify --output-dir <dir> <network.mininn> box <lb.bin> <ub.bin> [point <x.bin> ...]

Exactly one input is a ``box``; remaining inputs are ``point``s. Prints ``sat``
as the last stdout line if f(x) >= 0 is proven over the box. Otherwise writes one
``counterexample_<i>.bin`` per network input, prints their paths, then ``viol``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from minijax.serialize import load
from minijax.eval import Array
from minijax.jit import run_graph
from mininnverifier.ibp import Box
#from mininnverifier.input_splitting_bab import input_splitting_bab
from mininnverifier.input_splitting_batched_bab import input_splitting_bab

def _load_array(path, shape):
    return Array(np.fromfile(path, dtype=np.float64).reshape(shape))

def _parse_inputs(tokens, invars):
    inputs = [] # one Box or Array per network input
    box_index = None
    i = 0
    for idx, var in enumerate(invars):
        if i >= len(tokens):
            print("Error: too few input specs.", file=sys.stderr)
            sys.exit(1)
        marker = tokens[i]
        if marker == "box":
            if box_index is not None:
                print("Error: verify expects exactly one box input.", file=sys.stderr)
                sys.exit(1)
            inputs.append(Box(_load_array(tokens[i + 1], var.shape),
                              _load_array(tokens[i + 2], var.shape)))
            box_index = idx
            i += 3
        elif marker == "point":
            inputs.append(_load_array(tokens[i + 1], var.shape))
            i += 2
        else:
            print(f"Error: unknown marker {marker!r}.", file=sys.stderr)
            sys.exit(1)

    if box_index is None:
        print("Error: verify requires one box input.", file=sys.stderr)
        sys.exit(1)
    return inputs, box_index

def main():
    parser = argparse.ArgumentParser(description="Verify f(x) >= 0 over an input box.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("network", type=str)
    parser.add_argument("inputs", nargs="*", type=str)
    args = parser.parse_args()

    graph = load(args.network)
    inputs, box_index = _parse_inputs(args.inputs, graph.invars)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    box = inputs[box_index]

    # f as a function of the box input only; points are captured as constants.
    def f(box_val):
        full_inputs = list(inputs)
        full_inputs[box_index] = box_val
        return run_graph(graph, full_inputs)[0]

    counterexample = input_splitting_bab(f)(box)

    if counterexample is None:
        print("sat")
        return

    ce_paths = []
    for idx, inp in enumerate(inputs):
        ce = counterexample if idx == box_index else inp # a point is its own counterexample
        path = args.output_dir / f"counterexample_{idx}.bin"
        ce.array.tofile(path)
        ce_paths.append(path)

    for path in ce_paths:
        print(path)
    print("viol")


if __name__ == "__main__":
    main()
