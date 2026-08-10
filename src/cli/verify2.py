# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Verify a property of a mininn network via node-splitting branch-and-bound.

Usage:
    verify2 --output-dir <dir> <network.mininn> box <lb.bin> <ub.bin>

See verify.py for more details.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

np.seterr(divide="ignore", invalid="ignore", over="ignore")

from minijax.serialize import load
from minijax.jit import run_graph
from mininnverifier.ibp import Box
from mininnverifier.node_splitting_bab import node_splitting_bab

from cli.bounds import _parse_inputs


def main():
    parser = argparse.ArgumentParser(
        description="Verify margin(x) >= 0 over an input box via node-splitting BaB."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--time-limit", type=float, default=float(os.environ.get("VERIFY2_TIME_LIMIT", 540.0)),
        help="wall-clock budget in seconds; the test runner kills us at its own timeout, "
             "so stop a little earlier and spend what is left on a final attack.",
    )
    parser.add_argument("--beta-iters", type=int, default=8)
    parser.add_argument("network", type=str)
    parser.add_argument("inputs", nargs="*", type=str)
    args = parser.parse_args()

    graph = load(args.network)
    inputs = _parse_inputs(args.inputs, graph.invars)

    box_positions = [i for i, inp in enumerate(inputs) if isinstance(inp, Box)]
    if len(box_positions) != 1:
        print(
            f"Error: verify expects exactly one 'box' input (the verification "
            f"variable), got {len(box_positions)}.",
            file=sys.stderr,
        )
        sys.exit(1)
    box_pos = box_positions[0]
    x_box = inputs[box_pos]

    def margin(x):
        full_inputs = list(inputs)
        full_inputs[box_pos] = x
        return run_graph(graph, full_inputs)[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        counterexample = node_splitting_bab(
            margin, time_limit=args.time_limit, beta_iters=args.beta_iters
        )(x_box)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    if counterexample is None:
        print("sat")
        return

    # Violated: assemble the full witness (counterexample for the box input,
    # the fixed value for each point input), write one .bin per network input,
    # print the paths, and finish with the verdict on the final line.
    witness = list(inputs)
    witness[box_pos] = counterexample
    for i, inp in enumerate(witness):
        cx_path = args.output_dir / f"counterexample_{i}.bin"
        np.asarray(inp.array, dtype=np.float64).tofile(cx_path)
        print(cx_path)
    print("viol")


if __name__ == "__main__":
    main()