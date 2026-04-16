#!/usr/bin/env python3
# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Generate batched eval tests for all minijax primitives.

Run from the repository root:
    python tests/milestone1/base/unit/eval/generate_primitive_eval_tests.py
"""

import json
from pathlib import Path

import numpy as np

from minijax import core  # noqa: E402
from minijax.compute_graph import make_graph  # noqa: E402
from minijax.eval import Array  # noqa: E402
from minijax.serialize import dump  # noqa: E402


SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parents[4]


DEFAULT_TOLERANCE = 1e-4


def create_test(name, fn, inputs_data, *, tolerance=DEFAULT_TOLERANCE):
    test_dir = SCRIPT_DIR / name
    resources_dir = test_dir / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)

    arrays = [Array(d) for d in inputs_data]
    graph = make_graph(fn)(*arrays)

    network_file = f"{name}_network.mininn"
    dump(graph, resources_dir / network_file)

    input_files = []
    for i, d in enumerate(inputs_data):
        fname = f"input_{i}.bin"
        np.asarray(d, dtype=np.float64).tofile(resources_dir / fname)
        input_files.append(f"resources/{fname}")

    raw_out = fn(*arrays)
    if not isinstance(raw_out, (list, tuple)):
        raw_out = [raw_out]

    expected_files = []
    for i, out in enumerate(raw_out):
        fname = f"expected_output_{i}.bin"
        out.array.astype(np.float64).tofile(test_dir / fname)
        expected_files.append(fname)

    config = {
        "command": "eval",
        "network": f"resources/{network_file}",
        "inputs": input_files,
        "expected_outputs": expected_files,
    }
    if tolerance != DEFAULT_TOLERANCE:
        config["tolerance"] = tolerance
    (test_dir / "test.json").write_text(json.dumps(config, indent=4) + "\n")
    print(f"  created {test_dir.relative_to(REPO_ROOT)}")


def main():
    # neg: 100 values, shape (100,)
    create_test("neg", lambda x: core.neg(x), [np.linspace(-5.0, 5.0, 100)])

    # add: two inputs, shape (50,) each
    create_test(
        "add", lambda x, y: core.add(x, y), [np.linspace(-5.0, 5.0, 50), np.linspace(0.0, 10.0, 50)]
    )

    # mul: two inputs, shape (50,) each
    create_test(
        "mul", lambda x, y: core.mul(x, y), [np.linspace(-5.0, 5.0, 50), np.linspace(0.0, 10.0, 50)]
    )

    # dot: batched matrix multiply — 10 samples of 8 features → 5 outputs
    # x: (10, 8) = 80 values; y: (8, 5) = 40 values → output (10, 5)
    create_test(
        "dot",
        lambda x, y: core.dot(x, y),
        [np.linspace(-1.0, 1.0, 80).reshape(10, 8), np.linspace(-1.0, 1.0, 40).reshape(8, 5)],
    )

    # reciprocal: 50 strictly-positive values, shape (50,)
    create_test("reciprocal", lambda x: core.reciprocal(x), [np.linspace(0.5, 5.0, 50)])

    # relu: 100 values spanning negative and positive, shape (100,)
    create_test("relu", lambda x: core.relu(x), [np.linspace(-5.0, 5.0, 100)])

    # square: 100 values, shape (10, 10)
    create_test("square", lambda x: core.square(x), [np.linspace(-5.0, 5.0, 100).reshape(10, 10)])

    # sqrt: 100 strictly-positive values, shape (10, 10)
    create_test("sqrt", lambda x: core.sqrt(x), [np.linspace(0.01, 5.0, 100).reshape(10, 10)])

    # exp: 50 values, shape (50,) — moderate range to avoid overflow
    create_test("exp", lambda x: core.exp(x), [np.linspace(-3.0, 3.0, 50)])

    # log: 100 strictly-positive values, shape (10, 10)
    create_test("log", lambda x: core.log(x), [np.linspace(0.01, 10.0, 100).reshape(10, 10)])

    # where: 3 inputs each shape (50,)
    # condition: 1.0 for even indices, 0.0 for odd (truthy/falsy)
    create_test(
        "where",
        lambda c, x, y: core.where(c, x, y),
        [
            np.where(np.arange(50) % 2 == 0, 1.0, 0.0),
            np.linspace(-5.0, 5.0, 50),
            np.linspace(5.0, -5.0, 50),
        ],
    )

    # expand_dims: shape (10, 10) → (10, 10, 1), new axis at -1
    create_test(
        "expand_dims",
        lambda x: core.expand_dims(x, axes=-1),
        [np.linspace(-5.0, 5.0, 100).reshape(10, 10)],
    )

    # moveaxis: shape (10, 5, 2) → (5, 2, 10), move axis 0 to -1
    create_test(
        "moveaxis",
        lambda x: core.moveaxis(x, source=0, destination=-1),
        [np.linspace(-5.0, 5.0, 100).reshape(10, 5, 2)],
    )

    # reshape: shape (10, 10) → (20, 5)
    create_test(
        "reshape",
        lambda x: core.reshape(x, new_shape=(20, 5)),
        [np.linspace(-5.0, 5.0, 100).reshape(10, 10)],
    )

    # reduce_sum: shape (10, 10), sum over axis 1 → (10,)
    create_test(
        "reduce_sum",
        lambda x: core.reduce_sum(x, axes=1),
        [np.linspace(-5.0, 5.0, 100).reshape(10, 10)],
    )

    print("Done.")


if __name__ == "__main__":
    main()
