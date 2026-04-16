#!/usr/bin/env python3
# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Generate eval tests for trained neural network classifiers.

Networks are trained on Gaussian blob datasets using minijax.
Three families of networks are generated:
  - Shallow: single hidden layer, widths 4–128
  - Deep FC: uniform-width MLPs with 2–10 layers, widths 4–64
  - Residual: same as deep FC with skip connections on hidden layers

Run from the repository root:
    python tests/milestone1/base/unit/eval/generate_networks.py
"""

import json
from pathlib import Path

import numpy as np

from minijax import core, nn
from minijax.compute_graph import make_graph
from minijax.eval import Array
from minijax.grad import _backwards, _forward
from minijax.nested_containers import flatten, map_structure, unflatten
from minijax.serialize import dump

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parents[4]


# ======================================================================
# Batch-compatible network definitions
# ======================================================================

def mlp_batched(x, params):
    """MLP on batched input x of shape (batch, features)."""
    for p in params[:-1]:
        x = x @ p["weight"] + p["bias"]
        x = core.relu(x)
    return x @ params[-1]["weight"] + params[-1]["bias"]


def residual_mlp_batched(x, params):
    """MLP with residual connections on batched input x of shape (batch, features).

    Architecture:
      params[0]:    input projection  (in_dim  → width), followed by ReLU
      params[1:-1]: residual blocks   (width   → width), each relu(linear(x)) + x
      params[-1]:   output projection (width   → n_classes), no activation
    """
    x = core.relu(x @ params[0]["weight"] + params[0]["bias"])
    for p in params[1:-1]:
        residual = x
        x = core.relu(x @ p["weight"] + p["bias"])
        x = x + residual
    return x @ params[-1]["weight"] + params[-1]["bias"]


# ======================================================================
# Training with compiled gradients
# ======================================================================

def _make_compiled_value_and_grad(fn, example_primals):
    """Compile fn+gradient once; return a function that re-runs them cheaply.

    example_primals must have the same structure and shapes as all future
    calls. The graph is traced once and then executed with _forward/_backwards
    on every subsequent invocation, avoiding repeated Python-level tracing.
    """
    cg = make_graph(fn)(*example_primals)
    _, in_structure = flatten(example_primals)

    def v_and_g(*primals):
        flat_primals, _ = flatten(primals)
        primals_dict = _forward(cg, list(flat_primals))
        loss_val = primals_dict[cg.outvars[0]]
        in_tangents = _backwards(cg, primals_dict, [Array(1.0)])
        grads = unflatten(in_structure, in_tangents)
        return loss_val, grads

    return v_and_g


def train(
    X_arr, Y_arr, layer_sizes, n_classes,
    *, n_epochs, lr, weight_decay_coef, rng_key, residual=False, init_scale=1.0,
):
    """Train a classifier with SGD on one-hot targets Y_arr.

    Returns the trained params (list of {weight, bias} dicts).
    init_scale multiplies the Kaiming-uniform weight initialisation; use a
    value < 1 for very deep networks to prevent overflow in the softmax.
    """
    params = nn.init_mlp(X_arr.shape[1], layer_sizes + [n_classes], rng_key)
    if init_scale != 1.0:
        params = [
            {"weight": Array(p["weight"].array * init_scale), "bias": p["bias"]}
            for p in params
        ]
    forward = residual_mlp_batched if residual else mlp_batched

    def loss_fn(params):
        logits = forward(X_arr, params)
        return nn.cross_entropy(logits, Y_arr) + Array(weight_decay_coef) * nn.weight_decay(params)

    v_and_g = _make_compiled_value_and_grad(loss_fn, (params,))

    for _ in range(n_epochs):
        _, (grads,) = v_and_g(params)
        params = map_structure(
            lambda p, g: Array(p.array - lr * g.array),
            params, grads,
        )

    return params


# ======================================================================
# Data generation
# ======================================================================

def make_blobs(n_features, n_classes, *, rng_seed):
    """Gaussian blob classification dataset (n_classes blobs, 200 pts each)."""
    rng = np.random.default_rng(rng_seed)
    n_per_class = 200
    centers = rng.uniform(-3.0, 3.0, (n_classes, n_features))
    X_parts, y_parts = [], []
    for i, center in enumerate(centers):
        X_parts.append(rng.normal(center, 0.7, (n_per_class, n_features)))
        y_parts.append(np.full(n_per_class, i, dtype=np.int64))
    X = np.vstack(X_parts).astype(np.float64)
    y = np.concatenate(y_parts)
    Y = np.eye(n_classes, dtype=np.float64)[y]
    return X, Y


def make_linspace_inputs(X_train, n_points):
    """n_points test inputs covering the per-feature range of X_train.

    Each row traverses a linear path from the minimum to the maximum of the
    training data for each feature dimension independently.
    """
    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)
    t = np.linspace(0.0, 1.0, n_points)
    return (mins + np.outer(t, maxs - mins)).astype(np.float64)


# ======================================================================
# Test creation (same pattern as generate.py)
# ======================================================================

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


def build_test(
    name, layer_sizes, in_dim, n_classes,
    *, n_epochs, lr, n_inputs,
    weight_decay_coef=1e-4, residual=False, init_scale=1.0, rng_seed=42,
):
    """Train a network and generate a test directory for it."""
    X, Y = make_blobs(in_dim, n_classes, rng_seed=rng_seed * 13 + in_dim)

    # Standardise inputs so deep networks train without overflow.
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std

    X_arr, Y_arr = Array(X), Array(Y)

    params = train(
        X_arr, Y_arr, layer_sizes, n_classes,
        n_epochs=n_epochs, lr=lr, weight_decay_coef=weight_decay_coef,
        rng_key=rng_seed, residual=residual, init_scale=init_scale,
    )

    forward = residual_mlp_batched if residual else mlp_batched

    def network(x):
        return forward(x, params)

    x_test = make_linspace_inputs(X, n_inputs)
    create_test(name, network, [x_test])


# ======================================================================
# Test configurations
# ======================================================================

def main():
    print("Generating shallow network tests…")
    # Single hidden layer, widths 4–128; input dims 2–16
    build_test("shallow_4",   [4],   in_dim=2,  n_classes=2, n_epochs=500, lr=0.01, n_inputs=50,  rng_seed=42)
    build_test("shallow_8",   [8],   in_dim=2,  n_classes=3, n_epochs=500, lr=0.01, n_inputs=100, rng_seed=43)
    build_test("shallow_16",  [16],  in_dim=4,  n_classes=3, n_epochs=500, lr=0.01, n_inputs=100, rng_seed=44)
    build_test("shallow_32",  [32],  in_dim=8,  n_classes=4, n_epochs=500, lr=0.01, n_inputs=150, rng_seed=45)
    build_test("shallow_64",  [64],  in_dim=12, n_classes=5, n_epochs=500, lr=0.01, n_inputs=200, rng_seed=46)
    build_test("shallow_128", [128], in_dim=16, n_classes=3, n_epochs=500, lr=0.01, n_inputs=250, rng_seed=47)

    print("\nGenerating deep FC network tests…")
    # Uniform-width MLPs, 2–10 layers, widths 4–64; input dims 2–16
    build_test("fc_2x4",   [4]  * 2,  in_dim=2,  n_classes=2, n_epochs=600,  lr=0.01,  n_inputs=50,  rng_seed=50)
    build_test("fc_2x64",  [64] * 2,  in_dim=2,  n_classes=2, n_epochs=600,  lr=0.01,  n_inputs=100, rng_seed=51)
    build_test("fc_3x16",  [16] * 3,  in_dim=4,  n_classes=3, n_epochs=600,  lr=0.01,  n_inputs=100, rng_seed=52)
    build_test("fc_3x64",  [64] * 3,  in_dim=12, n_classes=4, n_epochs=600,  lr=0.01,  n_inputs=200, rng_seed=53)
    build_test("fc_5x4",   [4]  * 5,  in_dim=2,  n_classes=2, n_epochs=800,  lr=0.01,  n_inputs=100, rng_seed=54)
    build_test("fc_5x32",  [32] * 5,  in_dim=4,  n_classes=5, n_epochs=800,  lr=0.01,  n_inputs=150, rng_seed=55)
    build_test("fc_7x16",  [16] * 7,  in_dim=8,  n_classes=3, n_epochs=800,  lr=0.01,  n_inputs=200, init_scale=0.1, rng_seed=56)
    build_test("fc_10x4",  [4]  * 10, in_dim=16, n_classes=3, n_epochs=1000, lr=0.005, n_inputs=250, init_scale=0.1, rng_seed=57)
    build_test("fc_10x32", [32] * 10, in_dim=8,  n_classes=4, n_epochs=1000, lr=0.005, n_inputs=250, init_scale=0.1, rng_seed=58)

    print("\nGenerating residual FC network tests…")
    # Same architecture family, residual skip connections on hidden layers
    build_test("residual_2x64",  [64] * 2,  in_dim=2,  n_classes=2, n_epochs=600,  lr=0.01,  n_inputs=100, residual=True, rng_seed=60)
    build_test("residual_3x16",  [16] * 3,  in_dim=4,  n_classes=3, n_epochs=600,  lr=0.01,  n_inputs=100, residual=True, rng_seed=61)
    build_test("residual_3x64",  [64] * 3,  in_dim=12, n_classes=4, n_epochs=600,  lr=0.01,  n_inputs=200, residual=True, rng_seed=62)
    build_test("residual_5x4",   [4]  * 5,  in_dim=2,  n_classes=2, n_epochs=800,  lr=0.01,  n_inputs=100, residual=True, rng_seed=63)
    build_test("residual_5x32",  [32] * 5,  in_dim=4,  n_classes=5, n_epochs=800,  lr=0.01,  n_inputs=150, residual=True, rng_seed=64)
    build_test("residual_7x16",  [16] * 7,  in_dim=8,  n_classes=3, n_epochs=800,  lr=0.01,  n_inputs=200, residual=True, init_scale=0.1, rng_seed=65)
    build_test("residual_10x4",  [4]  * 10, in_dim=16, n_classes=3, n_epochs=1000, lr=0.002, n_inputs=250, residual=True, init_scale=0.1, rng_seed=66)
    build_test("residual_10x32", [32] * 10, in_dim=8,  n_classes=4, n_epochs=1000, lr=0.002, n_inputs=250, residual=True, init_scale=0.1, rng_seed=67)

    print("\nDone.")


if __name__ == "__main__":
    main()
