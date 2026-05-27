# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Train a neural network on a dataset and save checkpoints.

Usage:
    train --output-dir <dir> <dataset_id> <images.bin> <labels.bin>

The dataset_id determines hyperparameters (e.g., "MNIST").
Images and labels are float64 binary files in row-major order.
Checkpoints are saved as .mininn files after each epoch.

Output protocol (stdout):
    First line: eval_batch_size: <N>
    Subsequent lines: one checkpoint file path per line.
"""

import argparse
import itertools as it
import json
import sys
from pathlib import Path

import numpy as np

from minijax.compute_graph import make_graph
from minijax.core import add, div, mul, sqrt, square, sub
from minijax.eval import Array, zeros
from minijax.grad import value_and_grad
from minijax.jit import jit
from minijax.nested_containers import map_structure
from minijax.nn import conv_net, cross_entropy, init_conv_net, init_mlp, mlp
from minijax.serialize import dump
from minijax.vmap import vmap


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

HYPERPARAMS_DIR = Path(__file__).parent / "hyperparams"


def load_hyperparams(dataset_id):
    """Load hyperparameters from a JSON file in the hyperparams directory."""
    path = HYPERPARAMS_DIR / f"{dataset_id}.json"
    if not path.exists():
        available = [p.stem for p in HYPERPARAMS_DIR.glob("*.json")]
        print(
            f"Error: unknown dataset '{dataset_id}'. "
            f"Available: {', '.join(available)}",
            file=sys.stderr,
        )
        sys.exit(1)
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Optimizer (Adam) — adapted from examples/train_mnist.ipynb
# ---------------------------------------------------------------------------


def adam(params, grads, opt_state, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    def m_update(g, m_prev):
        return add(mul(Array(beta1), m_prev), mul(Array(1 - beta1), g))

    def v_update(g, v_prev):
        return add(mul(Array(beta2), v_prev), mul(Array(1 - beta2), square(g)))

    m_prevs, v_prevs, beta1powtm1, beta2powtm2 = opt_state
    m_new = map_structure(m_update, grads, m_prevs)
    v_new = map_structure(v_update, grads, v_prevs)
    beta1powt = mul(beta1powtm1, Array(beta1))
    beta2powt = mul(beta2powtm2, Array(beta2))

    def param_update(p, m, v):
        m_hat = div(m, sub(Array(1), beta1powt))
        v_hat = div(v, sub(Array(1), beta2powt))
        return sub(p, mul(Array(lr), div(m_hat, add(sqrt(v_hat), Array(eps)))))

    new_params = map_structure(param_update, params, m_new, v_new)
    return new_params, (m_new, v_new, beta1powt, beta2powt)


def init_adam_state(params):
    m = map_structure(lambda p: zeros(p.shape), params)
    v = map_structure(lambda p: zeros(p.shape), params)
    beta1powt = Array(1)
    beta2powt = Array(1)
    return m, v, beta1powt, beta2powt


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------


def save_checkpoint(params, forward_fn, output_dir, epoch, eval_batch_size, in_size):
    """Save a model checkpoint as a .mininn file using the closure pattern."""
    dummy_x = zeros((eval_batch_size, in_size))

    def model_fn(x):
        return forward_fn(x, params)

    graph = make_graph(model_fn)(dummy_x)
    path = output_dir / f"checkpoint_epoch_{epoch}.mininn"
    dump(graph, path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train a neural network.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("dataset", type=str)
    parser.add_argument("images", type=str)
    parser.add_argument("labels", type=str)
    args = parser.parse_args()

    cfg = load_hyperparams(args.dataset)
    in_size = cfg["in_size"]
    num_epochs = cfg["num_epochs"]
    batch_size = cfg["batch_size"]
    learning_rate = cfg["learning_rate"]
    eval_batch_size = cfg["eval_batch_size"]
    rng_key = cfg["rng_key"]
    model_type = cfg.get("model", "mlp")

    # Load data
    images = np.fromfile(args.images, dtype=np.float64)
    labels = np.fromfile(args.labels, dtype=np.float64)
    num_samples = images.size // in_size
    images = images.reshape(num_samples, in_size)

    # num_classes aus den tatsächlichen Labels ableiten (kann 5 oder 10 sein)
    num_classes = labels.size // num_samples
    labels = labels.reshape(num_samples, num_classes)

    # Model setup
    if model_type == "conv":
        conv_specs = [tuple(s) for s in cfg["conv_specs"]]
        dense_sizes = cfg.get("dense_sizes", [])
        params, arch = init_conv_net(conv_specs, dense_sizes, num_classes, rng_key=rng_key)

        def forward(x, params):
            return conv_net(x, params, arch)

        def loss(x, y_true, params):
            return cross_entropy(forward(x, params), y_true)

        # ConvNet VJP uses direct numpy — cannot trace under jit, run eager
        def train_step(x, y_true, params, opt_state):
            loss_val, (_, _, param_grads) = value_and_grad(loss)(x, y_true, params)
            new_params, new_opt_state = adam(params, param_grads, opt_state, lr=learning_rate)
            return new_params, new_opt_state, loss_val

    else:
        layer_sizes = list(cfg["layer_sizes"])
        layer_sizes[-1] = num_classes
        params = init_mlp(in_size, layer_sizes, rng_key=rng_key)
        _model = vmap(mlp, (0, None))

        def forward(x, params):
            return _model(x, params)

        def loss(x, y_true, params):
            return cross_entropy(forward(x, params), y_true)

        @jit
        def train_step(x, y_true, params, opt_state):
            loss_val, (_, _, param_grads) = value_and_grad(loss)(x, y_true, params)
            new_params, new_opt_state = adam(params, param_grads, opt_state, lr=learning_rate)
            return new_params, new_opt_state, loss_val

    # Output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Print eval batch size as first line of stdout
    print(f"eval_batch_size: {eval_batch_size}")

    # Training loop
    opt_state = init_adam_state(params)
    np_rng = np.random.default_rng(rng_key)

    for epoch in range(1, num_epochs + 1):
        rand_perm = np_rng.permutation(num_samples)
        for batch_idx in it.batched(rand_perm, batch_size):
            if len(batch_idx) != batch_size:
                continue
            x = Array(images[batch_idx, :])
            y = Array(labels[batch_idx, :])
            params, opt_state, loss_val = train_step(x, y, params, opt_state)

        # Save checkpoint after each epoch
        cp_path = save_checkpoint(
            params, forward, args.output_dir, epoch, eval_batch_size, in_size
        )
        print(cp_path)


if __name__ == "__main__":
    main()
