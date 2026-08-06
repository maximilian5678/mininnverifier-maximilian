# Copyright (c) 2025 by David Boetius
# Licensed under the MIT License.
"""Dispatch to eval, grad, bounds, affine_bounds, train, or verify entry points.

Usage:
    python -m cli {eval|grad|bounds|affine_bounds|train|verify} ...
"""

import sys

from cli.bounds import main as bounds_main
from cli.affine_bounds import main as affine_bounds_main
from cli.eval import main as eval_main
from cli.grad import main as grad_main
from cli.train import main as train_main
from cli.verify import main as verify_main
from cli.verify2 import main as verify2_main


SUBCOMMANDS = {
    "eval": eval_main,
    "grad": grad_main,
    "bounds": bounds_main,
    "affine_bounds": affine_bounds_main,
    "train": train_main,
    "verify": verify_main,
    "verify2": verify2_main,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv.pop(1)
    SUBCOMMANDS[cmd]()


if __name__ == "__main__":
    main()
