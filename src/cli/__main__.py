# Copyright (c) 2025 by David Boetius
# Licensed under the MIT License.
"""Dispatch to eval, grad, or train entry points.

Usage:
    python -m cli {eval|grad|train} ...
"""

import sys

from cli.eval import main as eval_main
from cli.grad import main as grad_main
from cli.train import main as train_main


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("eval", "grad", "train"):
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv.pop(1)
    if cmd == "eval":
        eval_main()
    elif cmd == "grad":
        grad_main()
    else:
        train_main()


if __name__ == "__main__":
    main()
