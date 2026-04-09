# Copyright (c) 2025 by David Boetius
# Licensed under the MIT License.
"""Import and export compute graphs.

The compute graphs are stored in a .zip file containing
a `graph.txt` file with the graph equations and an arbitrary
number of binary files containing the constant values.
For the constant `xyz`, the file `xyz.bin` contains the float64
values of the constant in row-major order. The shape of the constant
is specified in the `graph.txt` file.
"""

from .compute_graph import ComputeGraph, Equation


def dump(graph: ComputeGraph, file: str):
    pass


def _serialize_graph(graph) -> str:
    def _serialize_eqn(eqn):
        opts = "[" + ", ".join([f"{k}: {v}" for k, v in eqn.options.items()]) + "]"
        repr = f"{eqn.outvar} = {eqn.primitive.name}{opts} "
        return repr + " ".join(map(str, eqn.inputs))

    pass
    out = "input: " + " ".join(map(str, graph.invars)) + "\n"
    out += "\n".join([f"  {eqn}" for eqn in graph.equations]) + "\n"
    out += "output: " + " ".join(map(str, graph.outvars))
    return out


def load(file: str) -> ComputeGraph:
    pass
