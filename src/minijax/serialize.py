# Copyright (c) 2025 by David Boetius
# Licensed under the MIT License.
"""Import and export compute graphs.

The compute graphs are stored in a .zip file containing
a `graph.txt` file with the graph equations and an arbitrary
number of binary files containing the constant values.
For the constant `xyz`, the file `xyz.bin` contains the float64
values of the constant in row-major order. The shape of the constant
is specified in the `graph.txt` file.

Variables are lower case, constants are upper case.
"""

import ast
from pathlib import Path
import zipfile

import numpy as np

from . import core as minijax_core
from .compute_graph import ComputeGraph, Const, Equation, Var
from .eval import Array


def dump(graph: ComputeGraph, file: str | Path):
    """Export a compute graph to a .zip file."""
    with zipfile.ZipFile(file, "w") as zf:
        graph_str, consts = _serialize_graph(graph)

        with zf.open("graph.txt", "w") as f:
            f.write(graph_str.encode("utf-8"))

        for name, const in consts.items():
            array = const.value.array.astype(np.float64)
            data = array.tobytes("c")
            with zf.open(f"{name}.bin", "w") as f:
                f.write(data)


def load(file: str) -> ComputeGraph:
    with zipfile.ZipFile(file, "r") as zf:
        graph_str = zf.read("graph.txt").decode("utf-8")
        consts = {}
        for file in zf.namelist():
            if file.endswith(".bin"):
                name = file[:-4]
                consts[name] = zf.read(file)
    return _deserialize_graph(graph_str, consts)


def _serialize_graph(graph) -> tuple[str, dict[str, Const]]:
    var_ids = {}
    const_ids = {}

    def letters(i):
        if i < 26:
            return chr(97 + i)
        return letters(i // 26 - 1) + chr(97 + (i % 26))

    def serialize_atom(atom) -> str:
        id_dict = var_ids if isinstance(atom, Var) else const_ids
        if atom not in id_dict:
            id_dict[atom] = len(id_dict)

        i = id_dict[atom]
        name = letters(i)
        if isinstance(atom, Const):
            name = name.upper()

        shape_str = ",".join(map(str, atom.shape))
        return f"{name}[{shape_str}]"

    def serialize_eqn(eqn):
        opt_parts = [f"{k}: {v}" for k, v in eqn.options.items()]
        opts = "{" + "; ".join(opt_parts) + "}"

        outvar = serialize_atom(eqn.outvar)
        inputs = " ".join(map(serialize_atom, eqn.inputs))
        return f"{outvar} = {eqn.primitive.name}{opts} {inputs}"

    input_line = "input: " + "; ".join(map(serialize_atom, graph.invars))
    eqn_lines = "\n".join(map(serialize_eqn, graph.equations))
    output_line = "output: " + "; ".join(map(serialize_atom, graph.outvars))
    out = f"{input_line}\n{eqn_lines}\n{output_line}"

    consts = {letters(i).upper(): const for const, i in const_ids.items()}
    return out, consts


def _deserialize_graph(graph_repr: str, consts: dict[str, bytes]) -> ComputeGraph:
    lines = graph_repr.splitlines()
    input_line = lines[0].strip()
    output_line = lines[-1].strip()
    eqn_lines = lines[1:-1]
    atoms = {}

    def deserialize_atom(s: str):
        name, dims_str = s.strip().split("[", 1)
        raw_dims = dims_str.rstrip("]")

        if raw_dims:
            shape = tuple(int(d) for d in raw_dims.split(","))
        else:
            shape = ()

        if name not in atoms:
            if name.upper() == name:  # uppercase => Const
                val = np.frombuffer(consts[name]).reshape(shape)
                atoms[name] = Const(Array(val))
            else:
                atoms[name] = Var(shape)

        atom = atoms[name]
        assert atom.shape == shape, f"Inconsistent shapes for {name}: {atom.shape}, {shape}."
        return atom

    def deserialize_eqn(line: str):
        outvar_str, expr = line.split("=", 1)
        expr = expr.strip()

        brace_open = expr.index("{")
        brace_close = expr.index("}")

        primitive_name = expr[:brace_open]
        primitive = getattr(minijax_core, primitive_name)

        opts_block = expr[brace_open + 1 : brace_close].strip()
        if opts_block:
            opts = {}
            for pair in opts_block.split(";"):
                k, v = pair.split(":", 1)
                opts[k.strip()] = ast.literal_eval(v.strip())
        else:
            opts = {}

        outvar = deserialize_atom(outvar_str)

        inputs_str = expr[brace_close + 1 :].strip()
        inputs = tuple(deserialize_atom(t) for t in inputs_str.split())

        return Equation(primitive, inputs, outvar, opts)

    invar_strs = input_line[len("input:") :].split(";")
    invars = tuple(deserialize_atom(s) for s in invar_strs)

    eqns = tuple(map(deserialize_eqn, eqn_lines))

    outvar_strs = output_line[len("output:") :].split(";")
    outvars = tuple(deserialize_atom(s) for s in outvar_strs)

    return ComputeGraph(invars, outvars, eqns)
