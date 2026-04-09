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

from pathlib import Path
import zipfile

from .compute_graph import ComputeGraph, Const, Var


def dump(graph: ComputeGraph, file: str | Path):
    """Export a compute graph to a .zip file."""
    with zipfile.ZipFile(file, "w") as zf:
        graph_str, consts = _serialize_graph(graph)
        with zf.open("graph.txt", "w") as f:
            f.write(graph_str.encode("utf-8"))

        const_bytes = {name: const.value.tobytes("c") for name, const in consts.items()}
        for name, bytes in const_bytes.items():
            with zf.open(f"{name}.bin", "w") as f:
                f.write(bytes)


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
        return chr(97 + i) if i < 26 else letters(i // 26 - 1) + chr(97 + (i % 26))

    def serialize_atom(atom) -> str:
        id_dict = var_ids if isinstance(atom, Var) else const_ids
        if atom not in id_dict:
            id_dict[atom] = len(id_dict)
        i = id_dict[atom]
        name = letters(i)
        if isinstance(atom, Const):
            name = name.upper()
        return name + "[" + ", ".join(map(str, atom.shape)) + "]"

    def serialize_eqn(eqn):
        opts = "{" + ", ".join([f"{k}={v}" for k, v in eqn.options.items()]) + "}"
        repr = f"{serialize_atom(eqn.outvar)} = {eqn.primitive.name}{opts} "
        return repr + " ".join(map(serialize_atom, eqn.inputs))

    out = "input: " + ", ".join(map(serialize_atom, graph.invars)) + "\n"
    out += "\n".join(map(serialize_eqn, graph.equations)) + "\n"
    out += "output: " + ", ".join(map(serialize_atom, graph.outvars))

    consts = {letters(i).upper(): const for const, i in const_ids.items()}
    return out, consts


def _deserialize_graph(graph_repr: str, consts: dict[str, bytes]) -> ComputeGraph:
    vars = {}

    # TODO: implement
