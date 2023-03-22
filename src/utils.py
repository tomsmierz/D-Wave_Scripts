import os

import dwave_networkx as dnx
import pandas as pd


def load_pegasus(path: str, name: str = "001"):
    df = pd.read_csv(
        os.path.join(path, f"{name}.txt"),
        sep=" ",
        index_col=False,
        skiprows=1,
        header=None,
    )
    h = {}
    J = {}
    for index, row in df.iterrows():
        if row[0] == row[1]:
            h[int(row[0] - 1)] = row[2]
            # h[int(row[0])] = row[2]
        else:
            J[(int(row[0] - 1), int(row[1] - 1))] = row[2]
            # J[(int(row[0]), int(row[1]))] = row[2]
    return h, J


def load_pegasus_tuple(path: str, name: str = "001"):
    df = pd.read_csv(
        os.path.join(path, f"{name}.txt"),
        sep=";",
        index_col=False,
        skiprows=1,
        header=None,
    )
    h = {}
    J = {}
    for index, row in df.iterrows():
        if row[0] == row[1]:
            # h[int(row[0] - 1)] = row[2]\
            h[eval(row[0])] = row[2]
        else:
            # J[(int(row[0] - 1), int(row[1] - 1))] = row[2]
            J[(eval(row[0]), eval(row[1]))] = row[2]
    return h, J


def prepare_data_to_save(device, mapping, couplings, bias) -> list[dict, dict]:
    if device is None:
        return [bias, couplings]
    couplings_dv = {
        (mapping(edge[0]), mapping(edge[1])): value
        for edge, value in couplings.items()
    }
    bias_dv = {mapping(node): value for node, value in bias.items()}
    return [bias_dv, couplings_dv]


def linear_to_nice(s: int) -> tuple:
    return dnx.pegasus_coordinates(16).linear_to_nice(s)


def nice_to_linear(t: tuple) -> int:
    return dnx.pegasus_coordinates(16).nice_to_linear(t)


def nice_to_spin_glass(node: tuple, size: int) -> int:
    t, y, x, u, k = node
    if u == 1:
        a = 4 + k + 1
    else:
        a = abs(k - 3) + 1
    b = abs(y - (size - 2))

    spin_glas_linear = 8 * t + 24 * x + 24 * (size - 1) * b + a
    return spin_glas_linear
