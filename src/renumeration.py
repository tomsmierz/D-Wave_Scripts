import networkx as nx
import dwave_networkx as dnx
import pandas as pd
import random as rn
import matplotlib.pyplot as plt

from utils import load_pegasus, load_pegasus_tuple
from tqdm import tqdm
from typing import Dict


import os

cwd = os.getcwd()


def machine_to_5_tuple(h: Dict) -> Dict:
    h_tuple = {}

    l = sorted(list(h.keys()))
    s = l[0]
    s = dnx.pegasus_coordinates(16).linear_to_nice(s)
    if s[0] > 0:
        raise ValueError("s")
    for node in l:
        h_tuple[node] = dnx.pegasus_coordinates(16).linear_to_nice(node)
        tmp = h_tuple[node]
        t = (tmp[0], tmp[1] - s[1], tmp[2] - s[2], tmp[3], tmp[4])
        bl = [x >= 0 for x in t]
        if not all(bl):
            print(t)
            raise ValueError("t")
        h_tuple[node] = (tmp[0], tmp[1] - s[1], tmp[2] - s[2], tmp[3], tmp[4])

    return h_tuple


def tuple_to_linear(h_tuple: Dict, size: int) -> Dict:
    h_linear = {}
    for value in h_tuple.keys():
        if value[3] == 1:
            x = 4 + value[4] + 1
        else:
            x = abs(value[4] - 3) + 1
        y = abs(value[1] - (size - 2))

        h_linear[value] = 8 * value[0] + 24 * value[2] + 24 * (size - 1) * y + x
        # 24 * (size - 1) * value[0] + 24 * value[1] + 8 * value[2] + 4 * value[3] + value[4] + 1

    return h_linear


def tuple_to_dattani(h_tuple: Dict) -> Dict:
    h_dattani = {}

    for key, value in h_tuple.items():
        tmp = value
        h_dattani[key] = (tmp[2], tmp[1], tmp[0], tmp[3], tmp[4])

    return h_dattani


def dattani_to_linear(h_dattani: Dict, size: int) -> Dict:
    h_linear = {}
    for key, value in h_dattani.items():
        h_linear[key] = (
            24 * (size - 1) * value[0]
            + 24 * value[1]
            + 8 * value[2]
            + 4 * value[3]
            + value[4]
            + 1
        )

    return h_linear


def dattani_to_linear_2(h_dattani: Dict, size: int) -> Dict:
    h_linear = {}
    for key, value in h_dattani.items():
        h_linear[key] = (
            24 * (size - 1) * value[0]
            + 24 * value[2]
            + 8 * value[1]
            + 4 * value[3]
            + value[4]
            + 1
        )

    return h_linear


def renumerate(instance_path: str, name: str, size: int):
    h, J = load_pegasus_tuple(instance_path, name)
    J_rn = {}
    rn = {key: tuple_to_linear(h, size)[key] for key in h.keys()}
    h_rn = {rn[key]: value for key, value in h.items()}
    for key, value in J.items():
        J_rn[(rn[key[0]], rn[key[1]])] = value

    name = f"r_{name}_4.txt"

    with open(
            os.path.join(f"/home/tsmierzchalski/instances/renumerated/P{size}", name), "w"
        ) as f:
        f.write("# \n")

        h_rn_sorted = {k: h_rn[k] for k in sorted(h_rn)}
        J_rn_sorted = {k: J_rn[k] for k in sorted(J_rn)}

        for node, value in h_rn_sorted.items():
            f.write(f"{str(node)} {str(node)} {str(value)}" + "\n")
        for edge, value in J_rn_sorted.items():
            f.write(f"{str(edge[0])} {str(edge[1])} {str(value)}" + "\n")


def nice_to_spin_glass(node: tuple, size: int) -> int:
    t, y, x, u, k = node
    if u == 1:
        a = 4 + k + 1
    else:
        a = abs(k - 3) + 1
    b = abs(y - (size - 2))

    spin_glas_linear = 8 * t + 24 * x + 24 * (size - 1) * b + a
    return spin_glas_linear


def advantage_6_1_to_spinglass_int(r: int, size: int) -> int:
    if size not in [4, 8]:
        raise NotImplementedError("only work for P4 and P8")
    (t, y, x, u, k) = dnx.pegasus_coordinates(16).linear_to_nice(r)
    t_off = {4: 0, 8: 2}
    y_off = {(4, t): 2 for t in [0, 1, 2]} | {(8, 2): 3, (8, 0): 2, (8, 1): 2}
    x_off = {(4, t): 3 for t in [0, 1, 2]} | {(8, 2): 4, (8, 0): 5, (8, 1): 5}
    return nice_to_spin_glass(node=((t-t_off[size]) % 3, y-y_off[(size, t)], x-x_off[(size, t)], u, k), size=size)


def advantage_6_1_to_spinglass(node: tuple, size: int) -> int:
    t, y, x, u, k = node
    t_off = {4: 0, 8: 2}
    y_off = {(4, t): 2 for t in [0, 1, 2]} | {(8, 2): 3, (8, 0): 2, (8, 1): 2}
    x_off = {(4, t): 3 for t in [0, 1, 2]} | {(8, 2): 4, (8, 0): 5, (8, 1): 5}
    return nice_to_spin_glass(node=((t-t_off[size]) % 3, y-y_off[(size, t)], x-x_off[(size, t)], u, k), size=size)


if __name__ == "__main__":
    P8 = pd.read_csv(os.path.join(cwd, "..", "energies", "pegasus_random", "P8", "CBFM-P", "001_2000_445.csv"),
                     index_col=0)
    row_dict = P8.iloc[0].to_dict()
    del row_dict["energy"], row_dict["num_occurrences"]
    node_list = sorted([int(i) for i in list(row_dict.keys())])
    node_list = [dnx.pegasus_coordinates(16).linear_to_nice(i) for i in node_list]
    print(len(node_list))
    temp = dnx.pegasus_graph(8, nice_coordinates=True)
    temp2 = dnx.pegasus_graph(9, nice_coordinates=True, node_list=node_list)
    p = dnx.pegasus_graph(8, nice_coordinates=True)

    print(len(temp2.nodes))
    fig = plt.figure(figsize=(100, 100))
    #dnx.draw_pegasus(p,  with_labels=True)
    dnx.draw_pegasus(temp2, labels={node: advantage_6_1_to_spinglass(node, 8) for node in node_list}, with_labels=True)
    plt.savefig("P8_5.pdf")
    #plt.show()

# labels={node: nice_to_spin_glass(node, 8) for node in temp.nodes},
""" d = {}
    for i in range(5):
        x = (rn.randint(0,14), rn.randint(0,14), rn.randint(0,2), rn.randint(0,1), rn.randint(0,3))
        d[x] = x
    print(dattani_to_linear(d, 16))"""
