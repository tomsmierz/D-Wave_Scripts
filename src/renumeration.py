import networkx as nx
import dwave_networkx as dnx
import pandas as pd

from src.pegasus import get_pegasus
from tqdm import tqdm
from typing import Dict

import os

cwd = os.getcwd()


def machine_to_5_tuple(h: Dict) -> Dict:
    h_tuple = {}

    s = list(h.keys())[0]
    s = dnx.pegasus_coordinates(16).linear_to_nice(s)
    for node in h.keys():
        h_tuple[node] = dnx.pegasus_coordinates(16).linear_to_nice(node)
        tmp = h_tuple[node]
        h_tuple[node] = (tmp[0], tmp[1]-s[1], tmp[2]-s[2], tmp[3], tmp[4])

    return h_tuple


def tuple_to_dattani(h_tuple: Dict) -> Dict:
    h_dattani = {}

    for key, value in h_tuple.items():
        tmp = value
        h_dattani[key] = (tmp[2], tmp[1], tmp[0], tmp[3], tmp[4])

    return h_dattani


def dattani_to_linear(h_dattani: Dict, size: int) -> Dict:
    h_linear = {}
    for key, value in h_dattani.items():
        h_linear[key] = 24*(size-1) * value[0] + 24 * value[1] + 8 * value[2] + 4 * value[3] + value[4] + 1

    return h_linear


def renumerate(instance_path: str, name: str, size: int):
    h, J = get_pegasus(instance_path, name)
    rn = {}
    h_rn = {}
    J_rn = {}
    i = 1
    for key in h.keys():
        rn[key] = dattani_to_linear(tuple_to_dattani(machine_to_5_tuple(h)), size)[key]



    for key, value in h.items():
        h_rn[rn[key]] = value
    for key, value in J.items():
        J_rn[(rn[key[0]], rn[key[1]])] = value

    name = name + ".txt"

    with open(os.path.join(f"/home/tsmierzchalski/pycharm_projects/D-Wave_Scripts/instances_renumerated/uniform/P{size}", name), "w") as f:
        f.write("# \n")

        h_rn_sorted = {k: h_rn[k] for k in sorted(h_rn)}
        J_rn_sorted = {k: J_rn[k] for k in sorted(J_rn)}

        for node, value in h_rn_sorted.items():
            f.write(str(node) + " " + str(node) + " " + str(value) + "\n")
        for edge, value in J_rn_sorted.items():
            f.write(str(edge[0]) + " " + str(edge[1]) + " " + str(value) + "\n")


if __name__ == "__main__":

    x = nx.complete_bipartite_graph([],[])

    """
    for i in tqdm(range(100)):
        name = f"00{i+1}"[-3:]
        renumerate("/home/tsmierzchalski/pycharm_projects/D-Wave_Scripts/instances/uniform/P16", name, 16)

"""