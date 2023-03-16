import networkx as nx
import dwave_networkx as dnx
import pandas as pd
import random as rn

from src.utils import load_pegasus, load_pegasus_tuple
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
        t = (tmp[0], tmp[1]-s[1], tmp[2]-s[2], tmp[3], tmp[4])
        bl = [x >= 0 for x in t]
        if not all(bl):
            print(t)
            raise ValueError("t")
        h_tuple[node] = (tmp[0], tmp[1]-s[1], tmp[2]-s[2], tmp[3], tmp[4])

    return h_tuple


def tuple_to_linear(h_tuple: Dict, size: int) -> Dict:
    h_linear = {}
    for value in h_tuple.keys():
        if value[3] == 1:
            x = 4 + value[4] + 1
        else:
            x = abs(value[4] - 3) + 1
        y = abs(value[1] - (size-2))

        h_linear[value] = 8 * value[0] + 24 * value[2] + 24 * (size - 1) * y + x
            #24 * (size - 1) * value[0] + 24 * value[1] + 8 * value[2] + 4 * value[3] + value[4] + 1

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
        h_linear[key] = 24*(size-1) * value[0] + 24 * value[1] + 8 * value[2] + 4 * value[3] + value[4] + 1

    return h_linear

def dattani_to_linear_2(h_dattani: Dict, size: int) -> Dict:
    h_linear = {}
    for key, value in h_dattani.items():
        h_linear[key] = 24*(size-1) * value[0] + 24 * value[2] + 8 * value[1] + 4 * value[3] + value[4] + 1

    return h_linear


def renumerate(instance_path: str, name: str, size: int):
    h, J = load_pegasus_tuple(instance_path, name)
    rn = {}
    h_rn = {}
    J_rn = {}
    i = 1
    for key in h.keys():
        #rn[key] = dattani_to_linear_2(tuple_to_dattani(machine_to_5_tuple(h)), size)[key]
        rn[key] = tuple_to_linear(h, size)[key]
        #rn[key] = machine_to_5_tuple(h)[key]
    for key, value in h.items():
        h_rn[rn[key]] = value
    for key, value in J.items():
        J_rn[(rn[key[0]], rn[key[1]])] = value

    name = "r_" + name + "_4" + ".txt"

    with open(os.path.join(f"/home/tsmierzchalski/instances/renumerated/P{size}", name), "w") as f:
        f.write("# \n")

        h_rn_sorted = {k: h_rn[k] for k in sorted(h_rn)}
        J_rn_sorted = {k: J_rn[k] for k in sorted(J_rn)}

        for node, value in h_rn_sorted.items():
            f.write(str(node) + " " + str(node) + " " + str(value) + "\n")
        for edge, value in J_rn_sorted.items():
            f.write(str(edge[0]) + " " + str(edge[1]) + " " + str(value) + "\n")


if __name__ == "__main__":

    for i in tqdm(range(10)):
        name = f"{i+1}"
        name = name + "_nd_original"
        renumerate("/home/tsmierzchalski/instances/P8", name, 8)


""" d = {}
    for i in range(5):
        x = (rn.randint(0,14), rn.randint(0,14), rn.randint(0,2), rn.randint(0,1), rn.randint(0,3))
        d[x] = x
    print(dattani_to_linear(d, 16))"""
