import copy
import os
import json
import random
import time

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dwave_networkx as dnx
import networkx as nx

from collections import namedtuple
from renumeration import advantage_6_1_to_spinglass, advantage_6_1_to_spinglass_int
from scipy.spatial.distance import hamming
from typing import Optional, Union
from tqdm import tqdm

vector = Union[np.ndarray, list]
script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
root = os.path.dirname(script_dir)
instance_path = os.path.join(root, "instances", "pathological", "square_star_5x5.txt")


def xor(v1: vector, v2: vector) -> vector:
    assert len(v1) == len(v2)
    return [0 if v1[i] == v2[i] else 1 for i in range(len(v1))]



def create_spin_glass_peps_graph(file: str) -> nx.Graph:
    df = pd.read_csv(file, sep=" ", names=["v", "w", "J"], comment="#")
    edges = []
    nodes = []
    for row in df.itertuples():
        if row.v != row.w and row.J != 0:
            edges.append((row.v,row.w))
        elif row.v == row.w:
            nodes.append(row.v)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def connected_hamming_dist(state1: vector, state2: vector, graph) -> int:
    xor_state = xor(state1, state2)
    nodes = []
    for node in graph.nodes:
        if xor_state[node-1]:
            nodes.append(node)
    # begin = time.time()
    subgraph = nx.subgraph(graph, nodes)
    # end1 = time.time()
    largest_cc = max(nx.connected_components(subgraph), key=len)
    # end2 = time.time()
    # print("time subgraph: ", end1 - begin, " time largest connected components: ", end2 - end1)
    return len(largest_cc)


if __name__ == '__main__':
    graph = create_spin_glass_peps_graph(instance_path)
    v1 = [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1]
    v2 = [-1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1]
    dist = connected_hamming_dist(v1, v2, graph)
    print(dist)