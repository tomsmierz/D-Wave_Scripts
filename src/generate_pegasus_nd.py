import dwave_networkx as dnx  # type: ignore
import networkx as nx
import argparse
import numpy as np
import os

from typing import Dict
from dwave.system import DWaveSampler
from dwave.cloud import Client
from tqdm import tqdm

rng = np.random.default_rng()
path = os.getcwd()

sampler = DWaveSampler(solver="Advantage_system6.1")


def generate_pegasus_instances(number: int, size: int, out: str, distribution: str):
    pegasus = dnx.pegasus_graph(size, nice_coordinates=True)
    sampler = DWaveSampler(solver="Advantage_system6.1")
    #pegasus = sampler.to_networkx_graph()

    for i in tqdm(range(number), desc="generating pegasus instances: "):

        nodes = nx.get_node_attributes(pegasus, "linear_index")

        if distribution == "normal":
            pass
            #couplings = {edge: J_range() for edge in pegasus.edges}
            #couplings = normalize(couplings)
            #bias = {node: h_range() for node in pegasus.nodes}

        if distribution == "uniform":
            couplings = {edge: rng.uniform(-1, 1) for edge in pegasus.edges}
            bias = {node: rng.uniform(-4, 4) for node in pegasus.nodes}


        nx.set_node_attributes(pegasus, bias, "h")
        nx.set_edge_attributes(pegasus, couplings, "J")

        name = f"00{i+1}"[-3:]
        #name = f"{101 + i}"
        name = name + ".txt"

        with open(os.path.join(out, name), "w") as f:
            f.write("# \n")
            for node in pegasus.nodes.data("h"):
                f.write(str(nodes[node[0]] - 11) + " " + str(nodes[node[0]] - 11) + " " + str(node[1]) + "\n")
            for edge in pegasus.edges.data("J"):
                f.write(str(nodes[edge[0]] - 11) + " " + str(nodes[edge[1]] - 11) + " " + str(edge[2]) + "\n")


if __name__ == "__main__":
    generate_pegasus_instances(1, 4, "/home/tsmierzchalski/instances/P4","uniform")