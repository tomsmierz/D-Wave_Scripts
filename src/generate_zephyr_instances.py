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


def generate_zephyr_instance(number: int, size: int, out: str, distribution: str):

    zephyr = dnx.zephyr_graph(size)

    for i in tqdm(range(number), desc="generating zephyr instances: "):

        couplings = {edge: rng.uniform(-1, 1) for edge in zephyr.edges}
        bias = {node: rng.uniform(-4, 4) for node in zephyr.nodes}

        nx.set_node_attributes(zephyr, bias, "h")
        nx.set_edge_attributes(zephyr, couplings, "J")

        name = f"Z{size}"
        # name = f"00{i + 1}"[-3:]
        # name = f"{101 + i}"
        name = name + ".txt"

        with open(os.path.join(out, name), "w") as f:
            f.write("# \n")
            for node in zephyr.nodes.data("h"):
                f.write(str(node[0] + 1) + " " + str(node[0] + 1) + " " + str(node[1]) + "\n")
            for edge in zephyr.edges.data("J"):
                f.write(str(edge[0] + 1) + " " + str(edge[1] + 1) + " " + str(edge[2]) + "\n")

if __name__ == "__main__":

    generate_zephyr_instance(1, 3, "/home/tsmierzchalski/pycharm_projects/D-Wave_Scripts/instances/zephyr", "normal")
