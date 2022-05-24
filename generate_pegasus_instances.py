import dwave_networkx as dnx  # type: ignore
import networkx as nx
import argparse
import numpy as np
import os

from typing import Dict

rng = np.random.default_rng()
path = os.getcwd()


def normalize(d: Dict) -> Dict:
    max_value = max(d.values())
    normalized = {}
    for key in d.keys():
        normalized[key] = d[key]/max_value
    return normalized


def generate_pegasus_instances(number: int, size: int, out: str, distribution: str):
    pegasus = dnx.pegasus_graph(size, data=False, fabric_only=False)

    for i in range(number):

        if distribution == "normal":
            couplings = {edge: rng.normal(0, 1) for edge in pegasus.edges}
            #couplings = normalize(couplings)
            bias = {node: rng.normal(0, 1) for node in pegasus.nodes}

        nx.set_node_attributes(pegasus, bias, "h")
        nx.set_edge_attributes(pegasus, couplings, "J")

        name = f"00{i+1}"[-3:]
        name = name + ".txt"

        with open(os.path.join(out, name), "w") as f:
            f.write("# \n")

            for node in pegasus.nodes.data("h"):
                f.write(str(node[0] + 1) + " " + str(node[0] + 1) + " " + str(node[1]) + "\n")
            for edge in pegasus.edges.data("J"):
                f.write(str(edge[0] + 1) + " " + str(edge[1] + 1) + " " + str(edge[2]) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--size", type=int, choices=[i + 1 for i in range(16)], default=4,
                        help="Size of the chimera graph. Default is 4 (P4).")
    parser.add_argument("-N", "--number", type=int, default=1,
                        help="Number of instances to be generated. Default is 1, maximum 999.")
    parser.add_argument("-D", "--distribution", type=str, default="normal", choices=["normal", "uniform"],
                        help="Distribution of biases and couplings")
    parser.add_argument("--path", type=str, default=path,
                        help="path to folder where generated instances will be located. Default is working directory")

    args = parser.parse_args()

    if args.number and args.number >= 1000:
        parser.error("Maximum number of generated instances is 999.")

    generate_pegasus_instances(args.number, args.size, args.path, args.distribution)
