import dwave_networkx as dnx  # type: ignore
import argparse
import numpy as np
import os

from typing import Tuple
from dwave.system import DWaveSampler
from tqdm import tqdm

rng = np.random.default_rng()
path = os.getcwd()

sampler = DWaveSampler(solver="Advantage_system6.1")


def tuple_to_spin_glass(node: Tuple, size: int) -> int:
    t, y, x, u, k = node
    if u == 1:
        a = 4 + k + 1
    else:
        a = abs(k - 3) + 1
    b = abs(y - (size - 2))

    spin_glas_linear = 8 * t + 24 * x + 24 * (size - 1) * b + a
    # 24 * (size - 1) * value[0] + 24 * value[1] + 8 * value[2] + 4 * value[3] + value[4] + 1
    return spin_glas_linear


def rn(s):
    return dnx.pegasus_coordinates(16).linear_to_nice(s)


def generate_pegasus_instances(number: int, size: int, output_path: str, output_type: str,
                               category: str, no_diagonal: bool):

    source = dnx.pegasus_graph(size, nice_coordinates=True)

    if no_diagonal:
        for y in range(size - 1):
            for x in range(1, size):
                for i in range(4):
                    h = (0, y, x, 0, i)
                    h1 = (2, y + 1, x - 1, 1, 0)
                    h2 = (2, y + 1, x - 1, 1, 1)
                    v = (0, y, x, 1, i)
                    v1 = (2, y + 1, x - 1, 0, 2)
                    v2 = (2, y + 1, x - 1, 0, 3)
                    for e in [h1, h2]:
                        if source.has_edge(h, e):
                            source.remove_edge(h, e)
                    for e in [v1, v2]:
                        if source.has_edge(v, e):
                            source.remove_edge(v, e)

    if output_type == "SpinGlass":
        nodes = sorted([tuple_to_spin_glass(node, size) for node in source.nodes])
        edges = sorted([(tuple_to_spin_glass(node1, size), tuple_to_spin_glass(node2, size))
                 for (node1, node2) in source.edges])
    else:
        raise NotImplementedError("Other formats of output not implemented yet")

    for i in tqdm(range(number), desc="generating pegasus instances: "):

        if category == "RAU":
            couplings = {edge: rng.uniform(-1, 1) for edge in edges}
            bias = {node: rng.uniform(-0.1, 0.1) for node in nodes}
        else:
            raise NotImplementedError("Categories other than RAU not implemented yet")

        name = f"00{i+1}"[-3:]
        name = name + ".txt"

        with open(os.path.join(output_path, name), "w") as f:
            f.write("# \n")
            for node, value in bias.items():
                f.write(str(node) + " " + str(node) + " " + str(value) + "\n")
            for edge, value in couplings.items():
                f.write(str(edge[0]) + " " + str(edge[1]) + " " + str(value) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-S", "--size", type=int, default=4,
                        help="Size of the pegasus graph. Minimum 2. Default is 4 (P4).")
    parser.add_argument("-N", "--number", type=int, default=1,
                        help="Number of instances to be generated. Default is 1, maximum 999.")
    parser.add_argument("-C", "--category", type=str, default="RAU", choices=["RAU", "RAC", "AC3"],
                        help="Category of generated instances. RAU - random uniform, RAC - random couplings only, "
                             "AC3 - anti-cluster")
    parser.add_argument("--path", type=str, default=path,
                        help="path to folder where generated instances will be located. Default is working directory")

    args = parser.parse_args()

    if args.number and args.number >= 1000:
        parser.error("Maximum number of generated instances is 999.")
    if args.size and args.size <2:
        parser.error("Minimum size of pegasus instance is 2")

    generate_pegasus_instances(args.number, args.size, args.path, "SpinGlass", args.category, no_diagonal=False)

    # mapping, edges = find_map(args.size)
    # generate_pegasus_map(args.number, args.size, args.path, mapping, edges)
