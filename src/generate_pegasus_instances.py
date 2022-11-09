import dwave_networkx as dnx  # type: ignore
import networkx as nx
import argparse
import numpy as np
import os

from typing import Dict, Tuple, Union
from dwave.system import DWaveSampler
from dwave.cloud import Client
from tqdm import tqdm

rng = np.random.default_rng()
path = os.getcwd()

sampler = DWaveSampler(solver="Advantage_system6.1")


def normalize(d: Dict) -> Dict:
    max_value = max(d.values())
    normalized = {}
    for key in d.keys():
        normalized[key] = d[key]/max_value
    return normalized


def h_range():
    low = -4.0
    high = 4.0
    value = rng.normal(0, 1)
    r = value
    if value > high:
        r = high
    elif value < low:
        r = low
    return r


def J_range():
    low = -1.0
    high = 1.0
    value = rng.normal(0, 0.5)
    r = value
    if value > high:
        r = high
    elif value < low:
        r = low
    return r


def rn(s):
    return dnx.pegasus_coordinates(16).linear_to_nice(s)


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



def find_map(size: int):
    source = dnx.pegasus_graph(size, nice_coordinates=True)
    # target = dnx.pegasus_graph(16, nice_coordinates=True)
    target = sampler.to_networkx_graph()

    mappings = [mapping for mapping in dnx.pegasus_sublattice_mappings(source, target)]
    mapping = None
    edges = None
    if size<8:
        for i in tqdm(range(len(mappings)), desc="Searching for perfect mapping"):

            l = {node: mappings[i](node) for node in source.nodes()}
            nx.set_node_attributes(source, l, "mapping")

            em = nx.get_node_attributes(source, "mapping")

            h = {node: rng.uniform(-4, 4) for node in em.values()}
            # print(all(node in sampler.nodelist for node in h.keys()))
            J = {(em[edge[0]], em[edge[1]]): rng.uniform(-1, 1) for edge in source.edges()}
            # print(all(edge in sampler.edgelist for edge in J.keys()))
            if all(node in sampler.nodelist for node in h.keys()) and all(edge in sampler.edgelist for edge in J.keys()):
                mapping = i
                break

    if mapping is None:
        proposed = {}
        for i in tqdm(range(len(mappings)), desc="Searching for imperfect mapping"):

            l = {node: mappings[i](node) for node in source.nodes()}
            nx.set_node_attributes(source, l, "mapping")

            em = nx.get_node_attributes(source, "mapping")

            h = {node: rng.uniform(-4, 4) for node in em.values()}
            # print(all(node in sampler.nodelist for node in h.keys()))
            J = {(em[edge[0]], em[edge[1]]): rng.uniform(-1, 1) for edge in source.edges()}
            # print(all(edge in sampler.edgelist for edge in J.keys()))
            if all(node in sampler.nodelist for node in h.keys()):
                proposed[i] = list(set(J.keys()) - set(sampler.edgelist))

        mapping = list(proposed.keys())[0]
        edges = proposed[mapping]
    return mapping, edges


def generate_pegasus_instances_old(number: int, size: int, out: str, distribution: str):
    pegasus = dnx.pegasus_graph(size, nice_coordinates=True)
    sampler = DWaveSampler(solver="Advantage_system6.1")
    #pegasus = sampler.to_networkx_graph()

    for i in tqdm(range(number), desc="generating pegasus instances: "):

        nodes = nx.get_node_attributes(pegasus, "linear_index")

        if distribution == "normal":
            couplings = {edge: J_range() for edge in pegasus.edges}
            #couplings = normalize(couplings)
            bias = {node: h_range() for node in pegasus.nodes}
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
                f.write(str(node[0] + 1) + " " + str(node[0] + 1) + " " + str(node[1]) + "\n")
            for edge in pegasus.edges.data("J"):
                f.write(str(edge[0] + 1) + " " + str(edge[1] + 1) + " " + str(edge[2]) + "\n")
"""
            for node in pegasus.nodes.data("h"):
                f.write(str(nodes[node[0]] + 1) + " " + str(nodes[node[0]] + 1) + " " + str(node[1]) + "\n")
            for edge in pegasus.edges.data("J"):
                f.write(str(nodes[edge[0]] + 1) + " " + str(nodes[edge[1]] + 1) + " " + str(edge[2]) + "\n")
"""


def generate_pegasus_map(number: int, size: int, out: str, mapping: int, wrong_edges = None):

    source = dnx.pegasus_graph(size, nice_coordinates=True)
    # target = dnx.pegasus_graph(16, nice_coordinates=True)
    target = sampler.to_networkx_graph()

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

    mappings = [mapping for mapping in dnx.pegasus_sublattice_mappings(source, target)]

    l = {node: mappings[mapping](node) for node in source.nodes()}
    nx.set_node_attributes(source, l, "mapping")

    em = nx.get_node_attributes(source, "mapping")
    for i in tqdm(range(number), desc="generating pegasus instances: "):

        h = {item: rng.uniform(-4, 4) for item in em.items()}
        J = {(edge, (em[edge[0]], em[edge[1]])): rng.uniform(-1, 1) for edge in source.edges()}

        #h = {node: h_range() for node in em.values()}
        #J = {(em[edge[0]], em[edge[1]]): J_range() for edge in source.edges()}

        del J[(((2, 4, 6, 0, 3), (2, 4, 6, 1, 0)), (2032, 4270))]
        name_basic = f"00{i + 1}"[-3:]
        name = name_basic + "_nd" + ".txt"
        name_orig = name_basic + "_nd_original.txt"
        with open(os.path.join(out, name), "w") as f:
            f.write("# \n")

            for node, value in h.items():
                f.write(str(node[1] + 1) + " " + str(node[1] + 1) + " " + str(value) + "\n")
            for edge, value in J.items():
                f.write(str(edge[1][0] + 1) + " " + str(edge[1][1] + 1) + " " + str(value) + "\n")
            #if wrong_edges is not None:
            #    for edge in wrong_edges:
            #        f.write(str(edge[0] + 1) + " " + str(edge[1] + 1) + " " + str(0) + "\n")

        with open(os.path.join(out, name_orig), "w") as f:
            f.write("# \n")

            for node, value in h.items():
                f.write(str(node[0]) + ";" + str(node[0]) + ";" + str(value) + "\n")
            for edge, value in J.items():
                f.write(str(edge[0][0]) + ";" + str(edge[0][1]) + ";" + str(value) + "\n")


def generate_pegasus_nd_instances(number: int, size: int, out: str, mapping: int, wrong_edges = None):

    source = dnx.pegasus_graph(size, nice_coordinates=True)
    target = sampler.to_networkx_graph()

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

    mappings = [mapping for mapping in dnx.pegasus_sublattice_mappings(source, target)]

    l = {node: mappings[mapping](node) for node in source.nodes()}
    nx.set_node_attributes(source, l, "mapping")

    em = nx.get_node_attributes(source, "mapping")
    for i in tqdm(range(number), desc="generating pegasus instances: "):

        h = {item: rng.uniform(-4, 4) for item in em.items()}
        J = {(edge, (em[edge[0]], em[edge[1]])): rng.uniform(-1, 1) for edge in source.edges()}

        #h = {node: h_range() for node in em.values()}
        #J = {(em[edge[0]], em[edge[1]]): J_range() for edge in source.edges()}

        del J[(((2, 4, 6, 0, 3), (2, 4, 6, 1, 0)), (2032, 4270))]
        name_basic = f"00{i + 1}"[-3:]
        name = name_basic + "_nd" + ".txt"
        name_orig = name_basic + "_nd_original.txt"
        with open(os.path.join(out, name), "w") as f:
            f.write("# \n")

            for node, value in h.items():
                f.write(str(node[1] + 1) + " " + str(node[1] + 1) + " " + str(value) + "\n")
            for edge, value in J.items():
                f.write(str(edge[1][0] + 1) + " " + str(edge[1][1] + 1) + " " + str(value) + "\n")
            #if wrong_edges is not None:
            #    for edge in wrong_edges:
            #        f.write(str(edge[0] + 1) + " " + str(edge[1] + 1) + " " + str(0) + "\n")

        with open(os.path.join(out, name_orig), "w") as f:
            f.write("# \n")

            for node, value in h.items():
                f.write(str(node[0]) + ";" + str(node[0]) + ";" + str(value) + "\n")
            for edge, value in J.items():
                f.write(str(edge[0][0]) + ";" + str(edge[0][1]) + ";" + str(value) + "\n")


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

    elif output_type == "Original":
        nodes = source.nodes
        edges = source.edges
    else:
        raise NotImplementedError("Other formats of output not implemented yet")

    for i in tqdm(range(number), desc="generating pegasus instances: "):

        if category == "RAU":
            couplings = {edge: rng.uniform(-1, 1) for edge in edges}
            bias = {node: rng.uniform(-0.1, 0.1) for node in nodes}
        else:
            raise NotImplementedError("Categories other than RAU not implemented yet")

        name = f"00{i + 1}"[-3:]
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
                        help="Number of instances to be generated. Default is 1.")
    parser.add_argument("-C", "--category", type=str, default="RAU", choices=["RAU", "RAC", "AC3"],
                        help="Category of generated instances. RAU - random uniform, RAC - random couplings only, "
                             "AC3 - anti-cluster")
    parser.add_argument("-P", "--path", type=str, default=path,
                        help="path to folder where generated instances will be located. "
                             "Default is working directory")
    parser.add_argument("-T", "--type", type=str, default="Device",
                        choices=["SpinGlass", "Device", "Original", "MatrixMarket"], nargs="*")
    parser.add_argument("--no_diag", type=bool, default=False,
                        help="Generate pegasus instances with or without \"diagonal\" connections")

    args = parser.parse_args()

    if args.size and args.size < 2:
        parser.error("Minimum size of pegasus instance is 2")

    for output_type in args.type:
        generate_pegasus_instances(args.number, args.size, args.path, output_type,
                                   args.category, no_diagonal=args.no_diag)

#generate_pegasus_instances(args.number, args.size, args.path, args.distribution)

# mapping, edges = find_map(args.size)
# generate_pegasus_map(args.number, args.size, args.path, mapping, edges)
