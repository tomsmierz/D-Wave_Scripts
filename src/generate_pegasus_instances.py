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

def find_map(size: int):
    source = dnx.pegasus_graph(size, nice_coordinates=True)
    # target = dnx.pegasus_graph(16, nice_coordinates=True)
    target = sampler.to_networkx_graph()
    mappings = [mapping for mapping in dnx.pegasus_sublattice_mappings(source, target)]
    mapping = None
    edges = None
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



def generate_pegasus_instances(number: int, size: int, out: str, distribution: str):
    pegasus = dnx.pegasus_graph(size, fabric_only=True, nice_coordinates=True)

    #pegasus = sampler.to_networkx_graph()

    sampler = DWaveSampler(solver="Advantage_system6.1")


    for i in tqdm(range(number), desc="generating pegasus instances: "):

        nodes = nx.get_node_attributes(pegasus, "linear_index")

        if distribution == "normal":
            couplings = {edge: rng.normal(0, 1) for edge in pegasus.edges}
            #couplings = normalize(couplings)
            bias = {node: rng.normal(0, 1) for node in pegasus.nodes}
        if distribution == "uniform":
            couplings = {edge: rng.uniform(-1, 1) for edge in pegasus.edges}
            bias = {node: rng.uniform(-4, 4) for node in pegasus.nodes}


        nx.set_node_attributes(pegasus, bias, "h")
        nx.set_edge_attributes(pegasus, couplings, "J")

        #name = f"00{i+1}"[-3:]
        name = f"{101 + i}"
        name = name + ".txt"

        with open(os.path.join(out, name), "w") as f:
            f.write("# \n")

            for node in pegasus.nodes.data("h"):
                f.write(str(nodes[node[0]] + 1) + " " + str(nodes[node[0]] + -11) + " " + str(node[1]) + "\n")
            for edge in pegasus.edges.data("J"):
                f.write(str(nodes[edge[0]] + 1) + " " + str(nodes[edge[1]] + -11) + " " + str(edge[2]) + "\n")



def generate_pegasus_map(number: int, size: int, out: str, mapping: int, wrong_edges = None):

    source = dnx.pegasus_graph(size, nice_coordinates=True)
    # target = dnx.pegasus_graph(16, nice_coordinates=True)
    target = sampler.to_networkx_graph()
    mappings = [mapping for mapping in dnx.pegasus_sublattice_mappings(source, target)]

    l = {node: mappings[mapping](node) for node in source.nodes()}
    nx.set_node_attributes(source, l, "mapping")

    em = nx.get_node_attributes(source, "mapping")
    for i in tqdm(range(number), desc="generating pegasus instances: "):

        #h = {node: rng.uniform(-4, 4) for node in em.values()}
        #J = {(em[edge[0]], em[edge[1]]): rng.uniform(-1, 1) for edge in source.edges()}

        h = {node: h_range() for node in em.values()}
        J = {(em[edge[0]], em[edge[1]]): J_range() for edge in source.edges()}


        name = f"00{i + 1}"[-3:]
        name = name + ".txt"

        with open(os.path.join(out, name), "w") as f:
            f.write("# \n")

            for node, value in h.items():
                f.write(str(node + 1) + " " + str(node + 1) + " " + str(value) + "\n")
            for edge, value in J.items():
                f.write(str(edge[0] + 1) + " " + str(edge[1] + 1) + " " + str(value) + "\n")
            if wrong_edges is not None:
                for edge in wrong_edges:
                    f.write(str(edge[0]) + " " + str(edge[1]) + " " + str(0) + "\n")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--size", type=int, choices=[i + 1 for i in range(16)], default=4,
                        help="Size of the chimera graph. Default is 4 (P4).")
    parser.add_argument("-N", "--number", type=int, default=1,
                        help="Number of instances to be generated. Default is 1, maximum 999.")
    parser.add_argument("-D", "--distribution", type=str, default="uniform", choices=["normal", "uniform"],
                        help="Distribution of biases and couplings")
    parser.add_argument("--path", type=str, default=path,
                        help="path to folder where generated instances will be located. Default is working directory")

    args = parser.parse_args()

    if args.number and args.number >= 1000:
        parser.error("Maximum number of generated instances is 999.")

    #generate_pegasus_instances(args.number, args.size, args.path, args.distribution)

    mapping, edges = find_map(args.size)
    generate_pegasus_map(args.number, args.size, args.path, mapping, edges)
#P2
"""
for i in range(6):
    graph.remove_node(26 + i)
    graph.remove_node(16 + i)

for i in range(2):
    graph.remove_node(44 + i)
    graph.remove_node(2 + i)
"""

#P4
"""
for i in range(18):
    pegasus.remove_node(150 + i)
    pegasus.remove_node(120 + i)

for i in range(6):
    pegasus.remove_node(6 + i)
    pegasus.remove_node(276 + i)

"""

#P8

"""
for i in range(42):
    pegasus.remove_node(686 + i)
    pegasus.remove_node(616 + i)

for i in range(14):
    pegasus.remove_node(14 + i)
    pegasus.remove_node(1316 + i)

"""


#P16

"""
    for i in range(90):
        pegasus.remove_node(2910 + i)
        pegasus.remove_node(2760 + i)

    for i in range(30):
        pegasus.remove_node(30 + i)
        pegasus.remove_node(5700 + i)

    for i in broken_nodes:
        pegasus.remove_node(i)

    for e in [(161, 5100), (2032, 4270), (641, 5118), (4832, 4833)]:
        pegasus.remove_edge(e[0], e[1])
"""