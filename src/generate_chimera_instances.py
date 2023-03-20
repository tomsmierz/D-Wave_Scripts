import dwave_networkx as dnx  # type: ignore
import argparse
import os
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
from math import inf
from typing import List, Optional, Callable, Tuple
from dwave.system import DWaveSampler


rng = np.random.default_rng()
path = os.getcwd()


def reverse_chimera_sublattice_mapping(mapping: Callable, source: nx.Graph) -> Callable:
    node_dict = {node: mapping(node) for node in source.nodes}
    reversed_dict = {value: key for key, value in node_dict.items()}

    def func(node: int) -> Tuple:
        return reversed_dict[node]

    return func


def chimera_to_spin_glass(q: Tuple, size: int) -> int:
    return dnx.chimera_coordinates(size).chimera_to_linear(q)


def find_map(source: nx.Graph, sampler: DWaveSampler) -> Tuple:
    target = sampler.to_networkx_graph()
    perfect = False
    mappings = [mapp for mapp in dnx.chimera_sublattice_mappings(source, target)]
    mapping = None
    best_missing_nodes = None
    best_missing_edges = None

    min_num_of_missing_edges = inf
    min_num_of_missing_nodes = inf
    best_imperfect_mapping = None

    for i in tqdm(range(len(mappings)), desc="Searching for a perfect mapping"):

        node_dict = {node: mappings[i](node) for node in source.nodes()}
        edge_dict = {(v, w): (node_dict[v], node_dict[w]) for v, w in source.edges()}

        if all(node in target.nodes() for node in node_dict.values()) and \
                all(edge in target.edges() for edge in edge_dict.values()):

            mapping = mappings[i]
            print("\nPerfect map found")
            perfect = True
            break
        else:
            mapped_source_nodes_set = set(node_dict.values())
            mapped_source_edges_set = set([frozenset(edge) for edge in edge_dict.values()])

            real_nodes_set = set(sampler.nodelist)
            real_edges_set = set([frozenset(edge) for edge in sampler.edgelist])

            missing_nodes = list(mapped_source_nodes_set - real_nodes_set)
            missing_edges = list(mapped_source_edges_set - real_edges_set)
            num_of_missing_nodes = len(missing_nodes)
            num_of_missing_edges = len(missing_edges)

            if num_of_missing_nodes <= min_num_of_missing_nodes and num_of_missing_edges <= min_num_of_missing_edges:
                min_num_of_missing_nodes = num_of_missing_nodes
                min_num_of_missing_edges = num_of_missing_edges
                best_missing_nodes = missing_nodes
                best_missing_edges = missing_edges
                best_imperfect_mapping = mappings[i]

    if mapping is None:
        mapping = best_imperfect_mapping
        print(f"\nNo perfect map found. Returning imperfect map with {min_num_of_missing_nodes} missing nodes"
              f" and {min_num_of_missing_edges} missing edges")

    if mapping is None:
        raise RuntimeError("No map found. Possible problem with the source or the target graph")

    return mapping, perfect, best_missing_nodes, best_missing_edges


def generate_chimera_instances(number: int, size: int, output_path: str, output_types: List[str],
                               category: str, device: Optional[str] = None,
                               name: Optional[str] = None) -> None:
    source = dnx.chimera_graph(size, coordinates=True)
    username = True if name is not None else False

    if device is not None:
        if device not in ["DW_2000Q_6"]:
            raise AssertionError("Device should be set to \"DW_2000Q_6\" or None")
        if size > 16:
            raise AssertionError("Maximum size for working chimera device is 16")

        sampler = DWaveSampler(solver=device)
        mapping, perfect_mapping, missing_nodes, missing_edges = find_map(source, sampler)
        if perfect_mapping:
            graph = source
        else:
            reverse_mapping = reverse_chimera_sublattice_mapping(mapping, source)
            graph = source
            for node in missing_nodes:
                graph.remove_node(reverse_mapping(node))

            for edge in missing_edges:
                edge = tuple(edge)
                edge = (reverse_mapping(edge[0]), reverse_mapping(edge[1]))
                reversed_edge = (edge[1], edge[0])
                if edge in graph.edges():
                    graph.remove_edge(edge[0], edge[1])

                if reversed_edge in graph.edges():
                    graph.remove_edge(reversed_edge[0], reversed_edge[1])

    else:
        graph = source

    for i in tqdm(range(number), desc=f"generating Chimera instances size = {size}, category={category}: "):

        if category == "RAU":
            bias = {node: rng.uniform(-0.1, 0.1) for node in graph.nodes()}
            couplings = {edge: rng.uniform(-1, 1) for edge in graph.edges()}
        elif category == "RCO":
            bias = {node: 0 for node in graph.nodes()}
            couplings = {edge: rng.uniform(-1, 1) for edge in graph.edges()}
        elif category == "AC3":
            bias = {node: rng.uniform(-1 / 9, 1 / 9) for node in graph.nodes()}
            couplings = {edge: rng.uniform(-1 / 3, 1 / 3) if edge[0][1:3] == edge[1][1:3]
            else rng.uniform(-1, 1) for edge in graph.edges()}
        else:
            raise ValueError(f"Category {category} is not a valid choice. It should be \"RAU\", \"RCO\" or \"AC3\"")

        if username:
            name = name + f"{i + 1}"
        else:
            name = f"{i + 1}"

        for output_type in output_types:
            if output_type == "SpinGlass":  # renumeration is very cheap, and we can afford to do this every loop

                couplings_sg = {(chimera_to_spin_glass(edge[0], size) + 1, chimera_to_spin_glass(edge[1], size) + 1):
                                value for edge, value in couplings.items()}
                couplings_sg = dict(sorted(couplings_sg.items()))

                bias_sg = {chimera_to_spin_glass(node, size) + 1: value for node, value in bias.items()}
                bias_sg = dict(sorted(bias_sg.items()))

                output_name = name + "_sg.txt"

                with open(os.path.join(output_path, output_name), "w") as f:
                    f.write("# \n")
                    for node, value in bias_sg.items():
                        f.write(str(node) + " " + str(node) + " " + str(value) + "\n")
                    for edge, value in couplings_sg.items():
                        f.write(str(edge[0]) + " " + str(edge[1]) + " " + str(value) + "\n")

            elif output_type == "DWave":

                if device is not None:

                    couplings_dv = {(mapping(edge[0]), mapping(edge[1])): value for edge, value in couplings.items()}
                    bias_dv = {mapping(node): value for node, value in bias.items()}
                    data = [bias_dv, couplings_dv]
                else:
                    data = [bias, couplings]

                output_name = name + "_dv.pkl"
                with open(os.path.join(output_path, output_name), "wb") as f:
                    pickle.dump(data, f)

            elif output_type == "MatrixMarket":
                raise NotImplementedError("MatrixMarket output not implemented yet")

            else:
                raise ValueError(f"{output_type} is not valid output type. It should be \"SpinGlass\", \"DWave\", "
                                 f"or \"MatrixMarket\"")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-S", "--size", type=int, default=4,
                        help="Size of the Chimera graph. Default is 4 (C4).")
    parser.add_argument("-N", "--number", type=int, default=1,
                        help="Number of instances to be generated. Default is 1.")
    parser.add_argument("-C", "--category", type=str, default="RAU", choices=["RAU", "RCO", "AC3"],
                        help="Category of generated instances. RAU - random uniform, RCO - random couplings only, "
                             "AC3 - anti-cluster")
    parser.add_argument("-P", "--path", type=str, default=path,
                        help="path to folder where generated instances will be located. "
                             "Default is working directory")
    parser.add_argument("-T", "--types", type=str, default=["SpinGlass"],
                        choices=["SpinGlass", "DWave", "MatrixMarket"], nargs="*")
    parser.add_argument("-D", "--device", default=None,
                        choices=["DW_2000Q_6", None],
                        help="Map instance info physical D-Wave's device. Input None for no Mapping")

    args = parser.parse_args()

    generate_chimera_instances(args.number, args.size, args.path, args.types, args.category, device=args.device)




