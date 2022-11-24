import dwave_networkx as dnx  # type: ignore
import networkx as nx
import argparse
import numpy as np
import os
import pickle

from typing import Tuple, Union, Optional, List, Callable
from dwave.system import DWaveSampler
from tqdm import tqdm
from math import inf

rng = np.random.default_rng()
path = os.getcwd()


def rn(s: int) -> Tuple:
    return dnx.pegasus_coordinates(16).linear_to_nice(s)


def tuple_to_spin_glass(node: Tuple, size: int) -> int:
    t, y, x, u, k = node
    if u == 1:
        a = 4 + k + 1
    else:
        a = abs(k - 3) + 1
    b = abs(y - (size - 2))

    spin_glas_linear = 8 * t + 24 * x + 24 * (size - 1) * b + a
    return spin_glas_linear


def find_map(source: nx.Graph, target: nx.Graph, sampler: DWaveSampler) -> \
        Tuple[Callable, Union[List, None], Union[List, None]]:

    mappings = [mapp for mapp in dnx.pegasus_sublattice_mappings(source, target)]
    mapping = None
    missing_edges = None
    missing_nodes = None

    min_num_of_missing_edges = inf
    min_num_of_missing_nodes = inf
    best_imperfect_mapping = None

    for i in tqdm(range(len(mappings)), desc="Searching for a perfect mapping"):

        node_dict = {node: mappings[i](node) for node in source.nodes()}
        edge_dict = {(v, w): (node_dict[v], node_dict[w]) for v, w in source.edges()}

        if all(node in target.nodes() for node in node_dict.values()) and \
                all(edge in target.edges() for edge in edge_dict.values()):

            mapping = i
            print("\n Perfect map found")
            break
        else:
            mapped_source_nodes_set = set(node_dict.values())
            mapped_source_edges_set = set([set(edge) for edge in edge_dict.values()])

            real_nodes_set = set(sampler.nodelist)
            real_edges_set = set([set(edge) for edge in sampler.edgelist])

            missing_nodes = list(mapped_source_nodes_set - real_nodes_set)
            missing_edges = list(mapped_source_edges_set - real_edges_set)
            num_of_missing_nodes = len(missing_nodes)
            num_of_missing_edges = len(missing_edges)

            if num_of_missing_nodes <= min_num_of_missing_nodes and num_of_missing_edges <= min_num_of_missing_edges:
                min_num_of_missing_nodes = num_of_missing_nodes
                min_num_of_missing_edges = num_of_missing_edges
                best_imperfect_mapping = i

    if mapping is None:
        mapping = best_imperfect_mapping
        print(f"\n No perfect map found. Returning imperfect map with {num_of_missing_nodes} missing nodes"
              f" and {num_of_missing_edges} missing edges")

    if mapping is None:
        raise RuntimeError("No map found. Possible problem with the source or the target graph")

    return mapping, missing_nodes, missing_edges


def generate_pegasus_instances(number: int, size: int, output_path: str, output_types: List[str],
                               category: str, diagonal: bool = True, device: Optional[str] = None,
                               name: Optional[str] = None) -> None:

    source = dnx.pegasus_graph(size, nice_coordinates=True)

    if device is not None:
        if device not in ["Advantage_system4.1", "Advantage_system5.2", "Advantage_system6.1"]:
            raise AssertionError("Device should be set to \"Advantage_system4.1\", \"Advantage_system5.2\", "
                                 "\"Advantage_system6.1\" or None")
        sampler = DWaveSampler(solver=device)
        target = dnx.pegasus_graph(16, nice_coordinates=True)
        mapping, missing_nodes, missing_edges = find_map(source, target, sampler)

    if not diagonal:
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

    for i in tqdm(range(number), desc="generating pegasus instances: "):

        if category == "RAU":
            couplings = {edge: rng.uniform(-1, 1) for edge in source.edges()}
            bias = {node: rng.uniform(-0.1, 0.1) for node in source.nodes()}
        else:
            raise NotImplementedError("Categories other than RAU not implemented yet")
        if name is None:
            name = f"00{i + 1}"[-3:]

        for output_type in output_types:
            if output_type == "SpinGlass":  # renumeration is very cheap, and we can afford to do this every loop

                couplings_sg = {(tuple_to_spin_glass(edge[0], size), tuple_to_spin_glass(edge[1], size)): value
                                for edge, value in couplings.items()}
                couplings_sg = dict(sorted(couplings_sg.items()))

                bias_sg = {tuple_to_spin_glass(node, size): value for node, value in bias.items()}
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
                        help="Size of the pegasus graph. Minimum 2. Default is 4 (P4).")
    parser.add_argument("-N", "--number", type=int, default=1,
                        help="Number of instances to be generated. Default is 1.")
    parser.add_argument("-C", "--category", type=str, default="RAU", choices=["RAU", "RAC", "AC3"],
                        help="Category of generated instances. RAU - random uniform, RAC - random couplings only, "
                             "AC3 - anti-cluster")
    parser.add_argument("-P", "--path", type=str, default=path,
                        help="path to folder where generated instances will be located. "
                             "Default is working directory")
    parser.add_argument("-T", "--types", type=str, default="SpinGlass",
                        choices=["SpinGlass", "DWave", "MatrixMarket"], nargs="*")
    parser.add_argument("--diag", type=bool, default=True,
                        help="Generate pegasus instances with or without \"diagonal\" connections")
    parser.add_argument("-D", "--device", type=Union[str, None], default=None,
                        choices=["Advantage_system4.1", "Advantage_system5.2", "Advantage_system6.1", None],
                        help="Map instance info physical D-Wave's device. Input None for no Mapping")

    args = parser.parse_args()

    if args.size and args.size < 2:
        parser.error("Minimum size of pegasus instance is 2")

    generate_pegasus_instances(args.number, args.size, args.path, args.types,
                               args.category, diagonal=args.diag, device=args.device)
