import os
from math import inf
from typing import Callable

import networkx as nx
import pandas as pd
from dwave.system import DWaveSampler
from tqdm import tqdm


def load_pegasus(path: str, name: str = "001"):
    df = pd.read_csv(
        os.path.join(path, f"{name}.txt"),
        sep=" ",
        index_col=False,
        skiprows=1,
        header=None,
    )
    h = {}
    J = {}
    for index, row in df.iterrows():
        if row[0] == row[1]:
            h[int(row[0] - 1)] = row[2]
            # h[int(row[0])] = row[2]
        else:
            J[(int(row[0] - 1), int(row[1] - 1))] = row[2]
            # J[(int(row[0]), int(row[1]))] = row[2]
    return h, J


def load_pegasus_tuple(path: str, name: str = "001"):
    df = pd.read_csv(
        os.path.join(path, f"{name}.txt"),
        sep=";",
        index_col=False,
        skiprows=1,
        header=None,
    )
    h = {}
    J = {}
    for index, row in df.iterrows():
        if row[0] == row[1]:
            # h[int(row[0] - 1)] = row[2]\
            h[eval(row[0])] = row[2]
        else:
            # J[(int(row[0] - 1), int(row[1] - 1))] = row[2]
            J[(eval(row[0]), eval(row[1]))] = row[2]
    return h, J


def prepare_data_to_save(device, mapping, couplings, bias) -> list[dict, dict]:
    if device is None:
        return [bias, couplings]
    couplings_dv = {
        (mapping(edge[0]), mapping(edge[1])): value
        for edge, value in couplings.items()
    }
    bias_dv = {mapping(node): value for node, value in bias.items()}
    return [bias_dv, couplings_dv]


def find_best_mapping(mappings: list[Callable], sampler: DWaveSampler, source: nx.Graph):
    target = sampler.to_networkx_graph()
    real_nodes_set = set(sampler.nodelist)

    current_best_mapping = None
    best_missing_nodes = None
    best_missing_edges = None

    min_num_of_missing_edges = inf
    min_num_of_missing_nodes = inf

    for mapping in tqdm(mappings, desc="Searching for a perfect mapping"):
        node_dict = {node: mapping(node) for node in source.nodes()}
        edge_dict = {(v, w): (node_dict[v], node_dict[w]) for v, w in source.edges()}

        if all(node in target.nodes() for node in node_dict.values()) and \
                all(edge in target.edges() for edge in edge_dict.values()):
            print("\nPerfect map found")
            return mapping, True, 0, 0

        else:
            mapped_source_nodes_set = set(node_dict.values())
            mapped_source_edges_set = {frozenset(edge) for edge in edge_dict.values()}

            real_edges_set = {frozenset(edge) for edge in sampler.edgelist}

            missing_nodes = mapped_source_nodes_set - real_nodes_set
            missing_edges = mapped_source_edges_set - real_edges_set

            if (
                    len(missing_nodes) <= min_num_of_missing_nodes
                    and len(missing_edges) <= min_num_of_missing_edges
            ):
                min_num_of_missing_nodes = len(missing_nodes)
                min_num_of_missing_edges = len(missing_edges)
                best_missing_nodes = missing_nodes
                best_missing_edges = missing_edges
                current_best_mapping = mapping

    if current_best_mapping is None:
        raise RuntimeError(
            "No map found. Possible problem with the source or the target graph"
        )

    print(
        f"\nNo perfect map found. Returning imperfect map with {min_num_of_missing_nodes} missing nodes"
        f" and {min_num_of_missing_edges} missing edges"
    )
    return current_best_mapping, False, best_missing_nodes, best_missing_edges
