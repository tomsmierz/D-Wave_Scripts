from math import inf
from typing import Callable

import dwave_networkx as dnx
import networkx as nx
from dwave.system import DWaveSampler
from tqdm import tqdm


def reverse_sublattice_mapping(mapping: Callable, source: nx.Graph) -> Callable:
    node_dict = {node: mapping(node) for node in source.nodes}
    reversed_dict = {value: key for key, value in node_dict.items()}

    def func(node: int) -> tuple:
        return reversed_dict[node]

    return func


def chimera_to_spin_glass(q: tuple, size: int) -> int:
    return dnx.chimera_coordinates(size).chimera_to_linear(q)



def zephyr_to_spin_glass(q: tuple, size: int) -> int:
    return dnx.zephyr_coordinates(size).zephyr_to_linear(q)


def create_zephyr_spinglass_clusters(graph: nx.Graph) -> dict:
    clusters = {}
    for u, w, k, j, z in graph.nodes():
        if u == 0 and w % 2 == 0:
            clusters[(u, w, k, j, z)] = (2 * (z + 1), w + 1)
        elif u == 1 and w % 2 == 0:
            clusters[(u, w, k, j, z)] = (w + 1, 2 * (z + 1))
        elif u == 0 and w % 2 == 1:
            clusters[(u, w, k, j, z)] = (2 * z + 2 * j + 1, w + 1)
        else:
            clusters[(u, w, k, j, z)] = (w + 1, 2 * z + 2 * j + 1)
    return clusters


def align_graph_to_mapping(reverse_mapping, source: nx.Graph, missing_nodes, missing_edges):
    for node in missing_nodes:
        source.remove_node(reverse_mapping(node))

    for edge in missing_edges:
        edge = tuple(edge)
        edge = (reverse_mapping(edge[0]), reverse_mapping(edge[1]))
        reversed_edge = (edge[1], edge[0])
        if edge in source.edges():
            source.remove_edge(edge[0], edge[1])

        if reversed_edge in source.edges():
            source.remove_edge(reversed_edge[0], reversed_edge[1])
    return source


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
