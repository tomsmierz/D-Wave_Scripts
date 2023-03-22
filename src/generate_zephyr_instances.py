import argparse
import os
from pathlib import Path
from typing import List, Optional

import dwave_networkx as dnx
import numpy as np
from dwave.system import DWaveSampler
from tqdm import tqdm

from src.graph_operations import align_graph_to_mapping, find_best_mapping
from src.graph_operations import create_zephyr_spinglass_clusters, reverse_sublattice_mapping, \
    zephyr_to_spin_glass
from src.interfaces.filesystem_interface import write_dwave_file, write_spin_glass_file

rng = np.random.default_rng()
cwd = os.getcwd()


def generate_zephyr_instances(
        number: int,
        size: int,
        output_path: str,
        output_types: List[str],
        category: str,
        device: Optional[str] = None,
        name: Optional[str] = None,
) -> None:
    graph = dnx.zephyr_graph(size, coordinates=True)

    if device not in [None, "Advantage2_prototype1.1"]:
        raise ValueError(
            'Device should be set to "Advantage2_prototype1.1" or None'
        )
    if size > 4:
        raise AssertionError("Maximum size for Advantage2 prototype is 4")

    if device is not None:
        sampler = DWaveSampler(solver=device)
        target = sampler.to_networkx_graph()
        mappings = [mapp for mapp in dnx.zephyr_sublattice_mappings(graph, target)]

        mapping, perfect_mapping, missing_nodes, missing_edges = find_best_mapping(mappings, sampler, graph)

        if not perfect_mapping:
            reverse_mapping = reverse_sublattice_mapping(mapping, graph)
            align_graph_to_mapping(mapping, reverse_mapping, graph, missing_nodes, missing_edges)

    for i in tqdm(
            range(number),
            desc=f"generating zephyr instances size = {size}, category={category}: ",
    ):
        if category == "AC3":
            bias = {node: rng.uniform(-1 / 9, 1 / 9) for node in graph.nodes()}
            clusters = create_zephyr_spinglass_clusters(graph)
            couplings = {
                edge: rng.uniform(-1 / 3, 1 / 3)
                if clusters[edge[0]] == clusters[edge[1]]
                else rng.uniform(-1, 1)
                for edge in graph.edges
            }
        elif category == "RAU":
            bias = {node: rng.uniform(-0.1, 0.1) for node in graph.nodes()}
            couplings = {edge: rng.uniform(-1, 1) for edge in graph.edges()}
        elif category == "RCO":
            bias = {node: 0 for node in graph.nodes()}
            couplings = {edge: rng.uniform(-1, 1) for edge in graph.edges()}
        else:
            raise ValueError(
                f'Category {category} is not a valid choice. It should be "RAU", "RCO" or "AC3"'
            )

        name = f"{name}{i + 1}" if name is None else f"{i + 1}"
        for output_type in output_types:
            if output_type == "SpinGlass":
                couplings_sg = {
                    (
                        zephyr_to_spin_glass(edge[0], size) + 1,
                        zephyr_to_spin_glass(edge[1], size) + 1,
                    ): value
                    for edge, value in couplings.items()
                }
                couplings_sg = dict(sorted(couplings_sg.items()))

                bias_sg = {
                    zephyr_to_spin_glass(node, size) + 1: value
                    for node, value in bias.items()
                }
                bias_sg = dict(sorted(bias_sg.items()))

                target_path = Path(output_path) / f"{name}_sg.txt"
                write_spin_glass_file(bias_sg, couplings_sg, target_path)

            elif output_type == "DWave":
                target_path = Path(output_path) / f"{name}_dv.pkl"
                # TODO: check the logic path to mapping being undefined.
                write_dwave_file(device, mapping, bias, couplings, target_path)

            elif output_type == "MatrixMarket":
                raise NotImplementedError("MatrixMarket output not implemented yet")

            else:
                raise ValueError(
                    f'{output_type} is not valid output type. It should be "SpinGlass", "DWave", '
                    f'or "MatrixMarket"'
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-S",
        "--size",
        type=int,
        default=2,
        help="Size of the zephyr graph. Minimum 1. Default is 2 (Z2).",
    )
    parser.add_argument(
        "-N",
        "--number",
        type=int,
        default=1,
        help="Number of instances to be generated. Default is 1.",
    )
    parser.add_argument(
        "-C",
        "--category",
        type=str,
        default="RAU",
        choices=["RAU", "RCO", "AC3"],
        help="Category of generated instances. RAU - random uniform, RCO - random couplings only, "
             "AC3 - anti-cluster",
    )
    parser.add_argument(
        "-P",
        "--path",
        type=str,
        default=cwd,
        help="path to folder where generated instances will be located. "
             "Default is working directory",
    )
    parser.add_argument(
        "-T",
        "--types",
        type=str,
        default=["SpinGlass"],
        choices=["SpinGlass", "DWave", "MatrixMarket"],
        nargs="*",
    )
    parser.add_argument(
        "-D",
        "--device",
        default=None,
        choices=["Advantage2_prototype1.1", None],
        help="Map instance info physical D-Wave's device. Input None for no Mapping",
    )

    args = parser.parse_args()

    if args.size and args.size < 1:
        parser.error("Minimum size of zephyr instance is 1")

    generate_zephyr_instances(
        args.number, args.size, args.path, args.types, args.category, device=args.device
    )
