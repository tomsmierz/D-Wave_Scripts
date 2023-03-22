import argparse
import os
from pathlib import Path
from typing import List, Optional

import dwave_networkx as dnx
import numpy as np
from dwave.system import DWaveSampler
from tqdm import tqdm

from src.graph_operations import align_graph_to_mapping, chimera_to_spin_glass, find_best_mapping, \
    reverse_sublattice_mapping
from src.interfaces.filesystem_interface import write_dwave_file, write_spin_glass_file

rng = np.random.default_rng()
path = os.getcwd()


def generate_chimera_instances(
        number: int,
        size: int,
        output_path: str,
        output_types: List[str],
        category: str,
        device: Optional[str] = None,
        name: Optional[str] = None,
) -> None:
    graph = dnx.chimera_graph(size, coordinates=True)

    if device not in [None, "DW_2000Q_6"]:
        raise ValueError('Device should be set to "DW_2000Q_6" or None')

    if size > 16:
        raise ValueError("Maximum size for working chimera device is 16")

    if device is not None:
        sampler = DWaveSampler(solver=device)
        target = sampler.to_networkx_graph()
        mappings = [mapp for mapp in dnx.chimera_sublattice_mappings(graph, target)]

        mapping, found_perfect_mapping, missing_nodes, missing_edges = find_best_mapping(mappings, sampler, graph)

        if not found_perfect_mapping:
            reverse_mapping = reverse_sublattice_mapping(mapping, graph)
            align_graph_to_mapping(reverse_mapping, graph, missing_nodes, missing_edges)

    for i in tqdm(
            range(number),
            desc=f"generating Chimera instances size = {size}, category={category}: ",
    ):
        if category == "RAU":
            bias = {node: rng.uniform(-0.1, 0.1) for node in graph.nodes()}
            couplings = {edge: rng.uniform(-1, 1) for edge in graph.edges()}
        elif category == "RCO":
            bias = {node: 0 for node in graph.nodes()}
            couplings = {edge: rng.uniform(-1, 1) for edge in graph.edges()}
        elif category == "AC3":
            bias = {node: rng.uniform(-1 / 9, 1 / 9) for node in graph.nodes()}
            couplings = {
                edge: rng.uniform(-1 / 3, 1 / 3)
                if edge[0][1:3] == edge[1][1:3]
                else rng.uniform(-1, 1)
                for edge in graph.edges()
            }
        else:
            raise ValueError(f'Category {category} is not a valid choice. It should be "RAU", "RCO" or "AC3"')
        if name is not None:
            name = f"{i + 1}"
        else:
            name = f"{name}{i + 1}"

        for output_type in output_types:
            if output_type == "SpinGlass":
                couplings_sg = {
                    (
                        chimera_to_spin_glass(edge[0], size) + 1,
                        chimera_to_spin_glass(edge[1], size) + 1,
                    ): value
                    for edge, value in couplings.items()
                }
                couplings_sg = dict(sorted(couplings_sg.items()))

                bias_sg = {
                    chimera_to_spin_glass(node, size) + 1: value
                    for node, value in bias.items()
                }
                bias_sg = dict(sorted(bias_sg.items()))
                target_path = Path(output_path) / f"{name}_sg.txt"
                write_spin_glass_file(bias_sg, couplings_sg, target_path)

            elif output_type == "DWave":
                target_path = Path(output_path) / f"{name}_dv.pkl"
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
        default=4,
        help="Size of the Chimera graph. Default is 4 (C4).",
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
        default=path,
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
        choices=["DW_2000Q_6", None],
        help="Map instance info physical D-Wave's device. Input None for no Mapping",
    )

    args = parser.parse_args()

    generate_chimera_instances(
        args.number, args.size, args.path, args.types, args.category, device=args.device
    )
