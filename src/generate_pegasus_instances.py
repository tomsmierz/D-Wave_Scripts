import itertools
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import dwave_networkx as dnx  # type: ignore
import networkx as nx
import numpy as np
from dwave.system import DWaveSampler
from joblib import Parallel, delayed
from tqdm import tqdm

from src.graph_operations import align_graph_to_mapping, find_best_mapping
from src.interfaces.filesystem_interface import write_dwave_file, write_spin_glass_file
from src.utils import nice_to_spin_glass

path = os.getcwd()


def reverse_pegasus_sublattice_mapping(mapping: Callable, source: nx.Graph) -> Callable:
    node_dict = {node: mapping(node) for node in source.nodes}
    reversed_dict = {value: key for key, value in node_dict.items()}

    def func(node: int) -> Tuple:
        return reversed_dict[node]

    return func


def job(i, category, graph, output_types, output_path, mapping, name, username):
    if category == "AC3":
        bias = {node: np.random.uniform(-1 / 9, 1 / 9) for node in graph.nodes()}
        couplings = {
            edge: np.random.uniform(-1 / 3, 1 / 3)
            if edge[0][1:3] == edge[1][1:3]
            else np.random.uniform(-1, 1)
            for edge in graph.edges()
        }
    elif category == "CBFM-P":
        bias = {node: np.random.choice([-1, 0], p=[0.85, 0.15]) for node in graph.nodes()}
        couplings = {
            edge: np.random.choice([-1, 0, 1], p=[0.1, 0.35, 0.55])
            for edge in graph.edges()
        }
    elif category == "RAU":
        bias = {node: np.random.uniform(-0.1, 0.1) for node in graph.nodes()}
        couplings = {edge: np.random.uniform(-1, 1) for edge in graph.edges()}
    elif category == "RCO":
        bias = {node: 0 for node in graph.nodes()}
        couplings = {edge: np.random.uniform(-1, 1) for edge in graph.edges()}
    else:
        raise ValueError(
            f'Category {category} is not a valid choice. It should be "RAU", "RCO" or "AC3"'
        )

    name = f"{name}{i + 1}" if username else f"{i + 1}"
    for output_type in output_types:
        if (
                output_type == "SpinGlass"
        ):
            couplings_sg = {
                (
                    nice_to_spin_glass(edge[0], size),
                    nice_to_spin_glass(edge[1], size),
                ): value
                for edge, value in couplings.items()
            }
            couplings_sg = dict(sorted(couplings_sg.items()))

            bias_sg = {
                nice_to_spin_glass(node, size): value
                for node, value in bias.items()
            }

            target_path = Path(output_path) / f"{name}_sg.txt"
            write_spin_glass_file(bias_sg, couplings_sg, target_path)

        elif output_type == "DWave":
            target_path = Path(target_path) / f"{name}_dv.pkl"
            # TODO: check the logic path to mapping being undefined.
            write_dwave_file(device, mapping, bias, couplings, target_path)

        elif output_type == "MatrixMarket":
            raise NotImplementedError("MatrixMarket output not implemented yet")

        else:
            raise ValueError(
                f'{output_type} is not valid output type. It should be "SpinGlass", "DWave", '
                f'or "MatrixMarket"'
            )


def generate_pegasus_instances(
        number: int,
        size: int,
        output_path: str,
        output_types: List[str],
        category: str,
        diagonal: bool = True,
        device: Optional[str] = None,
        name: Optional[str] = None,
) -> None:
    graph = dnx.pegasus_graph(size, nice_coordinates=True)
    username = name is not None

    if device not in [None, "Advantage_system4.1", "Advantage_system5.2", "Advantage_system6.1"]:
        raise ValueError(
            'Device should be set to "Advantage_system4.1", "Advantage_system5.2", '
            '"Advantage_system6.1" or None'
        )
    if size > 16:
        raise ValueError("Maximum size for working device is 16")

    if device is not None:
        sampler = DWaveSampler(solver=device)
        target = sampler.to_networkx_graph()
        mappings = [mapp for mapp in dnx.pegasus_sublattice_mappings(graph, target)]
        mapping, perfect_mapping, missing_nodes, missing_edges = find_best_mapping(mappings, sampler, graph)

        if not perfect_mapping:
            reverse_mapping = reverse_pegasus_sublattice_mapping(mapping, graph)
            align_graph_to_mapping(reverse_mapping, graph, missing_nodes, missing_edges)

    if not diagonal:
        for y, x, i in itertools.product(range(size - 1), range(1, size), range(4)):
            h = (0, y, x, 0, i)
            h1 = (2, y + 1, x - 1, 1, 0)
            h2 = (2, y + 1, x - 1, 1, 1)
            v = (0, y, x, 1, i)
            v1 = (2, y + 1, x - 1, 0, 2)
            v2 = (2, y + 1, x - 1, 0, 3)
            for e in [h1, h2]:
                if graph.has_edge(h, e):
                    graph.remove_edge(h, e)
            for e in [v1, v2]:
                if graph.has_edge(v, e):
                    graph.remove_edge(v, e)

    Parallel(n_jobs=16)(delayed(job)(i, category, graph, output_types, output_path, mapping, name, username) for i in tqdm(range(number)))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-S",
    #     "--size",
    #     type=int,
    #     default=4,
    #     help="Size of the pegasus graph. Minimum 2. Default is 4 (P4).",
    # )
    # parser.add_argument(
    #     "-N",
    #     "--number",
    #     type=int,
    #     default=1,
    #     help="Number of instances to be generated. Default is 1.",
    # )
    # parser.add_argument(
    #     "-C",
    #     "--category",
    #     type=str,
    #     default="RAU",
    #     choices=["RAU", "RCO", "AC3", "CBFM-P"],
    #     help="Category of generated instances. RAU - random uniform, RCO - random couplings only, "
    #          "AC3 - anti-cluster",
    # )
    # parser.add_argument(
    #     "-P",
    #     "--path",
    #     type=str,
    #     default=path,
    #     help="path to folder where generated instances will be located. "
    #          "Default is working directory",
    # )
    # parser.add_argument(
    #     "-T",
    #     "--types",
    #     type=str,
    #     default=["SpinGlass"],
    #     choices=["SpinGlass", "DWave", "MatrixMarket"],
    #     nargs="*",
    # )
    # parser.add_argument(
    #     "--diag",
    #     type=bool,
    #     default=True,
    #     help='Generate pegasus instances with or without "diagonal" connections',
    # )
    # parser.add_argument(
    #     "-D",
    #     "--device",
    #     default=None,
    #     choices=[
    #         "Advantage_system4.1",
    #         "Advantage_system5.2",
    #         "Advantage_system6.1",
    #         None,
    #     ],
    #     help="Map instance info physical D-Wave's device. Input None for no Mapping",
    # )

    pegasus_graph_size = 10
    category = "RAU"

    test_set_size = 300
    val_set_size = 300
    train_set_size = 15000
    target_path_test = "~/projects/ncbr-qubo-ml/data/pegasus_lite/test"
    target_path_val = "~/projects/ncbr-qubo-ml/data/pegasus_lite/val"
    target_path_train = "~/projects/ncbr-qubo-ml/data/pegasus_lite/train"
    types = ["DWave"]
    diagonal = True
    device = "Advantage_system6.1"

    for size, path in [
        (test_set_size, target_path_test),
        (val_set_size, target_path_val),
        (train_set_size, target_path_train)]:
        generate_pegasus_instances(size, pegasus_graph_size, path, types, category, diagonal, device)
