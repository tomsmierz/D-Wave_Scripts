import os
import networkx as nx
import numpy as np
import pandas as pd
from typing import Union, Tuple
from functools import reduce

rng = np.random.default_rng()


def factors(n):
    return set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


def generate_lattice_instances(size: Union[int, Tuple], path: str):
    if isinstance(size, int):
        fc = sorted(list(factors(size)))
        m = int(len(fc)/2)
        dim = (fc[m-1], fc[m])
    else:
        dim = size
    graph = nx.convert_node_labels_to_integers(nx.grid_graph(dim))
    df = pd.DataFrame(columns=["i", "j", "v"])

    for node in graph.nodes:
        v = rng.uniform(-1, 1)
        df.loc[len(df)] = [node, node, v]
    for edge in graph.edges:
        v = rng.uniform(-1, 1)
        df.loc[len(df)] = [edge[0], edge[1], v]
    df = df.astype({'i': 'int64', 'j': 'int64'})

    df.to_csv(path, index=False)


generate_lattice_instances(60, os.path.join(os.getcwd(), "60.csv"))

