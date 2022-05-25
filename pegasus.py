from dwave.system import AutoEmbeddingComposite, DWaveSampler
import dwave_networkx as dnx
import networkx as nx
from numpy.random import default_rng
import pandas as pd
from dwave.cloud import Client
from tqdm import tqdm
import dwave.inspector
import matplotlib.pyplot as plt
import os

path = os.getcwd()


def get_pegasus(size, number: str = "001"):
    df = pd.read_csv(os.path.join(path, f"instances/{size}/{number}.txt"),
                     sep=" ", index_col=False, skiprows=1, header=None)
    h = {}
    J = {}
    for index, row in df.iterrows():
        if row[0] == row[1]:
            h[int(row[0] - 1)] = row[2]
        else:
            J[(int(row[0] - 1), int(row[1] - 1))] = row[2]
    return h, J


client = Client.from_config()
num_reads = 5000

min_time = 0.5
default_time = 20
long_time = 150
max_time = 2000

sampler = DWaveSampler(solver="Advantage_system6.1")

graph = dnx.pegasus_graph(16, data=False, fabric_only=False)

real_nodes = sampler.nodelist
real_edges = sampler.edgelist

broken_nodes = list(set(graph.nodes) - set(real_nodes))
broken_edges = list(set(graph.edges) - set(real_edges))

#solver = AutoEmbeddingComposite(sampler)
# solver.sample_ising(num_reads=1, label="Pegasus", )

for time in [long_time]:
    with open(os.path.join(path, f"energies_P16_{time}.txt"), "w") as f:
        for i in tqdm(range(100)):
            name = f"00{i+1}"[-3:]
            h, J = get_pegasus("P16", name)
            for v in broken_nodes:
                del h[v]
            for e in broken_edges:
                del J[e]

            sampleset = sampler.sample_ising(h, J, num_reads=num_reads, label='test',
                                             auto_scale=True, annealing_time=time)

            best = sampleset.first

            f.write(name + ".txt" + " " + ":" + " " + f"{best[1]:.6f}" + " ")
            for value in best[0].values():
                f.write(str(int((value+1)/2)) + " ")
            f.write("\n")

