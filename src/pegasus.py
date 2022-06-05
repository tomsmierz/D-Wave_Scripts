import greedy
import os
import dwave.inspector

import dwave_networkx as dnx
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from dwave.system import LazyEmbeddingComposite,AutoEmbeddingComposite, DWaveSampler
from numpy.random import default_rng
from dwave.cloud import Client
from tqdm import tqdm
from typing import List

rng = default_rng()
cwd = os.getcwd()
client = Client.from_config()

min_time = 0.5
default_time = 20
long_time = 150
max_time = 2000

sampler = DWaveSampler(solver="Advantage_system6.1")
solver = AutoEmbeddingComposite(sampler)


def get_pegasus(path: str, name: str = "001"):
    df = pd.read_csv(os.path.join(path, f"{name}.txt"),
                     sep=" ", index_col=False, skiprows=1, header=None)
    h = {}
    J = {}
    for index, row in df.iterrows():
        if row[0] == row[1]:
            h[int(row[0] - 1)] = row[2]
        else:
            J[(int(row[0] - 1), int(row[1] - 1))] = row[2]
    return h, J


def anneal(input_path: str, output_name: str, output_path: str = cwd,
           num_reads: int = 5000, annealing_time: float = 20.0, random: bool = True):

    with open(os.path.join(output_path, f"{output_name}_{annealing_time}.txt"), "w") as f:
        for i in tqdm(range(100)):

            if random:
                name = f"00{i + 1}"[-3:]
            else:
                name = f"0{i}"[-2:]

            h, J = get_pegasus(input_path, name)
            del(J[(2032, 4270)])

            sampleset = sampler.sample_ising(h, J, num_reads=num_reads, auto_scale=False,
                                             label=f'{output_name}_{annealing_time}', annealing_time=annealing_time)

            best = sampleset.first

            f.write(name + ".txt" + " " + ":" + " " + f"{best[1]:.6f}" + " ")
            for value in best[0].values():
                f.write(str(int((value + 1) / 2)) + " ")
            f.write("\n")
"""

graph = dnx.pegasus_graph(16, nice_coordinates=True)

real_nodes = sampler.nodelist
real_edges = sampler.edgelist

broken_nodes = list(set(graph.nodes) - set(real_nodes))
broken_edges = list(set(graph.edges) - set(real_edges))
"""
#solver = greedy.SteepestDescentSolver()

#solver = AutoEmbeddingComposite(sampler)
"""
h, J = get_pegasus(2, "001")

sampleset = sampler.sample_ising(h, J, num_reads=1, label="test")
dwave.inspector.show(sampleset)
"""
"""
source = dnx.pegasus_graph(8, nice_coordinates=True)
#target = dnx.pegasus_graph(16, nice_coordinates=True)
target = sampler.to_networkx_graph()
mappings = [mapping for mapping in dnx.pegasus_sublattice_mappings(source, target)]

for i in tqdm(range(len(mappings))):

    l = {node: mappings[i](node) for node in source.nodes()}
    nx.set_node_attributes(source, l, "mapping")

    em = nx.get_node_attributes(source, "mapping")

    h = {node: rng.uniform(-4, 4) for node in em.values()}
    #print(all(node in sampler.nodelist for node in h.keys()))
    J = {(em[edge[0]], em[edge[1]]): rng.uniform(-1, 1) for edge in source.edges()}
    #print(all(edge in sampler.edgelist for edge in J.keys()))
    if all(node in sampler.nodelist for node in h.keys()): #and all(edge in sampler.edgelist for edge in J.keys()):
        print(i, list(set(J.keys()) - set(real_edges)))

"""

"""
l = {node: mappings[29](node) for node in source.nodes()}
nx.set_node_attributes(source, l, "mapping")

em = nx.get_node_attributes(source, "mapping")

h = {node: rng.uniform(-4, 4) for node in em.values()}
print(all(node in sampler.nodelist for node in h.keys()))
J = {(em[edge[0]], em[edge[1]]): rng.uniform(-1, 1) for edge in source.edges()}
print(all(edge in sampler.edgelist for edge in J.keys()))

"""





"""
asadssf
with open(os.path.join(path, f"energies_P16_greedy.txt"), "w") as f:
    for i in tqdm(range(100)):
        name = f"00{i + 1}"[-3:]
        h, J = get_pegasus("P16", name)

        sampleset = solver.sample_ising(h, J)

        best = sampleset.first

        f.write(name + ".txt" + " " + ":" + " " + f"{best[1]:.6f}" + " ")
        for value in best[0].values():
            f.write(str(int((value + 1) / 2)) + " ")
        f.write("\n")
"""

if __name__ == "__main__":

    annealing_times = [min_time, default_time, long_time]  # [min_time, default_time, long_time]
    path = "/home/tsmierzchalski/pycharm_projects/D-Wave_Scripts/instances/normal/P8"
    name = "P8_normal"
    for time in annealing_times:
        anneal(path, f"energies_{name}", annealing_time=time)

    """
    h, J = get_pegasus("/home/tsmierzchalski/pycharm_projects/D-Wave_Scripts/instances/normal/P4", "001")

    sampleset = sampler.sample_ising(h, J, num_reads=1, label="test")
    dwave.inspector.show(sampleset)
    """



    """
    annealing_times = [min_time, default_time] #[min_time, default_time, long_time]
    for pg in ["pegasus_2x2x3"]: #, "pegasus_3x3x3", "Pegasus_4x4x3"]:
        for x in os.scandir(f"C:/Users/walle/PycharmProjects/D-Wave_Scripts/instances/pegasus/{pg}"):
            for time in annealing_times:
                anneal(x.path, f"energies_{x.name}", annealing_time=time, random=False)

    """