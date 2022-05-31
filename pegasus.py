from dwave.system import AutoEmbeddingComposite, DWaveSampler
import dwave_networkx as dnx
import greedy
import networkx as nx
from numpy.random import default_rng
import pandas as pd
from dwave.cloud import Client
from tqdm import tqdm
import dwave.inspector
import matplotlib.pyplot as plt
import os
from pegasus_from_scratch import embed_pegasus

rng = default_rng()
path = os.getcwd()


def get_pegasus(size, number: str = "001"):
    df = pd.read_csv(f"C:/users/walle/PycharmProjects/D-Wave_Scripts/instances/cross_only/P{size}/{number}.txt",
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

#solver = greedy.SteepestDescentSolver()

#solver = AutoEmbeddingComposite(sampler)



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
l = {node: mappings[29](node) for node in source.nodes()}
nx.set_node_attributes(source, l, "mapping")

em = nx.get_node_attributes(source, "mapping")

h = {node: rng.uniform(-4, 4) for node in em.values()}
print(all(node in sampler.nodelist for node in h.keys()))
J = {(em[edge[0]], em[edge[1]]): rng.uniform(-1, 1) for edge in source.edges()}
print(all(edge in sampler.edgelist for edge in J.keys()))

"""


"""
for time in [long_time]:
    with open(os.path.join(path, f"energies_P4_crosses_{time}.txt"), "w") as f:
        for i in tqdm(range(100)):
            name = f"00{i+1}"[-3:]
            h, J = get_pegasus(4, name)

            sampleset = sampler.sample_ising(h, J, num_reads=num_reads, label=f'P4_{time}', annealing_time=time)

            best = sampleset.first

            f.write(name + ".txt" + " " + ":" + " " + f"{best[1]:.6f}" + " ")
            for value in best[0].values():
                f.write(str(int((value+1)/2)) + " ")
            f.write("\n")


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