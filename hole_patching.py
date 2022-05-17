from dwave.system import EmbeddingComposite, DWaveSampler
import dwave_networkx as dnx
import networkx as nx
from numpy.random import default_rng
import pandas as pd
from dwave.cloud import Client
from tqdm import tqdm

rng = default_rng()
NUM_READS = 10 #default value for chimera

client = Client.from_config()   
#print(client.get_solvers())

h_list = []
J_list = []
spins = []
energies = []

# Define the sampler that will be used to run the problem
graph = dnx.chimera_graph(16,16,4)

sampler = DWaveSampler(solver="DW_2000Q_6")
all_nodes = sampler.nodelist
all_edges = sampler.edgelist

real_chimera = nx.Graph()
real_chimera.add_edges_from(all_edges) #creates graph with identical structure as real life chimera

broken_nodes = len(list(set(graph.nodes) - set(all_nodes))) #difference between theory and real

real_chimera_sub = real_chimera.subgraph(all_nodes[0:graph.number_of_nodes()-broken_nodes])


for _ in tqdm(range(500)):
    h = {node: rng.normal(0,1) for node in all_nodes} #real_chimera_sub.nodes}
    #zeros = {x: 0 for x in list(set(all_nodes) - set(h.keys()))}
    #h.update(zeros)

    J = {edge: rng.normal(0,1) for edge in all_edges}  #real_chimera_sub.edges}
    #zeros = {x: 0 for x in list(set(all_edges) - set(J.keys()))}
    #J.update(zeros)

        # Run the problem on the sampler and print the results
    sampleset = sampler.sample_ising(h, J,
                                        num_reads = NUM_READS,
                                        label='Chimera 2048')

    best = sampleset.first
    h_list.append(h)
    J_list.append(J)
    spins.append(best[0])
    energies.append(best[1])

d = {'h': h_list, 'J': J_list, "spin_conf": spins, "energy": energies}

df = pd.DataFrame(data=d)
df.to_csv("data/2048.csv")


