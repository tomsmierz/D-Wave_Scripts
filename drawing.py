from dwave.system import AutoEmbeddingComposite, DWaveSampler
import dwave_networkx as dnx
import greedy
import networkx as nx
from numpy.random import default_rng
import pandas as pd
from dwave.cloud import Client
from tqdm import tqdm
#import dwave.inspector
import matplotlib.pyplot as plt
import os

path = os.getcwd()

sampler = DWaveSampler(solver="Advantage_system6.1")

graph = dnx.pegasus_graph(16, fabric_only=True)


"""
for i in range(6):
    graph.remove_node(26 + i)
    graph.remove_node(16 + i)

for i in range(2):
    graph.remove_node(44 + i)
    graph.remove_node(2 + i)
    """

"""
for i in range(18):
    graph.remove_node(150 + i)
    graph.remove_node(120 + i)

for i in range(6):
    graph.remove_node(6 + i)
    graph.remove_node(276 + i)
"""


"""
for i in range(42):
    graph.remove_node(686 + i)
    graph.remove_node(616 + i)

for i in range(14):
    graph.remove_node(14 + i)
    graph.remove_node(1316 + i)

"""


for i in range(90):
    graph.remove_node(2910 + i)
    graph.remove_node(2760 + i)
    
    
for i in range(30):
    graph.remove_node(30 + i)
    graph.remove_node(5700 + i)



real_nodes = sampler.nodelist
real_edges = sampler.edgelist

print(len(real_edges))
print(len(graph.edges))

broken_nodes = list(set(graph.nodes) - set(real_nodes))
broken_edges = list(set(graph.edges) - set(real_edges))

for i in broken_nodes:
    graph.remove_node(i)

print(len(broken_edges))


atributes = {edge: 'black' if edge not in broken_edges else 'r' for edge in graph.edges}
n_atributes = {node: 'b' if node not in broken_nodes else 'r' for node in graph.nodes}

nx.set_edge_attributes(graph, atributes, "color")
nx.set_node_attributes(graph, n_atributes, "color")

edges = graph.edges()

colors_e = [graph[u][v]['color'] for u,v in edges]
colors_n = [u for u in nx.get_node_attributes(graph, "color").values()]




dnx.draw_pegasus(graph, with_labels=True, crosses=True)
                 #edge_color=colors_e, node_color = colors_n)

fig = plt.gcf()
fig.set_size_inches(64, 64)
plt.show()
#fig.savefig('test2png.png', dpi=1800)
