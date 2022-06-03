import os
from dwave.system import DWaveSampler
import dwave_networkx as dnx

path = os.getcwd()

sampler = DWaveSampler(solver="Advantage_system6.1")
real_graph = sampler.to_networkx_graph()

graph = dnx.pegasus_graph(16, data=False, fabric_only=True)

real_nodes = sampler.nodelist
real_edges = sampler.edgelist


for i in range(90):
    graph.remove_node(2910 + i)
    graph.remove_node(2760 + i)

for i in range(30):
    graph.remove_node(30 + i)
    graph.remove_node(5700 + i)


print(len(real_edges))
print(len(graph.edges))

broken_nodes = list(set(graph.nodes) - set(real_nodes))
broken_edges = list(set(graph.edges) - set(real_edges))

print(len(broken_edges))
print(len(broken_nodes))


c=0
with open(os.path.join(path, "../instances/cross_only/P16/002.txt"), "a") as f:
    for node in broken_nodes:
        f.write(str(node + 1) + " " + str(node + 1) + " " + str(0) + "\n")
        c+=1
    for edge in broken_edges:
        f.write(str(edge[0] + 1) + " " + str(edge[1] + 1) + " " + str(0) + "\n")
        c += 1
print(c)

