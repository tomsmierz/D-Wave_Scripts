import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt



#dnx.generators.pegasus_coordinates()

def embed_pegasus(size, pos):
    graph = dnx.pegasus_graph(2, nice_coordinates=True)
    em = {node: (node[0], node[1] + pos[0], node[2] + pos[1], node[3], node[4]) for node in graph.nodes()}
    nx.set_node_attributes(graph, em, "embedding")

    return(graph)

g = embed_pegasus(2, (1, 1))

print(nx.get_node_attributes(g, "embedding"))
