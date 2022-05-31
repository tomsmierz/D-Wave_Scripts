import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt


def embed_pegasus(size: int, pos):
    graph = dnx.pegasus_graph(size, nice_coordinates=True)
    converter = dnx.generators.pegasus_coordinates(size)
    em = {node: (node[0], node[1] + pos[0], node[2] + pos[1], node[3], node[4]) for node in graph.nodes()}
    nx.set_node_attributes(graph, em, "embedding")
    em_l = {node: converter.nice_to_linear(em[node]) for node in graph.nodes()}
    nx.set_node_attributes(graph, em_l, "embedding_linear")

    return(graph)


