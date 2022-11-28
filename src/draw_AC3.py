import matplotlib.pyplot as plt
import dwave_networkx as dnx


graph = dnx.pegasus_graph(5, nice_coordinates=True)

edge_col = ["red" if edge[0][1:3] == edge[1][1:3] else "blue" for edge in graph.edges()]

dnx.draw_pegasus(graph, crosses=True, node_color="black", edge_color=edge_col)


fig = plt.gcf()
fig.set_size_inches(32, 32)
plt.show()