import matplotlib.pyplot as plt
import dwave_networkx as dnx

m = 2
graph = dnx.zephyr_graph(m, coordinates=True)



clusters = {}
for u, w, k, j, z in graph.nodes():
    if u == 0 and w % 2 == 0:
        clusters[(u, w, k, j, z)] = (2*(z+1), w + 1)
    elif u == 1 and w % 2 == 0:
        clusters[(u, w, k, j, z)] = (w + 1, 2*(z+1))
    elif u == 0 and w % 2 == 1:
        clusters[(u, w, k, j, z)] = (2*z + 2*j + 1, w + 1)
    else:
        clusters[(u, w, k, j, z)] = (w + 1, 2*z + 2*j + 1)

node_col = ["red" if clusters[node] == (2,1) else "blue" if clusters[node] == (3,2)
            else "green" if clusters[node] == (1,2) else "yellow" if clusters[node] == (2,3) else
            "purple" if clusters[node] == (4,1) else "darkorange" if clusters[node] == (5,2) else
            "black" for node in graph.nodes()]

edge_col = ["crimson" if clusters[edge[0]] == clusters[edge[1]] else "aqua" for edge in graph.edges()]
dnx.draw_zephyr(graph, node_color=node_col, edge_color=edge_col)


fig = plt.gcf()
fig.set_size_inches(32, 32)
plt.show()
fig.savefig(f'AC3_test_Z{m}.png', dpi=600)
