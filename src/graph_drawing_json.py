import networkx as nx
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import json

rng = random.default_rng()

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':
    graph = nx.Graph()

    df = pd.read_csv(os.path.join(root, "instances", "example", "diagonal_5x5.txt"), header=None, sep=" ",
                      names=["n1", "n2", "v"])
    file = os.path.join(root, "energies", "example", "diagonal5x5droplets", "6426090948207743555.json")
    with open(file, encoding='utf-8') as f:
        sol_df = json.load(f)

    # solution is saved as Julia dict which is read as string. We perform string manipulation to get python dict.
        i = sol_df['columns'][sol_df['colindex']['lookup']['ig_states']-1][0]
        solution = []
        for a in i:
            sol = {int(key)-1: value for key, value in a.items()}
            sol = dict(sorted(sol.items()))
            solution.append(sol)

        h = {}
        J = {}
        for row in df.itertuples():
            if row.n1 == row.n2:
                h[row.n1-1] = row.v
            else:
                J[(row.n1-1, row.n2-1)] = row.v

        graph.add_edges_from(J.keys())
        x_values = np.array([0, 5, 10, 15, 20])
        y_values = np.array([0, 5, 10, 15, 20])

        base_vectors = np.array([[0, 0], [1, 0], [0.5, 0.5], [1.5, 0.5]])

        pos = np.array([[x + vx, y + vy] for x in x_values for y in y_values for vx, vy in base_vectors])
        plt.figure(figsize=(7.5, 7))
        ax = plt.gca()
        ax.set_axis_off()
        nodes = nx.draw_networkx_nodes(graph, pos, node_color='black', node_size=50)
        edges = nx.draw_networkx_edges(graph, pos, edge_color=J.values(), edge_cmap=plt.cm.seismic, edge_vmin=-1, edge_vmax=1)
        # plt.colorbar(edges)
        cbar = plt.colorbar(edges, cax=plt.gcf().add_axes([0.87, 0.2, 0.03, 0.6]))
        cbar.ax.tick_params(labelsize=15)
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.show()
        
        s = solution[0]
        soln = {int(key): value for key, value in s.items()}
        soln = dict(sorted(soln.items()))
        colors = {node: "red" if soln[node] == -1 else "blue" for node in sorted(graph.nodes)}

        plt.figure(figsize=(7, 7))
        ax = plt.gca()
        ax.set_axis_off()
        nx.draw_networkx_nodes(graph, pos=pos, node_color=list(colors.values()), node_size=100)
        nx.draw_networkx_edges(graph, pos=pos, edge_color='grey')
        # nx.draw(graph, pos=pos, node_color=list(colors.values()), edge_color="grey")
        # legend_labels = {'Spin +1': 'red', 'Spin -1': 'blue'}
        # legend_elements = [plt.Line2D([0], [0], marker='o', color=color, label=label, linestyle='None') for label, color in legend_labels.items()]

        # plt.legend(handles=legend_elements, loc='upper right')
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.show()
        
        for i in range(len(i)-1):
            s0 = solution[0]
            sol0 = {int(key): value for key, value in s0.items()}
            sol0 = dict(sorted(sol0.items()))
            
            s1 = solution[i+1]
            sol1 = {int(key): value for key, value in s1.items()}
            sol1 = dict(sorted(sol1.items()))
            
            highlight_nodes = [key for key in sol0 if key in sol1 and sol0[key] != sol1[key]]    
            edge_colors = ['green' if ((u, v) in J and (u in highlight_nodes and v in highlight_nodes)) or ((v, u) in J and (u in highlight_nodes and v in highlight_nodes)) else 'grey' for u, v in graph.edges]
            edge_widths = [3 if ((u, v) in J and (u in highlight_nodes and v in highlight_nodes)) or ((v, u) in J and (u in highlight_nodes and v in highlight_nodes))  else 0.5 for u, v in graph.edges]
            plt.figure(figsize=(7, 7))
            ax = plt.gca()
            ax.set_axis_off()
            other_nodes = set(graph.nodes) - set(highlight_nodes)
            nx.draw_networkx_nodes(graph, pos, nodelist=other_nodes, node_color='black', node_size=100)
            nx.draw_networkx_nodes(graph, pos, nodelist=highlight_nodes, node_color='green', node_size=100)

            nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=edge_widths)
            # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            plt.show()
