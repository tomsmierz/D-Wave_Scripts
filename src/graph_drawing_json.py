import networkx as nx
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import json
from matplotlib.colors import ListedColormap


rng = random.default_rng()

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(root, "instances", "example", "diagonal_5x5.txt"), header=None, sep=" ",
                names=["n1", "n2", "v"])
file = os.path.join(root, "energies", "example", "diagonal5x5droplets", "6426090948207743555.json")
scale = 7

def load_dict(f):
    sol_df = json.load(f)
    i = sol_df['columns'][sol_df['colindex']['lookup']['ig_states']-1][0]
    cl = sol_df['columns'][sol_df['colindex']['lookup']['clusters']-1]
    result_dict = {}
    for item in cl:
        for key_str, value_dict in list(item.items()):
            key = eval(key_str)
            labels = [label - 1 for label in value_dict['labels']]
            result_dict[key] = labels
    solution = []
    for a in i:
        sol = {int(key)-1: value for key, value in a.items()}
        sol = dict(sorted(sol.items()))
        solution.append(sol)
    return result_dict, solution

def draw_instance(df, file, scale):
    graph = nx.Graph()

    with open(file, encoding='utf-8') as f:
        result_dict, _ = load_dict(f)

        h = {}
        J = {}
        for row in df.itertuples():
            if row.n1 == row.n2:
                h[row.n1-1] = row.v
            else:
                J[(row.n1-1, row.n2-1)] = row.v
        graph.add_edges_from(J.keys())

        plt.figure(figsize=(7.5, 7))
        ax = plt.gca()
        ax.set_axis_off()

        pos = {}
        for cluster, spins in result_dict.items():
            cluster_center = cluster
            cluster_pos = nx.circular_layout(range(len(spins)))
            pos.update({spin: (cluster_center[0] + cluster_pos[i][0]/scale, cluster_center[1] + cluster_pos[i][1]/scale) for i, spin in enumerate(spins)})
            nx.draw_networkx_nodes(graph, pos, nodelist=spins, node_color='black', node_size=50)
        edges = nx.draw_networkx_edges(graph, pos, edge_color=J.values(), edge_cmap=plt.cm.bwr_r, edge_vmin=-1, edge_vmax=1)
        cbar = plt.colorbar(edges, cax=plt.gcf().add_axes([0.87, 0.2, 0.03, 0.6]))
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.set_title(r'$J_{ij}$', fontsize=20)
        plt.show()
        
            
def draw_solution(df, file, scale):
    graph = nx.Graph()

    with open(file, encoding='utf-8') as f:
        result_dict, solution = load_dict(f)

        h = {}
        J = {}
        for row in df.itertuples():
            if row.n1 == row.n2:
                h[row.n1-1] = row.v
            else:
                J[(row.n1-1, row.n2-1)] = row.v
        graph.add_edges_from(J.keys()) 
        
        s = solution[0]
        soln = {int(key): value for key, value in s.items()}
        soln = dict(sorted(soln.items()))
        colors = {node: "blue" if soln[node] == -1 else "red" for node in sorted(graph.nodes)}
        
        plt.figure(figsize=(7, 7))
        ax = plt.gca()
        ax.set_axis_off()
        
        pos = {}
        for cluster, spins in result_dict.items():
            cluster_center = cluster
            cluster_pos = nx.circular_layout(range(len(spins)))
            pos.update({spin: (cluster_center[0] + cluster_pos[i][0]/scale, cluster_center[1] + cluster_pos[i][1]/scale) for i, spin in enumerate(spins)})
        
        nx.draw_networkx_nodes(graph, pos=pos, node_color=list(colors.values()), node_size=100)
        nx.draw_networkx_edges(graph, pos=pos, edge_color='grey')
        plt.show()
          
    
def draw_droplet(df, file, scale):
    graph = nx.Graph()

    with open(file, encoding='utf-8') as f:
        result_dict, solution = load_dict(f)

        h = {}
        J = {}
        for row in df.itertuples():
            if row.n1 == row.n2:
                h[row.n1-1] = row.v
            else:
                J[(row.n1-1, row.n2-1)] = row.v
        graph.add_edges_from(J.keys())
        
        pos = {}
        for cluster, spins in result_dict.items():
            cluster_center = cluster
            cluster_pos = nx.circular_layout(range(len(spins)))
            pos.update({spin: (cluster_center[0] + cluster_pos[i][0]/scale, cluster_center[1] + cluster_pos[i][1]/scale) for i, spin in enumerate(spins)})
        
        for i in range(len(solution)-1):
            s0 = solution[0]
            sol0 = {int(key): value for key, value in s0.items()}
            sol0 = dict(sorted(sol0.items()))
            
            s1 = solution[i+1]
            sol1 = {int(key): value for key, value in s1.items()}
            sol1 = dict(sorted(sol1.items()))
            colors = {node: "blue" if sol1[node] == -1 else "red" for node in sorted(graph.nodes)}

            highlight_nodes = [key for key in sol0 if key in sol1 and sol0[key] != sol1[key]]
            colors_droplets = [colors[hn] for hn in highlight_nodes]
    
            edge_colors = ['grey' if ((u, v) in J and (u in highlight_nodes and v in highlight_nodes)) or ((v, u) in J and (u in highlight_nodes and v in highlight_nodes)) else 'grey' for u, v in graph.edges]
            edge_widths = [5 if ((u, v) in J and (u in highlight_nodes and v in highlight_nodes)) or ((v, u) in J and (u in highlight_nodes and v in highlight_nodes))  else 0.25 for u, v in graph.edges]
            
            plt.figure(figsize=(7, 7))
            ax = plt.gca()
            ax.set_axis_off()
            other_nodes = set(graph.nodes) - set(highlight_nodes)
            nx.draw_networkx_nodes(graph, pos, node_color=list(colors.values()), node_size=100, alpha=0.3)
            nx.draw_networkx_nodes(graph, pos, nodelist=highlight_nodes, node_color=colors_droplets, node_size=100)

            nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=edge_widths)
            plt.show()


if __name__ == '__main__':
    draw_instance(df, file, scale)
    draw_solution(df, file, scale)
    draw_droplet(df, file, scale)