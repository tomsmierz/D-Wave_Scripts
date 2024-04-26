import networkx as nx
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dwave_networkx as dnx

from numpy import random
from renumeration import advantage_6_1_to_spinglass, advantage_6_1_to_spinglass_int

rng = random.default_rng()
instance_size = 4
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(root, "instances", "pegasus_random", "P4", "RAU", "001_sg.txt"), header=None, sep=" ",
                     names=["n1", "n2", "v"])
    
    h = {}
    J = {}
    for row in df.itertuples():
        if row.n1 == row.n2:
            h[row.n1-1] = row.v
        else:
            J[(row.n1-1, row.n2-1)] = row.v

    fig = plt.figure(figsize=(10, 7)) 

    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 0.05])

    # Create the graph subplot
    ax_graph = fig.add_subplot(gs[0])
    p4 = dnx.pegasus_graph(4, nice_coordinates=True)
    labels = {node: dnx.pegasus_coordinates(4).nice_to_linear(node) for node in p4.nodes()}
    sm = plt.cm.ScalarMappable(cmap=plt.cm.seismic, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    dnx.draw_pegasus(p4, node_color='black', node_size=30, edge_color=J.values(), edge_cmap=plt.cm.seismic,
                     edge_vmin=-1, edge_vmax=1, ax=ax_graph)
    ax_graph.set_aspect('equal', adjustable='box')

    # Create the colorbar subplot with adjusted width and height
    ax_cbar = fig.add_subplot(gs[1])
    cbar = plt.colorbar(sm, cax=ax_cbar)  # Adjust aspect here
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout()  # Zapewnia, że nie będzie nakładania się na siebie

    plt.show()


