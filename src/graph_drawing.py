import networkx as nx
import os
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random

rng = random.default_rng()

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    graph = nx.Graph()

    df = pd.read_csv(os.path.join(root, "instances", "example", "diagonal_5x5.txt"), header=None, sep=" ",
                     names=["n1", "n2", "v"])

    sol_df = pd.read_csv(os.path.join(root, "energies", "example", "diagonal_5x5.csv"), sep=";")

    # solution is saved as Julia dict which is read as string. We perform string manipulation to get python dict.
    a = sol_df["ig_states"][0]
    a = a.replace("Dict(", "")
    a = a.replace(")", "")
    splited = a.split(",")
    splited = [s.replace(" ", "") for s in splited]
    splited = [s.split("=>") for s in splited]

    solution = {eval(d[0]): eval(d[1]) for d in splited}
    solution = dict(sorted(solution.items()))
    print(solution)

    h = {}
    J = {}
    for row in df.itertuples():
        if row.n1 == row.n2:
            h[row.n1] = row.v
        else:
            J[(row.n1, row.n2)] = row.v

    graph.add_edges_from(J.keys())

    colors = {node: "red" if solution[node] == -1 else "blue" for node in sorted(graph.nodes)}
    plt.figure(figsize=(16, 16))
    nx.draw(graph, node_color=list(colors.values()), edge_color="grey")
    plt.show()