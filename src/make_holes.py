from dwave.system import AutoEmbeddingComposite, DWaveSampler
import dwave_networkx as dnx
import networkx as nx
from numpy.random import default_rng
import pandas as pd
from dwave.cloud import Client

import dwave.inspector

import os

client = Client.from_config()
path = os.getcwd()
sampler = DWaveSampler(solver="Advantage_system6.1")

graph = dnx.pegasus_graph(16, data=False, fabric_only=False)

real_nodes = sampler.nodelist
real_edges = sampler.edgelist

broken_nodes = list(set(graph.nodes) - set(real_nodes))

directory = "/home/tsmierzchalski/PycharmProjects/D-Wave_Scripts/instances/P16"
output_directory = "/home/tsmierzchalski/PycharmProjects/D-Wave_Scripts/instances/instances_with_holes/P16"

for filename in os.scandir(directory):
    c = 0
    if filename.is_file():
        #print(filename.path)
        with open(filename.path, "r") as f:
            data = f.readlines()

        with open(os.path.join(output_directory, os.path.basename(filename.path)), "w") as f:

            for line in data:
                to_write = True
                for node in broken_nodes:
                    if " " + str(node) + " " in line.strip("\n"):
                        to_write = False
                        c += 1
                if to_write:
                    f.write(line)
    print(c)

