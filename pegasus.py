from dwave.system import AutoEmbeddingComposite, DWaveSampler
import dwave_networkx as dnx
import networkx as nx
from numpy.random import default_rng
import pandas as pd
from dwave.cloud import Client
from tqdm import tqdm
import dwave.inspector
import matplotlib.pyplot as plt

df = pd.read_csv("instances\\P4\\001.txt", sep=" ")


client = Client.from_config()
num_reads = 5000

mim_time = 0.5
default_time = 20
max_time = 200

sampler = DWaveSampler(solver="Advantage_system6.1", auto_scale=True)

graph = dnx.pegasus_graph(16, data=False, fabric_only=False)

real_nodes = sampler.nodelist
real_edges = sampler.edgelist

broken_nodes = list(set(graph.nodes) - set(real_nodes))
broken_edges = list(set(graph.edges) - set(real_edges))

solver = AutoEmbeddingComposite(sampler)
#solver.sample_ising(num_reads=1, label="Pegasus", )
