from dwave.system import EmbeddingComposite, DWaveSampler
import dwave_networkx as dnx
import networkx as nx
from numpy.random import default_rng
import pandas as pd
from dwave.cloud import Client
from tqdm import tqdm

client = Client.from_config()
num_reads = 5000

sampler = DWaveSampler(solver="Advantage_system6.1")

graph = dnx.pegasus_graph(16, data=False, fabric_only=False)

real_nodes = sampler.nodelist
real_edges = sampler.edgelist

broken_nodes = len(list(set(graph.nodes) - set(real_nodes)))

print(sampler.properties["j_range"])
