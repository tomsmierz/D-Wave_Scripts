import dwave_networkx as dnx
import pandas as pd
from dwave.system import DWaveSampler

from src.graph_operations import find_best_mapping

df = pd.read_csv(
    "C:\\Users\\tsmierzchalski\\PycharmProjects\\D-Wave_Scripts\\src\\z1.csv"
)

spin_glass_dict = {row.first - 1: row.second for row in df.itertuples(index=False)}

df2 = pd.read_csv(
    "C:\\Users\\tsmierzchalski\\PycharmProjects\\D-Wave_Scripts\\energies\\zephyr_random\\Z1\\RAU\\001_2000_300.csv"
)
df2 = df2.drop("Unnamed: 0", axis="columns")

dwave_dict = {column[0]: column[1][0] for column in df2.items()}
source = dnx.zephyr_graph(1, coordinates=True)
sampler = DWaveSampler(solver="Advantage2_prototype1.1")

target = sampler.to_networkx_graph()
mappings = [mapp for mapp in dnx.zephyr_sublattice_mappings(source, target)]

mapping, _, _, _ = find_best_mapping(mappings, sampler, source)

print(spin_glass_dict)
print(dwave_dict)

spin_glass_converted = {mapping(dnx.zephyr_coordinates(1).linear_to_zephyr(item)): value
                        for item, value
                        in spin_glass_dict.items()}

print(spin_glass_converted)
