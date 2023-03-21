import pandas as pd
import dwave_networkx as dnx
from src.generate_zephyr_instances import find_map
from dwave.system import DWaveSampler


df = pd.read_csv(
    "C:\\Users\\tsmierzchalski\\PycharmProjects\\D-Wave_Scripts\\src\\z1.csv"
)

spin_glass_dict = {row.first - 1: row.second for row in df.itertuples(index=False)}

df2 = pd.read_csv(
    "C:\\Users\\tsmierzchalski\\PycharmProjects\\D-Wave_Scripts\\energies\\zephyr_random\\Z1\\RAU\\001_2000_300.csv"
)
df2 = df2.drop("Unnamed: 0", axis="columns")

dwave_dict = {column[0]: column[1][0] for column in df2.items()}
z1 = dnx.zephyr_graph(1, coordinates=True)
sampler = DWaveSampler(solver="Advantage2_prototype1.1")
mapping, _, _, _ = find_map(z1, sampler)

print(spin_glass_dict)
print(dwave_dict)

spin_glass_converted = {mapping(dnx.zephyr_coordinates(1).linear_to_zephyr(item)): value for item, value in spin_glass_dict.items()}


print(spin_glass_converted)
