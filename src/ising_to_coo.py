import dimod
from dimod.serialization import coo
import pickle
import os
from src.generate_pegasus_instances import nice_to_spin_glass, find_map, reverse_pegasus_sublattice_mapping
import dwave_networkx as dnx
import networkx as nx
import pandas as pd
from tqdm import tqdm

cwd = os.getcwd()


if __name__ == "__main__":
    for i in tqdm(range(1, 101)):
        name = f"00{i}"[-3:]
        name_full = name + "_sg.txt"
        instance = pd.read_csv(os.path.join(cwd, "..", "instances", "pegasus_random", "P16", "CBFM-P", name_full),
                               sep=" ", index_col=False, header=None, comment="#", names=["s1", "s2", "v"])
        h = {}
        J = {}

        for row in instance.itertuples():
            if row.s1 == row.s2:
                h[int(row.s1)-1] = int(row.v)
            else:
                J[(int(row.s1)-1, int(row.s2)-1)] = int(row.v)
        print(len(h.keys()))
        for i in range(5400):
            if i not in h.keys():
                h[i] = 0
                J[(i, i+1)] = 0
        print(len(h.keys()))
        bqm = dimod.BinaryQuadraticModel(vartype="SPIN")
        bqm = bqm.from_ising(h, J)
        print(bqm.num_variables)
        with open(os.path.join(cwd, "..", f"instances/pegasus_random/P16/CBFM-P/COO/{name}.txt"), "w") as f:
            coo.dump(bqm, f)



