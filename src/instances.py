import os
import pandas as pd
import numpy as np
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
from minorminer import find_embedding
import dimod

cwd = os.getcwd()
rng = np.random.default_rng()

def load_instance(path):
    df = pd.read_csv(path, sep=",")
    Q = {}
    for row in df.itertuples(index=False):
        Q[(row.ICOORD, row.JCOORD)] = row.QVALUE
    return Q


def convert_to_spinglass(Q, path):
    h, J, offset = dimod.qubo_to_ising(Q)
    h = dict(sorted(h.items()))
    J = dict(sorted(J.items()))
    with open(path, "w") as f:
        f.write(f"# offset: {offset} \n")
        for i, v in h.items():
            f.write(f"{i} {i} {v}\n")
        for (i, j), v in J.items():
            f.write(f"{i} {j} {v}\n")


def vectorize(h: dict, J: dict):
    # We assume that h an J are sorted
    h_vect = np.array(list(h.values()))
    n = len(h_vect)
    J_vect = np.zeros((n, n))
    for key, value in J.items():
        J_vect[key[0]][key[1]] = value
    return h_vect, J_vect


if __name__ == "__main__":
    for i in range(1, 11):
        Q = load_instance(os.path.join(cwd, f"..\\instances\\matyas_instances\\G{i}_py.csv"))
        convert_to_spinglass(Q, os.path.join(cwd, f"..\\instances\\matyas_instances\\G{i}.txt"))

