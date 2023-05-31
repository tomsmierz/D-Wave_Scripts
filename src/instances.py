import os
import pandas as pd
import numpy as np
from minorminer import find_embedding
import dimod
import dwave_networkx as dnx
import pickle

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
    # target = dnx.pegasus_graph(16, nice_coordinates=True)
    target = dnx.zephyr_graph(19)
    print("searching for embedding...")
    embedding = find_embedding(J, target)
    if embedding:
        print("embedding found")
    else:
        print("no embedding found")
    print(embedding)

    # with open(path, "w") as f:
    #     f.write(f"# offset: {offset} \n")
    #     for i, v in h.items():
    #         f.write(f"{i} {i} {v}\n")
    #     for (i, j), v in J.items():
    #         f.write(f"{i} {j} {v}\n")


def embedding_spinglass(Q):
    embedding = {}
    h, J, offset = dimod.qubo_to_ising(Q)
    i = 15
    while not embedding:
        i += 1
        print(f"searching for embedding in Z{i}")
        target = dnx.zephyr_graph(i)
        embedding = find_embedding(J, target)
        if not embedding:
            print(f"no embedding found Z{i}")
    print(f"embedding found for Z{i}")
    with open("embedding.pkl", "wb") as f:
        pickle.dump(embedding, f)
    print("embedding saved")


def vectorize(h: dict, J: dict):
    # We assume that h an J are sorted
    h_vect = np.array(list(h.values()))
    n = len(h_vect)
    J_vect = np.zeros((n, n))
    for key, value in J.items():
        J_vect[key[0]][key[1]] = value
    return h_vect, J_vect


if __name__ == "__main__":
    Q = load_instance(os.path.join(cwd, f"..\\instances\\matyas_instances\\G1_py.csv"))
    embedding_spinglass(Q)
    # for i in range(1, 11):
    #     Q = load_instance(os.path.join(cwd, f"..\\instances\\matyas_instances\\G{i}_py.csv"))
    #     convert_to_spinglass(Q, os.path.join(cwd, f"..\\instances\\matyas_instances\\G{i}.txt"))
