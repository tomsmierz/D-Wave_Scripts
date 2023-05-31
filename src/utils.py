import pandas as pd
import os


def load_pegasus(path: str, name: str = "001"):
    df = pd.read_csv(
        os.path.join(path, f"{name}.txt"),
        sep=" ",
        index_col=False,
        skiprows=1,
        header=None,
    )
    h = {}
    J = {}
    for index, row in df.iterrows():
        if row[0] == row[1]:
            h[int(row[0] - 1)] = row[2]
            # h[int(row[0])] = row[2]
        else:
            J[(int(row[0] - 1), int(row[1] - 1))] = row[2]
            # J[(int(row[0]), int(row[1]))] = row[2]
    return h, J


def load_pegasus_tuple(path: str, name: str = "001"):
    df = pd.read_csv(
        os.path.join(path, f"{name}.txt"),
        sep=";",
        index_col=False,
        skiprows=1,
        header=None,
    )
    h = {}
    J = {}
    for index, row in df.iterrows():
        if row[0] == row[1]:
            # h[int(row[0] - 1)] = row[2]\
            h[eval(row[0])] = row[2]
        else:
            # J[(int(row[0] - 1), int(row[1] - 1))] = row[2]
            J[(eval(row[0]), eval(row[1]))] = row[2]
    return h, J
