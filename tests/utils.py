import dwave_networkx
import networkx as nx
import pandas as pd
import os

from src.pegasus import get_pegasus

wpath = os.getcwd()


def read_result(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=" ", index_col=False, header=None)
    df = df.drop(columns=[0, 1, 219])
    return df


def test_energy(result: pd.DataFrame, size: int, number: int):
    for index, row in df.iterrows():
        energy = row[2]
        for column in df.columns[1::]:
            pass


