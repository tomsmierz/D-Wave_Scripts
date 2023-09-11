import os
import json

import pandas as pd
import numpy as np
import dwave_networkx as dnx

from dwave.system import DWaveSampler
from renumeration import advantage_6_1_to_spinglass, advantage_6_1_to_spinglass_int
from scipy.spatial.distance import hamming
from copy import deepcopy
from tqdm import tqdm
from typing import Optional

cwd = os.getcwd()

# TODO: add version to ignore trivial symmetries, h = min(h(x, y), h(-x, y))
# TODO: add PT-data
# TODO: add SBM-data
# TODO: X random choices of initial droplet, create set, chose biggest set


def df_to_states(df: pd.DataFrame, energy_cutoff: float) -> dict:
    gs_energy = df["energy"].min()
    df = df[df["energy"] <= gs_energy + energy_cutoff]  # this operation preserves indices of the original dataframe
    return df.to_dict("index")


def dict_hamming(d1: dict, d2: dict) -> int:
    if len(d1) != len(d2):
        raise ValueError("dictionaries have different length")
    s = 0
    for key in d1.keys():
        if d1[key] != d2[key]:
            s += 1
    return s


def find_droplets_hamming(states: dict, hamming_cutoff: int, size: int, permutation: Optional[dict] = None) -> list:
    accepted_states = []
    for idx, row in tqdm(states.items()):
        temp = deepcopy(row)
        del temp["energy"], temp["num_occurrences"]
        row_spins = {advantage_6_1_to_spinglass_int(int(k), size): v for k, v in temp.items()}
        row_spins = dict(sorted(row_spins.items()))
        state = {"index": idx, "spins": row_spins, "energy": row["energy"]}
        if not accepted_states:
            accepted_states.append(state)
        else:
            h_list = []
            for drop in accepted_states:
                drop_dict =drop["spins"]
                h = dict_hamming(row_spins, drop_dict)
                h_list.append(h)
            if all([h >= hamming_cutoff for h in h_list]):
                accepted_states.append(state)
    return accepted_states


def read_json_files_first_batch(directory: str) -> dict:
    json_files = {}
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(file):
            with open(file) as f:
                spinglass_state = json.load(f)
            name = spinglass_state["columns"][0][0][0:3]
            state_list = []
            for state in spinglass_state["columns"][15][0]:
                temp = {int(k): v for k, v in state.items()}
                state_list.append(temp)
            if name in json_files.keys():
                json_files[name].append(state_list)
            else:
                json_files[name] = [state_list]
    return json_files


if __name__ == '__main__':
    cutoff_energy = 1.01
    cutoff_hamming = 20
    json_directory = os.path.join(cwd, "..", "droplets", "P4", "CFBM", "SpinGlass", "P4_droplets_new_i6-10")
    json_files = read_json_files_first_batch(json_directory)
    for name in json_files.keys():
        print("instance: ", name)
        P4 = pd.read_csv(os.path.join(cwd, "..", "energies", "pegasus_random", "P4", "CBFM-P", f"{name}_2_5000.csv"),
                         index_col=0)
        states = df_to_states(P4, cutoff_energy)
        droplets_dwave = find_droplets_hamming(states, cutoff_hamming, 4)
        print("droplets dwave: ", len(droplets_dwave))
        for transformation in json_files[name]:
            print("droplets spinglass: ", len(transformation))
            for d1 in transformation:
                for d2 in droplets_dwave:
                    if d1 == d2["spins"]:
                        print("yey")
                    else:
                        print("hamming: ", dict_hamming(d1, d2["spins"]))


