import os
import json
import random

import pandas as pd
import numpy as np
import dwave_networkx as dnx
import matplotlib.pyplot as plt

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

def find_droplets_hamming_union(states: list, hamming_cutoff: int, size: int, permutation: Optional[dict] = None) -> list:
    random.shuffle(states)  # Shuffle the order of states
    # Sort the shuffled states by energy (lowest to highest)

    accepted_states = []
#     for row in tqdm(states):
    for row in states:

        temp = deepcopy(row)
        row_spins = dict(sorted(row.items()))
        row_list = list(row_spins.values())
        if not accepted_states:
            accepted_states.append(row_spins)
        else:
            h_list = []
            for drop in accepted_states:
                drop_list = list(drop.values())
                h = hamming(drop_list, row_list) * len(row_list)
                h_list.append(h)
            if all([h >= hamming_cutoff for h in h_list]):
                accepted_states.append(row_spins)
    return accepted_states


def find_max_set(n: int, states: dict, hamming_cutoff: int, size: int, permutation: Optional[dict] = None):
    max_set = []
    max_set_size = 0
    
    for i in range(n):
        accepted_states = find_droplets_hamming(states, hamming_cutoff, size, permutation)
        
        if len(accepted_states) > max_set_size:
            max_set = accepted_states
            max_set_size = len(accepted_states)
    
    return max_set

def find_max_set_union(n: int, states: dict, hamming_cutoff: int, size: int, permutation: Optional[dict] = None):
    max_set = []
    max_set_size = 0
    
    for i in range(n):
        accepted_states = find_droplets_hamming_union(states, hamming_cutoff, size, permutation)
        
        if len(accepted_states) > max_set_size:
            max_set = accepted_states
            max_set_size = len(accepted_states)
    
    return max_set


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
    json_directory = os.path.join(cwd, "droplets", "P4", "CBFM-P", "P4_droplets_new_i6-10")
    json_files = read_json_files_first_batch(json_directory)
    
    instance_names = []  # To store instance names
    counts = []  # To store count_states_in_union

    for name in json_files.keys():
        print("instance: ", name)
        P4 = pd.read_csv(os.path.join(cwd, "energies", "pegasus_random", "P4", "CBFM-P", f"{name}_2_5000.csv"),
                         index_col=0)
        states = df_to_states(P4, cutoff_energy)
        droplets_dwave = find_max_set(50, states, cutoff_hamming, 4)
        
        droplets_union = []

        # Create a set to keep track of unique dictionaries
        unique_dicts = set()
            
        # Append dictionaries from droplets_dwave
        for d2 in droplets_dwave:
            unique_dict_key = tuple(sorted(d2['spins'].items()))
            if unique_dict_key not in unique_dicts:
                droplets_union.append(d2['spins'])
                unique_dicts.add(unique_dict_key)
#             print(unique_dicts)
        for transformation in json_files[name]:
            # Append dictionaries from transformation
            for d1 in transformation:
                unique_dict_key = tuple(sorted(d1.items()))
                if unique_dict_key not in unique_dicts:
                    droplets_union.append(d1)
                    unique_dicts.add(unique_dict_key)
        
        independent_droplets_union = find_max_set_union(50, droplets_union, cutoff_hamming, 4)

        # Count how many states from transformation are in independent_droplets_union
        count_states_in_union = sum(1 for state in transformation if state in independent_droplets_union)
        count_states_in_union = count_states_in_union/len(independent_droplets_union)
        print("How many tensor network droplets are in maximal independent set of all droplets:", count_states_in_union)
        instance_names.append(name)
        counts.append(count_states_in_union)

    # Sort the instance names and counts based on instance numbers
    sorted_data = sorted(zip(instance_names, counts))
    instance_names, counts = zip(*sorted_data)

    # Plot the graph
    plt.figure(figsize=(10, 5))
    plt.bar(instance_names, counts)
    plt.xlabel("Instance Index")
    plt.ylabel("Droplets (Normalized)")
#     plt.title("Count Droplets vs. Instance Number")
    # plt.xticks(range(0, 5), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()