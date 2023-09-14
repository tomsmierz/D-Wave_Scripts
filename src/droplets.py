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

def filter_states_by_energy(states: list, energies: list, energy_cutoff: float) -> tuple:
    gs_energy = energies.min()
    mask = energies <= gs_energy + energy_cutoff
    filtered_states = states[mask]
    filtered_energies = energies[mask]
    return filtered_states, filtered_energies


def hamming_dist(d1, d2) -> int:
    if len(d1) != len(d2):
        raise ValueError("Vectors have different lengths")
    d1_array = np.array(d1)
    d2_array = np.array(d2)
    distance = np.sum(d1_array != d2_array)
    
    return distance

def find_droplets_hamming(states: list, energies: list, hamming_cutoff: int, permutation: Optional[list] = None) -> list:
    accepted_states = []
    accepted_energies = []
    perm_states = [states[i] for i in permutation]
    perm_energies = [energies[i] for i in permutation]

    for idx, state in enumerate(perm_states):
        if not accepted_states:
            accepted_states.append(state)
            accepted_energies.append(perm_energies[idx])
        else:
            h_list = []
            for drop in accepted_states:
                h = hamming_dist(state, drop)
                h_list.append(h)
            if all([h >= hamming_cutoff for h in h_list]):
                accepted_states.append(state)
                accepted_energies.append(perm_energies[idx])

    return accepted_states, accepted_energies

def find_max_set(n: int, states: list, energies: list, hamming_cutoff: int):
    max_set_st = []
    max_set_eng = []
    max_set_size = 0
    permutation = list(range(len(states)))
    
    for i in range(n):
        random.shuffle(permutation)
        accepted_states, accepted_energies = find_droplets_hamming(states, energies, hamming_cutoff, permutation)
        
        if len(accepted_states) > max_set_size:
            max_set_st = accepted_states
            max_set_eng = accepted_energies
            max_set_size = len(accepted_states)
    
    return max_set_st, max_set_eng


def read_json_files_first_batch(directory: str) -> dict:
    states_tn = {}
    energies_tn = {}
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(file):
            with open(file) as f:
                spinglass_state = json.load(f)
            name = spinglass_state["columns"][0][0][0:3]
            state_list = [list(state.values()) for state in spinglass_state["columns"][16][0]]
            state_array = np.array(state_list)
            energy_array = np.array(spinglass_state["columns"][17][0])
            
            if name in states_tn:
                if state_array.ndim > 0:
                    states_tn[name].append(state_array)
                if energy_array.ndim > 0:
                    energies_tn[name].append(energy_array)
            else:
                states_tn[name] = [state_array] if state_array.ndim > 0 else []
                energies_tn[name] = [energy_array] if energy_array.ndim > 0 else []

    # Concatenate the arrays for each name
    for name in states_tn.keys():
        if states_tn[name]:
            states_tn[name] = np.concatenate(states_tn[name])
        if energies_tn[name]:
            energies_tn[name] = np.concatenate(energies_tn[name])
    return states_tn, energies_tn

def states_dwave(states):
    states_dwave = states.iloc[:, 0:216]
    energies_dwave = P4["energy"].to_numpy()
    idx = [advantage_6_1_to_spinglass_int(int(k), 4) - 1 for k, _ in states_dwave.items()]
    states_dwave = states_dwave.to_numpy()
    # st_dw = states_dwave[:, idx]
    states_dwave_reordered = np.empty_like(states_dwave)
    for i, new_index in enumerate(idx):
        states_dwave_reordered[:, new_index] = states_dwave[:, i]
    states_dw, energies_dw = filter_states_by_energy(states_dwave_reordered, energies_dwave, cutoff_energy)
    return states_dw, energies_dw 

if __name__ == '__main__':
    cutoff_energy = 2.01
    cutoff_hamming = 10
    iterations = 100
    json_directory = os.path.join(cwd, "droplets", "P4", "CBFM-P", "P4_droplets_new")
    st_tn, eng_tn = read_json_files_first_batch(json_directory)
    
    instance_names = []  # To store instance names
    counts = []  # To store count_states_in_union

    for name in st_tn.keys():
        print("instance: ", name)
        P4 = pd.read_csv(os.path.join(cwd, "energies", "pegasus_random", "P4", "CBFM-P", f"{name}_2000_300.csv"),
                        index_col=0)
        states_dw, energies_dw = states_dwave(P4)
        states_tn, energies_tn = st_tn[name], eng_tn[name]
        droplet_states_dwave, droplet_energies_dwave = find_max_set(iterations, states_dw, energies_dw, cutoff_hamming)
        union = list(set(map(tuple, droplet_states_dwave)) | set(map(tuple, states_tn)))
        unique_states_tn = set(map(tuple, states_tn))
        independent_states_union, independent_energy_union = find_max_set(iterations, union, np.zeros(len(union)), cutoff_hamming)
            
        count_states_in_union = 0
        for state in independent_states_union:
            hamming_distances = [hamming_dist(state_tn, state) for state_tn in unique_states_tn]
            if any(distance < cutoff_hamming for distance in hamming_distances):
                count_states_in_union += 1

        print("Fraction of states from TN in all states:", count_states_in_union / len(independent_states_union))
        counts.append(count_states_in_union / len(independent_states_union))
        instance_names.append(name)

    # Sort the results based on counts
    sorted_results = sorted(zip(instance_names, counts), key=lambda x: x[0])
    sorted_names, sorted_values = zip(*sorted_results)

    # Plot the graph
    plt.figure(figsize=(10, 5))
    plt.ylim((0, 1))
    plt.bar(sorted_names, sorted_values)
    plt.xlabel("Instance index")
    plt.ylabel("Fraction of TN droplets")
    plt.title("CBFM-P")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()