import os
import json
import random

import pandas as pd
import numpy as np
import dwave_networkx as dnx
import matplotlib.pyplot as plt

from collections import namedtuple
from dwave.system import DWaveSampler
from renumeration import advantage_6_1_to_spinglass, advantage_6_1_to_spinglass_int
from scipy.spatial.distance import hamming
from copy import deepcopy
from tqdm import tqdm
from typing import Optional

cwd = os.getcwd()

def hamming_dist(d1, d2) -> int:
    if len(d1) != len(d2):
        raise ValueError("Vectors have different lengths")
    distance = np.sum(d1 != d2)
    return distance

def filter_states_by_energy(states, energies, energy_cutoff):
    gs_energy = energies.min()
    mask = energies <= gs_energy + energy_cutoff
    filtered_states = states[mask]
    filtered_energies = energies[mask]
    return filtered_states, filtered_energies


def find_droplets_hamming(state_energy_tuple: namedtuple, hamming_cutoff: int, permutation: Optional[list] = None):
    accepted_states = [] 
    accepted_energies = [] 
    AcceptedStateEnergy = namedtuple('AcceptedStateEnergy', ['state', 'energy'])

    if permutation is not None:
        perm_states = [state_energy_tuple.state[i] for i in permutation]
        perm_energies = [state_energy_tuple.energy[i] for i in permutation]
    else:
        perm_states = state_energy_tuple.state
        perm_energies = state_energy_tuple.energy

    for idx, state in enumerate(perm_states):
        if not accepted_states:
            accepted_states.append(state)
            accepted_energies.append(perm_energies[idx])
        elif any(np.array_equal(state, accepted_state) for accepted_state in accepted_states):
            pass
        else:
            h_list = []
            for drop in accepted_states:
                h = hamming_dist(state, drop)
                h_list.append(h)
            if all(h >= hamming_cutoff for h in h_list):
                accepted_states.append(state)
                accepted_energies.append(perm_energies[idx])
    return AcceptedStateEnergy(np.array(accepted_states), np.array(accepted_energies))


def find_max_set(n: int, state_energy_tuple: namedtuple, hamming_cutoff: int):
    max_set_size = 0
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    permutation = list(range(len(state_energy_tuple.state)))
    for i in range(n):
        random.shuffle(permutation)
        accepted_state_energy_tuple = find_droplets_hamming(state_energy_tuple, hamming_cutoff, permutation)
        if len(accepted_state_energy_tuple.state) > max_set_size:
            new_se = StateEnergy(accepted_state_energy_tuple.state, accepted_state_energy_tuple.energy)
            max_set_size = len(accepted_state_energy_tuple.state)
    return new_se

def create_union(states1: namedtuple, states2: namedtuple):
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    new_states = np.vstack((states1.state, states2.state))
    new_engs = np.hstack((states1.energy, states2.energy))
    _, idx = np.unique(new_states, axis=0, return_index=True)

    return StateEnergy(new_states[idx], new_engs[idx])

def count_states(independent_states_union, unique_states_tn):
    count_states_in_union = 0
    for state in independent_states_union.state:
        hamming_distances = [hamming_dist(state_tn, state) for state_tn in unique_states_tn.state]
        if any(distance < cutoff_hamming for distance in hamming_distances):
            count_states_in_union += 1
    return count_states_in_union


def array_from_dict(dict_list):
    # Determine the number of states and maximum index to determine the size of the resulting array
    num_states = len(dict_list)
    max_index = max(int(key) for d in dict_list for key in d.keys())
    result_array = np.zeros((num_states, max_index))
    for i, d in enumerate(dict_list):
        for key, value in d.items():
            result_array[i, int(key) - 1] = value
    return result_array

def read_json_files(directory):
    instance_data = {}
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
        if os.path.isfile(file):
            with open(file) as f:
                json_data = json.load(f)
                instance_name = json_data['columns'][json_data['colindex']['lookup']['instance']-1][0].split('_')[0]
                energy_data = json_data['columns'][json_data['colindex']['lookup']['drop_eng']-1][0]
                state_data = json_data['columns'][json_data['colindex']['lookup']['ig_states']-1][0]
                state_data_np = array_from_dict(state_data)
                energy_data_np = np.array(energy_data)
                # Check if the instance_name is already in instance_data
                if instance_name in instance_data:
                    # If it is, add the state and energy data to the existing named tuple
                    existing_state_energy = instance_data[instance_name]
                    new_state_data = np.concatenate((existing_state_energy.state, state_data_np))
                    new_energy_data = np.concatenate((existing_state_energy.energy, energy_data_np))
                    # Remove duplicate rows from new_state_data and filter corresponding energy values
                    unique_rows, unique_indices = np.unique(new_state_data, axis=0, return_index=True)
                    filtered_energy_data = new_energy_data[unique_indices]
                    instance_data[instance_name] = StateEnergy(unique_rows, filtered_energy_data)                
                else:
                    # If it's not, create a new named tuple for the instance_name
                    instance_data[instance_name] = StateEnergy(state_data_np, energy_data_np)
    return instance_data

def states_dwave(states, cutoff_energy, n, t):
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    states_dwave = states.iloc[:, 0:n]
    energies_dwave = np.array(states["energy"])
    idx = [advantage_6_1_to_spinglass_int(int(k), t) - 1 for k, _ in states_dwave.items()]
    states_dwave = np.array(states_dwave)
    states_dwave_reordered = np.empty_like(states_dwave)
    for i, new_index in enumerate(idx):
        states_dwave_reordered[:, new_index] = states_dwave[:, i]
    filtered_states, filtered_energies = filter_states_by_energy(states_dwave_reordered, energies_dwave, cutoff_energy) 
    filtered_state_energy_tuple = StateEnergy(filtered_states, filtered_energies)
    return filtered_state_energy_tuple

if __name__ == '__main__':
    cutoff_energy = 1.01
    cutoff_hamming = 20
    iterations = 100
    n = 216
    t = 4 
    json_directory = os.path.join(cwd, "droplets", "P4", "RCO", "P4_droplets_new")
    stn = read_json_files(json_directory)
    
    instance_names = []  # To store instance names
    counts = []  # To store count_states_in_union

    for name in stn.keys():
        # print("instance: ", name)
        state_energy_net = stn[name]
        P4 = pd.read_csv(os.path.join(cwd, "energies", "pegasus_random", "P4", "RCO", f"{name}_2000_300.csv"),
                        index_col=0)
        state_energy_tuple = states_dwave(P4, cutoff_energy, n, t)
        state_energy_tn = find_max_set(iterations, state_energy_net, cutoff_hamming)
        state_energy_dw = find_max_set(iterations, state_energy_tuple, cutoff_hamming)

        concatenated_se = create_union(state_energy_dw, state_energy_tn)
        independent_union = find_max_set(iterations, concatenated_se, cutoff_hamming)
    
        count_states_in_union = count_states(independent_union, state_energy_tn)
        counts.append(count_states_in_union / len(independent_union.energy))
        instance_names.append(name)

    # Sort the results based on counts
    sorted_results = sorted(zip(instance_names, counts), key=lambda x: x[0])
    sorted_names, sorted_values = zip(*sorted_results)

    # Plot the graph
    plt.figure(figsize=(10, 5))
    # plt.ylim((0, 1))
    plt.bar(sorted_names, sorted_values)
    plt.xlabel("Instance index")
    plt.ylabel("Fraction of TN droplets")
    plt.title("RCO")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
