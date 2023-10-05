import copy
import os
import json
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple
from renumeration import advantage_6_1_to_spinglass, advantage_6_1_to_spinglass_int
from scipy.spatial.distance import hamming
from typing import Optional, Union
from tqdm import tqdm

# TODO: use @dataclass to store droplets?
# TODO: move helper functions to another file?

# Constants
CUTOFF_ENERGY = 9.4
CUTOFF_HAMMING = 20
ITERATIONS = 100
APPROX_RATIO = 1e-3
BETA = 0.5
ENG = 0.94
BD = 8

# Instance characteristic
TOPOLOGY = "pegasus"
INSTANCE_SYMBOL = "P4"
INSTANCE_TYPE = "CBFM-P"
TOPOLOGY_SIZE = 4

# Directories
cwd = os.getcwd()
json_directory = os.path.join(cwd, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, f"{INSTANCE_SYMBOL}_droplets_i1-3")
dwave_directory = os.path.join(cwd, "energies", f"{TOPOLOGY}_random_aggregated", INSTANCE_SYMBOL, INSTANCE_TYPE)
minimum_path = os.path.join(cwd, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "minimum.csv")

# Type aliases
vector = Union[np.ndarray, list]

# Helper functions

def hamming_dist(v1: vector, v2: vector) -> int:
    return hamming(v1, v2) * len(v1)


def get_state_energy_from_dwave(data: pd.DataFrame, cutoff_energy: float, instance_size: int, ground_eng: float) -> namedtuple:
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    states_dwave = copy.deepcopy(data)
    states_dwave.drop(["energy", "num_occurrences", "annealing_time", "num_reads", "pause_time", "reverse"], axis=1, inplace=True)
    energies_dwave = data["energy"].to_numpy()

    renum = {k: advantage_6_1_to_spinglass_int(int(k), instance_size) - 1 for k in states_dwave.columns}
    states_dwave.rename(columns=renum, inplace=True)

    states_dwave_reordered = states_dwave[sorted(list(states_dwave.columns))]
    states_dwave_reordered = states_dwave_reordered.to_numpy()

    filtered_states, filtered_energies = filter_states_by_energy(states_dwave_reordered, energies_dwave, cutoff_energy, ground_eng)
    filtered_state_energy_tuple = StateEnergy(filtered_states, filtered_energies)

    return filtered_state_energy_tuple


def filter_states_by_energy(states: np.ndarray, energies: np.ndarray, cutoff_energy: float, ground_eng: float) -> (np.ndarray, np.ndarray):
    """
    states: Expected to be square matrix, with states in rows
    """
    mask = energies <= ground_eng + cutoff_energy
    filtered_states = states[mask]
    filtered_energies = energies[mask]
    return filtered_states, filtered_energies


def find_droplets_hamming(state_energy_tuple: namedtuple, hamming_cutoff: int, ground_eng: float, energy_cutoff: float, permutation: Optional[list] = None):
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
            for (i, drop) in enumerate(accepted_states):
                h = hamming_dist(state, drop)
                h_list.append(h)
                eng = (abs(perm_energies[idx] - ground_eng))
            if eng <= energy_cutoff and all(h >= hamming_cutoff for h in h_list):
                accepted_states.append(state)
                accepted_energies.append(perm_energies[idx])
    return AcceptedStateEnergy(np.array(accepted_states), np.array(accepted_energies))


def find_max_set(iterations: int, state_energy_tuple: namedtuple, hamming_cutoff: int, ground_eng: float, energy_cutoff: float):
    set_size = 0
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    permutation = list(range(len(state_energy_tuple.state)))

    for i in range(iterations):
        random.shuffle(permutation)
        accepted_state_energy_tuple = find_droplets_hamming(state_energy_tuple, hamming_cutoff, ground_eng, energy_cutoff, permutation)
        if len(accepted_state_energy_tuple.state) > set_size:
            new_state_energy = StateEnergy(accepted_state_energy_tuple.state, accepted_state_energy_tuple.energy)
            set_size = len(accepted_state_energy_tuple.state)

    return new_state_energy


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
        if any(distance < CUTOFF_HAMMING for distance in hamming_distances):
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


def read_json_files(directory, beta, eng, bd, cutoff_energy, df_min) -> dict:
    instance_data = {}
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
        if os.path.isfile(file):
            with open(file) as f:
                json_data = json.load(f)
                if json_data['columns'][json_data['colindex']['lookup']['β']-1][0] == beta and json_data['columns'][json_data['colindex']['lookup']['eng']-1][0] == eng and json_data['columns'][json_data['colindex']['lookup']['bond_dim']-1][0] == bd:
                    instance_name = json_data['columns'][json_data['colindex']['lookup']['instance']-1][0].split('_')[0]
                    energy_data = json_data['columns'][json_data['colindex']['lookup']['drop_eng']-1][0]
                    state_data = json_data['columns'][json_data['colindex']['lookup']['ig_states']-1][0]
                    state_data_np = array_from_dict(state_data)
                    energy_data_np = np.array(energy_data)
                    ground_eng = df_min[df_min.index == instance_name]['Ground energy'].values[0]
                    filtered_states, filtered_energies = filter_states_by_energy(state_data_np, energy_data_np, cutoff_energy, ground_eng)

                    # Check if the instance_name is already in instance_data
                    if instance_name in instance_data:
                        # If it is, add the state and energy data to the existing named tuple
                        existing_state_energy = instance_data[instance_name]
                        new_state_data = np.concatenate((existing_state_energy.state, filtered_states))
                        new_energy_data = np.concatenate((existing_state_energy.energy, filtered_energies))
                        # Remove duplicate rows from new_state_data and filter corresponding energy values
                        unique_rows, unique_indices = np.unique(new_state_data, axis=0, return_index=True)
                        filtered_energy_data = new_energy_data[unique_indices]
                        instance_data[instance_name] = StateEnergy(unique_rows, filtered_energy_data)                
                    else:
                        # If it's not, create a new named tuple for the instance_name
                        instance_data[instance_name] = StateEnergy(filtered_states, filtered_energies)
                else:
                    pass
    return instance_data


def compute_dwave_spinglass_droplets(dwave_path, spinglass_path, minimum_path, beta, eng, bd):
    result_names = []  # To store instance names
    counts_dw = []  # To store count_states_in_union
    counts_tn = []
    df = pd.read_csv(minimum_path, index_col=0)
    df.index = df.index.map(lambda x: str(x).zfill(3))
    spinglass_states = read_json_files(spinglass_path, beta, eng, bd, CUTOFF_ENERGY, df)

    for name, state_energy_net in tqdm(spinglass_states.items()):
        # print("instance: ", name)
        ground_eng = df[df.index == name]['Ground energy'].values[0]
        instance_df = pd.read_csv(os.path.join(dwave_path, f"{name}.csv"), index_col=0)
        state_energy_tuple = get_state_energy_from_dwave(instance_df, CUTOFF_ENERGY, TOPOLOGY_SIZE, ground_eng)
        energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)
        state_energy_sg = find_max_set(ITERATIONS, state_energy_net, CUTOFF_HAMMING, ground_eng, energy_cutoff)
        state_energy_dwave = find_max_set(ITERATIONS, state_energy_tuple, CUTOFF_HAMMING,  ground_eng, energy_cutoff)
        
        count_dw = len(state_energy_dwave.energy)
        count_tn = len(state_energy_sg.energy)

        counts_dw.append(count_dw)
        counts_tn.append(count_tn)

        result_names.append(name)

    return result_names, counts_dw, counts_tn


def count_tn_droplets(dwave_path, spinglass_path, minimum_path, beta, eng, bd):
    result_names = []  # To store instance names
    counts = []  # To store count_states_in_union
    df = pd.read_csv(minimum_path, index_col=0)
    df.index = df.index.map(lambda x: str(x).zfill(3))
    spinglass_states = read_json_files(spinglass_path, beta, eng, bd, CUTOFF_ENERGY, df)

    for name, state_energy_net in tqdm(spinglass_states.items()):
        # print("instance: ", name)
        ground_eng = df[df.index == name]['Ground energy'].values[0]
        instance_df = pd.read_csv(os.path.join(dwave_path, f"{name}.csv"), index_col=0)
        state_energy_tuple = get_state_energy_from_dwave(instance_df, CUTOFF_ENERGY, TOPOLOGY_SIZE, ground_eng)
        energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)
        state_energy_sg = find_max_set(ITERATIONS, state_energy_net, CUTOFF_HAMMING, ground_eng, energy_cutoff)
        state_energy_dwave = find_max_set(ITERATIONS, state_energy_tuple, CUTOFF_HAMMING,  ground_eng, energy_cutoff)
        
        concatenated_se = create_union(state_energy_dwave, state_energy_sg)
        independent_union = find_max_set(ITERATIONS, concatenated_se, CUTOFF_HAMMING, ground_eng, energy_cutoff)

        count_states_in_union = count_states(independent_union, state_energy_sg)
        counts.append(count_states_in_union / len(independent_union.energy))
        result_names.append(name)

    return result_names, counts


if __name__ == '__main__':

    result_names, counts_dw, counts_tn = compute_dwave_spinglass_droplets(dwave_directory, json_directory, minimum_path, BETA, ENG, BD)
    # Sort the results based on counts
    sorted_results_dw = sorted(zip(result_names, counts_dw), key=lambda x: x[0])
    sorted_names_dw, sorted_values_dw = zip(*sorted_results_dw)
    sorted_results_tn = sorted(zip(result_names, counts_tn), key=lambda x: x[0])
    sorted_names_tn, sorted_values_tn = zip(*sorted_results_tn)
    # Plot the graph
    fig, ax = plt.subplots(figsize=(10, 5))

    # plt.ylim((0, 1))
    ax.plot(sorted_names_dw, sorted_values_dw, label = "DW", color = 'red', marker = "o")
    ax.plot(sorted_names_tn, sorted_values_tn, label = "TN", color = 'blue', marker = "o")
    ax.legend()

    plt.xlabel("Instance index")
    plt.ylabel("Droplets")
    plt.title(f"{INSTANCE_SYMBOL}, {INSTANCE_TYPE}, beta={BETA}, eng={ENG}, bond={BD}, approx_ratio={APPROX_RATIO}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    
    # result_names, counts = count_tn_droplets(dwave_directory, json_directory, minimum_path, BETA, ENG, BD)
    # # Sort the results based on counts
    # sorted_results = sorted(zip(result_names, counts), key=lambda x: x[0])
    # sorted_names, sorted_values = zip(*sorted_results)

    # # Plot the graph
    # fig, ax = plt.subplots(figsize=(10, 5))

    # # plt.ylim((0, 1))
    # ax.plot(sorted_names, sorted_values, color = 'red', marker = "o")
    # ax.legend()

    # plt.xlabel("Instance index")
    # plt.ylabel("Droplets")
    # plt.title(f"{INSTANCE_SYMBOL}, {INSTANCE_TYPE}, beta={BETA}, eng={ENG}, bond={BD}, approx_ratio={APPROX_RATIO}")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()
