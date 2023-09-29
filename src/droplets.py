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
CUTOFF_ENERGY = 2.01
CUTOFF_HAMMING = 50
ITERATIONS = 100

# Instance characteristic
TOPOLOGY = "pegasus"
INSTANCE_SYMBOL = "P8"
INSTANCE_TYPE = "CBFM-P"
TOPOLOGY_SIZE = 8
ANNEALING_TIME = 2000  # TODO: remove when all dwave data is aggregated
NUM_READS = 445  # TODO: remove when all dwave data is aggregated

# Directories
cwd = os.getcwd()
json_directory = os.path.join(cwd, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, f"{INSTANCE_SYMBOL}_droplets_new")
dwave_directory = os.path.join(cwd, "energies", f"{TOPOLOGY}_random", INSTANCE_SYMBOL, INSTANCE_TYPE)

# Type aliases
vector = Union[np.ndarray, list]

# Helper functions


def hamming_dist(v1: vector, v2: vector) -> int:
    return hamming(v1, v2) * len(v1)


def get_state_energy_from_dwave(data: pd.DataFrame, cutoff_energy: float, instance_size: int) -> namedtuple:
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    states_dwave = copy.deepcopy(data)
    states_dwave.drop(["energy", "num_occurrences"], axis=1, inplace=True)
    energies_dwave = data["energy"].to_numpy()

    renum = {k: advantage_6_1_to_spinglass_int(int(k), instance_size) - 1 for k in states_dwave.columns}
    states_dwave.rename(columns=renum, inplace=True)

    states_dwave_reordered = states_dwave[ sorted(list(states_dwave.columns))]
    states_dwave_reordered = states_dwave_reordered.to_numpy()

    filtered_states, filtered_energies = filter_states_by_energy(states_dwave_reordered, energies_dwave, cutoff_energy)
    filtered_state_energy_tuple = StateEnergy(filtered_states, filtered_energies)

    return filtered_state_energy_tuple


def filter_states_by_energy(states: np.ndarray, energies: np.ndarray, energy_cutoff: float) -> (np.ndarray, np.ndarray):
    """
    states: Expected to be square matrix, with states in rows
    """
    gs_energy = energies.min()
    mask = energies <= gs_energy + energy_cutoff
    filtered_states = states[mask]
    filtered_energies = energies[mask]
    return filtered_states, filtered_energies


def find_droplets_hamming(state_energy_tuple: namedtuple, hamming_cutoff: int, energy_cutoff: float, permutation: Optional[list] = None):
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
            eng_list = []
            for (i, drop) in enumerate(accepted_states):
                h = hamming_dist(state, drop)
                h_list.append(h)
                eng_list.append(abs(perm_energies[idx] - accepted_energies[i]))
            if all(h >= hamming_cutoff for h in h_list) and all(eng <= energy_cutoff for eng in eng_list):
                accepted_states.append(state)
                accepted_energies.append(perm_energies[idx])
    return AcceptedStateEnergy(np.array(accepted_states), np.array(accepted_energies))


def find_max_set(iterations: int, state_energy_tuple: namedtuple, hamming_cutoff: int, energy_cutoff: float):
    set_size = 0
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    permutation = list(range(len(state_energy_tuple.state)))

    for i in range(iterations):
        random.shuffle(permutation)
        accepted_state_energy_tuple = find_droplets_hamming(state_energy_tuple, hamming_cutoff, energy_cutoff, permutation)
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


def read_json_files(directory) -> dict:
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


def compute_dwave_spinglass_droplets(dwave_path, spinglass_path):
    result_names = []  # To store instance names
    counts = []  # To store count_states_in_union
    spinglass_states = read_json_files(spinglass_path)

    for name, state_energy_net in tqdm(spinglass_states.items()):
        # print("instance: ", name)
        instance_df = pd.read_csv(os.path.join(dwave_path, f"{name}_{ANNEALING_TIME}_{NUM_READS}.csv"),
                                  index_col=0)
        state_energy_tuple = get_state_energy_from_dwave(instance_df, CUTOFF_ENERGY, TOPOLOGY_SIZE)
        state_energy_spin_glass = find_max_set(ITERATIONS, state_energy_net, CUTOFF_HAMMING, CUTOFF_ENERGY)
        state_energy_dwave = find_max_set(ITERATIONS, state_energy_tuple, CUTOFF_HAMMING, CUTOFF_ENERGY)

        concatenated_se = create_union(state_energy_dwave, state_energy_spin_glass)
        independent_union = find_max_set(ITERATIONS, concatenated_se, CUTOFF_HAMMING, CUTOFF_ENERGY)

        count_states_in_union = count_states(independent_union, state_energy_spin_glass)
        counts.append(count_states_in_union / len(independent_union.energy))
        result_names.append(name)

    return result_names, counts


if __name__ == '__main__':

    result_names, counts = compute_dwave_spinglass_droplets(dwave_directory, json_directory)

    # Sort the results based on counts
    sorted_results = sorted(zip(result_names, counts), key=lambda x: x[0])
    sorted_names, sorted_values = zip(*sorted_results)

    # Plot the graph
    plt.figure(figsize=(10, 5))
    # plt.ylim((0, 1))
    plt.bar(sorted_names, sorted_values)
    plt.xlabel("Instance index")
    plt.ylabel("Fraction of TN droplets")
    plt.title(f"{INSTANCE_SYMBOL}, {INSTANCE_TYPE}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
