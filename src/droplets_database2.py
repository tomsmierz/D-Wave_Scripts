import copy
import os
import json
import random
import time

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dwave_networkx as dnx
import networkx as nx

from collections import namedtuple
from renumeration import advantage_6_1_to_spinglass, advantage_6_1_to_spinglass_int
from scipy.spatial.distance import hamming
from typing import Optional, Union
from tqdm import tqdm

# TODO: use @dataclass to store droplets?
# TODO: move helper functions to another file?

# Constants
CUTOFF_ENERGY = 6 #10 CBFMP, 6 RAU
CUTOFF_HAMMING = 27
ITERATIONS = 25
APPROX_RATIO = 1e-2
BETA = 0.5
ENG = 10 #10 CBFMP, 6 RAU
BD = 4
BD1 = 8 
BD2 = 12
MAX_STATES = 64
MAX_STATES1 = 256
MAX_STATES2 = 1024


# CUTOFF_ENERGY = 60
# CUTOFF_HAMMING = 100
# ITERATIONS = 10
# APPROX_RATIO = 1e-3
# BETA = 0.5
# ENG = 60
# BD = 4

# Instance characteristic
TOPOLOGY = "pegasus"
INSTANCE_SYMBOL = "P4"
INSTANCE_TYPE = "CBFM-P"
TOPOLOGY_SIZE = 4

# Directories
script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
root = os.path.dirname(script_dir)
json_directory = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "final_bench")

# json_directory = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, f"{INSTANCE_SYMBOL}_droplets_i1-2")
dwave_directory = os.path.join(root, "energies", f"{TOPOLOGY}_random_aggregated", INSTANCE_SYMBOL, INSTANCE_TYPE)
h5_directory = os.path.join(root, "energies", "sbm", f"{TOPOLOGY}_random", INSTANCE_SYMBOL, INSTANCE_TYPE,
                            "SpinGlass", "tmp")
minimum_path = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "minimum.csv")
instance_path = os.path.join(root, "instances", f"{TOPOLOGY}_random", INSTANCE_SYMBOL, INSTANCE_TYPE)
output_directory = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "results")
# Type aliases
vector = Union[np.ndarray, list]


# Helper functions

def xor(v1: vector, v2: vector) -> vector:
    assert len(v1) == len(v2)
    return [0 if v1[i] == v2[i] else 1 for i in range(len(v1))]


def hamming_dist(v1: vector, v2: vector) -> int:
    return hamming(v1, v2) * len(v1)


def create_spin_glass_peps_graph(file: str) -> nx.Graph:
    df = pd.read_csv(file, sep=" ", names=["v", "w", "J"], comment="#")
    edges = []
    nodes = []
    for row in df.itertuples():
        if row.v != row.w and row.J != 0:
            edges.append((row.v,row.w))
        elif row.v == row.w:
            nodes.append(row.v)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def connected_hamming_dist(state1: vector, state2: vector, graph) -> int:
    xor_state = xor(state1, state2)
    nodes = []
    for node in graph.nodes:
        if xor_state[node-1]:
            nodes.append(node)
    if nodes == []:
        largest_cc = []
    else:
        subgraph = nx.subgraph(graph, nodes)
        largest_cc = max(nx.connected_components(subgraph), key=len)
    return len(largest_cc)


def get_state_energy_from_dwave(data: pd.DataFrame, cutoff_energy: float,
                                instance_size: int, ground_eng: float) -> namedtuple:
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    states_dwave = copy.deepcopy(data)
    states_dwave.drop(["energy", "num_occurrences", "annealing_time", "num_reads", "pause_time", "reverse"],
                      axis=1, inplace=True)
    energies_dwave = data["energy"].to_numpy()

    renum = {k: advantage_6_1_to_spinglass_int(int(k), instance_size) - 1 for k in states_dwave.columns}
    states_dwave.rename(columns=renum, inplace=True)

    states_dwave_reordered = states_dwave[sorted(list(states_dwave.columns))]
    states_dwave_reordered = states_dwave_reordered.to_numpy()

    filtered_states, filtered_energies = filter_states_by_energy(states_dwave_reordered, energies_dwave,
                                                                 cutoff_energy, ground_eng)
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


def find_droplets_hamming_connected(graph: nx.Graph, state_energy_tuple: namedtuple, hamming_cutoff: int,
                                    ground_eng: float, energy_cutoff: float, permutation: Optional[list] = None,
                                    checkpoint: Optional[namedtuple] = None):
    accepted_states = [] if not checkpoint else [list(checkpoint.state[j]) for j in
                                                 range(len(checkpoint.energy))]
    accepted_energies = [] if not checkpoint else list(checkpoint.energy)
    AcceptedStateEnergy = namedtuple('AcceptedStateEnergy', ['state', 'energy'])

    if permutation is not None:
        perm_states = [state_energy_tuple.state[i] for i in permutation]
        perm_energies = [state_energy_tuple.energy[i] for i in permutation]
    else:
        perm_states = state_energy_tuple.state
        perm_energies = state_energy_tuple.energy
    for idx, state in tqdm(enumerate(perm_states)):
        if not accepted_states:
            accepted_states.append(state)
            accepted_energies.append(perm_energies[idx])
        elif any(np.array_equal(state, accepted_state) for accepted_state in accepted_states):
            pass
        else:
            h_list = []
            for drop in accepted_states:
                h = connected_hamming_dist(state, drop, graph)
                h_list.append(h)
                eng = (abs(perm_energies[idx] - ground_eng))
            if eng <= energy_cutoff and all(h >= hamming_cutoff for h in h_list):
                accepted_states.append(state)
                accepted_energies.append(perm_energies[idx])
    return AcceptedStateEnergy(np.array(accepted_states), np.array(accepted_energies))

def find_droplets_hamming(state_energy_tuple: namedtuple, hamming_cutoff: int, ground_eng: float,
                          energy_cutoff: float, permutation: Optional[list] = None):
    accepted_states = [] 
    accepted_energies = [] 
    AcceptedStateEnergy = namedtuple('AcceptedStateEnergy', ['state', 'energy'])

    if permutation is not None:
        perm_states = [state_energy_tuple.state[i] for i in permutation]
        perm_energies = [state_energy_tuple.energy[i] for i in permutation]
    else:
        perm_states = state_energy_tuple.state
        perm_energies = state_energy_tuple.energy
    for idx, state in tqdm(enumerate(perm_states)):
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
                eng = (abs(perm_energies[idx] - ground_eng))
            if eng <= energy_cutoff and all(h >= hamming_cutoff for h in h_list):
                accepted_states.append(state)
                accepted_energies.append(perm_energies[idx])
    return AcceptedStateEnergy(np.array(accepted_states), np.array(accepted_energies))



# def find_max_set_connected(iterations: int, state_energy_tuple: namedtuple, hamming_cutoff: int,
#                  ground_eng: float, energy_cutoff: float, graph: nx.Graph):
#     set_size = 0
#     StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
#     permutation = list(range(len(state_energy_tuple.state)))

#     for i in range(iterations):
#         random.shuffle(permutation)
#         accepted_state_energy_tuple = find_droplets_hamming_connected(graph, state_energy_tuple, hamming_cutoff,
#                                                             ground_eng, energy_cutoff, permutation)
#         if len(accepted_state_energy_tuple.state) > set_size:
#             new_state_energy = StateEnergy(accepted_state_energy_tuple.state, accepted_state_energy_tuple.energy)
#             set_size = len(accepted_state_energy_tuple.state)
#     # TODO: check edge case
#     return new_state_energy

def find_max_set_connected(iterations: int, state_energy_tuple: namedtuple, hamming_cutoff: int,
                 ground_eng: float, energy_cutoff: float, graph: nx.Graph, file_path: str):
    set_size = 0
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    permutation = list(range(len(state_energy_tuple.energy)))

    if os.path.exists(file_path):
        result_df = pd.read_csv(file_path)
        max_iterations_stored = result_df['Iterations'].max()
        
        if max_iterations_stored >= iterations:
            max_iterations_row = result_df[result_df['Iterations'] == iterations]
            max_state = np.array(eval(max_iterations_row['State'].values[0]))
            max_energy = np.array(eval(max_iterations_row['Energy'].values[0]))
            new_state_energy_tuple = StateEnergy(max_state, max_energy)
        else:
            max_iterations_row = result_df[result_df['Iterations'] == max_iterations_stored]
            max_state = np.array(eval(max_iterations_row['State'].values[0]))
            max_energy = np.array(eval(max_iterations_row['Energy'].values[0]))
            new_state_energy_tuple = StateEnergy(max_state, max_energy)
            for i in range(max_iterations_stored+1, iterations+1):
                print(i)
                random.shuffle(permutation)
                accepted_state_energy_tuple = find_droplets_hamming_connected(graph, state_energy_tuple, hamming_cutoff,
                                                                ground_eng, energy_cutoff, permutation, new_state_energy_tuple)
                set_size = max_iterations_row['Count'].values[0]
                if len(accepted_state_energy_tuple.state) > set_size:
                    new_state_energy_tuple = StateEnergy(accepted_state_energy_tuple.state, accepted_state_energy_tuple.energy)
                new_data = {'Iterations': i, 'Energy': [list(new_state_energy_tuple.energy)],
                            'State': [[list(new_state_energy_tuple.state[j])
                                   for j in range(len(new_state_energy_tuple.energy))]],
                            'Count': len(new_state_energy_tuple.energy)}
                result_df = pd.concat([result_df, pd.DataFrame(new_data)], ignore_index=True)
                result_df.to_csv(file_path, index=False)
    else:
        result_df = pd.DataFrame(columns=['Iterations', 'Count', 'Energy', 'State'])
        for i in range(iterations):
            random.shuffle(permutation)
            accepted_state_energy_tuple = find_droplets_hamming_connected(graph, state_energy_tuple, hamming_cutoff,
                                                                ground_eng, energy_cutoff, permutation)
            if len(accepted_state_energy_tuple.state) > set_size:
                new_state_energy_tuple = StateEnergy(accepted_state_energy_tuple.state, accepted_state_energy_tuple.energy)
                set_size = len(accepted_state_energy_tuple.state)
            new_data = {'Iterations': i+1, 'Energy': [list(new_state_energy_tuple.energy)],
                        'State': [[list(new_state_energy_tuple.state[j])
                                   for j in range(len(new_state_energy_tuple.energy))]],
                        'Count': len(new_state_energy_tuple.energy)}
            result_df = pd.concat([result_df, pd.DataFrame(new_data)], ignore_index=True)
            result_df.to_csv(file_path, index=False)
    
    # TODO: check edge case
    return new_state_energy_tuple


def find_max_set(iterations: int, state_energy_tuple: namedtuple, hamming_cutoff: int,
                 ground_eng: float, energy_cutoff: float):
    set_size = 0
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    permutation = list(range(len(state_energy_tuple.state)))

    for i in range(iterations):
        random.shuffle(permutation)
        accepted_state_energy_tuple = find_droplets_hamming(state_energy_tuple, hamming_cutoff,
                                                            ground_eng, energy_cutoff, permutation)
        if len(accepted_state_energy_tuple.state) > set_size:
            new_state_energy = StateEnergy(accepted_state_energy_tuple.state, accepted_state_energy_tuple.energy)
            set_size = len(accepted_state_energy_tuple.state)
    # TODO: check edge case
    return new_state_energy


def create_union(states1: namedtuple, states2: namedtuple):
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    new_states = np.vstack((states1.state, states2.state))
    new_engs = np.hstack((states1.energy, states2.energy))
    _, idx = np.unique(new_states, axis=0, return_index=True)

    return StateEnergy(new_states[idx], new_engs[idx])


def count_states_connected(independent_states_union, unique_states_tn, graph: nx.Graph):
    count_states_in_union = 0
    for state in independent_states_union.state:
        hamming_distances = [connected_hamming_dist(state_tn, state, graph) for state_tn in unique_states_tn.state]
        if any(distance < CUTOFF_HAMMING for distance in hamming_distances):
            count_states_in_union += 1
    return count_states_in_union

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


def read_h5_files(directory, min_path):
    instance_data = {}
    df_min = pd.read_csv(min_path, index_col=0)
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
        if os.path.isfile(file):
            file_extension = os.path.splitext(filename)[-1].lower()
            if file_extension == ".h5":
                f = h5py.File(file, "r")
                instance_name = filename.split("_")[0]
                energies = f['Spectrum']['energies']
                states = f['Spectrum']["states"]
                #ground_eng = df_min[df_min.index == instance_name]['Ground energy'].values[0]
                ground_eng = energies[0]
                energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)
                filtered_states, filtered_energies = filter_states_by_energy(states, energies,
                                                                             energy_cutoff, ground_eng)
                instance_data[instance_name] = StateEnergy(filtered_states, filtered_energies)
    return instance_data


def read_json_files(directory, beta, eng, bd, ms, cutoff_energy, df_min) -> dict:
    instance_data = {}
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
        if os.path.isfile(file):
            with open(file, encoding='utf-8') as f:
                json_data = json.load(f)
                if json_data['columns'][json_data['colindex']['lookup']['β']-1][0] == beta and \
                        json_data['columns'][json_data['colindex']['lookup']['eng']-1][0] == eng and \
                        json_data['columns'][json_data['colindex']['lookup']['bond_dim']-1][0] == bd and \
                        json_data['columns'][json_data['colindex']['lookup']['max_states']-1][0] == ms:
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


def compute_droplets(path: str, best_found_path: str, instance_path: str, output_directory: str, solver: Optional[str], metric: Optional[str], **kwargs):
    if solver not in ["Dwave", "SpinGlass", "SBM", "PT", None]:
        raise ValueError("Solver should be \"Dwave\", \"SpinGlass\", \"SBM\", \"PT\" or None")

    min_df = pd.read_csv(best_found_path, index_col=0)
    min_df.index = min_df.index.map(lambda x: str(x).zfill(3))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Folder '{output_directory}' created successfully.")

    if solver == "Dwave":
        result_names, counts, se = compute_droplets_dwave(path, min_df, instance_path, metric, output_directory)

    elif solver == "SpinGlass":
        if "beta" not in kwargs or "eng" not in kwargs or "bd" not in kwargs or "ms" not in kwargs:
            raise ValueError("To compute droplets for SpinGlassPEPS parameters beta, eng (cutoff energy in "
                             "SpinGlassPEPS), bd (bond dimension) and ms (max states) are needed")
        beta = kwargs["beta"]
        eng = kwargs["eng"]
        bd = kwargs["bd"]
        ms = kwargs["ms"]
        result_names, counts, se = compute_droplets_spiglass(path, min_df, instance_path, metric, beta, eng, bd, ms, output_directory)

    elif solver == "SBM":
        result_names, counts, se = compute_droplets_sbm(path, best_found_path, min_df, instance_path, metric, output_directory)

    elif solver == "PT":
        raise NotImplementedError()

    else:
        result_names, counts, se = [], [], []

    return result_names, counts, se


def compute_droplets_sbm(path: str, best_found_path: str, min_df: pd.DataFrame, instance_path: str, metric: Optional[str], output_directory):
    result_names = []
    counts = []
    state_energy = {}
    sbm_states = read_h5_files(path, best_found_path)
    for name, state_energy_h in tqdm(sbm_states.items()):
        if name not in ["001", "002", "003", "004", "005"]:
            break
        ground_eng = min_df[min_df.index == name]['Ground energy'].values[0]
        energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)
        if metric == "connected":
            file_inst = os.path.join(instance_path, name + "_sg.txt")
            graph = create_spin_glass_peps_graph(file_inst)
            state_energy_sbm = find_max_set_connected(ITERATIONS, state_energy_h, CUTOFF_HAMMING, ground_eng, energy_cutoff, graph)
        else: 
            state_energy_sbm = find_max_set(ITERATIONS, state_energy_h, CUTOFF_HAMMING, ground_eng, energy_cutoff)

        state_energy[name] = state_energy_sbm
        count = len(state_energy_sbm.energy)

        counts.append(count)
        result_names.append(name)

    return result_names, counts, state_energy


def compute_droplets_spiglass(path: str, min_df: pd.DataFrame, instance_path: str, metric: Optional[str], beta: float, eng: float, bd: int, ms: int, history_directory):
    result_names = []
    counts = []
    state_energy = {}

    spinglass_states = read_json_files(path, beta, eng, bd, ms, CUTOFF_ENERGY, min_df)

    for name, state_energy_tn in tqdm(spinglass_states.items()):
        if name not in ["001", "002", "003", "004", "005"]:
            break
        # if solver == "SpinGlass":
        instance_parameters = f"_beta{beta}_eng{eng}_bd{bd}_ms{ms}_{metric}_SpinGlass"
        # else:
        #     instance_parameters = f"_{metric}_{solver}"
        history_file_name = os.path.join(history_directory, f"{name}{instance_parameters}.csv")
        print(history_file_name)
        ground_eng = min_df[min_df.index == name]['Ground energy'].values[0]
        energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)
        if metric == "connected":
            file_inst = os.path.join(instance_path, name + "_sg.txt")
            graph = create_spin_glass_peps_graph(file_inst)
            state_energy_sg = find_max_set_connected(ITERATIONS, state_energy_tn, CUTOFF_HAMMING, ground_eng, energy_cutoff, graph, history_file_name)
        else:
            state_energy_sg = find_max_set(ITERATIONS, state_energy_tn, CUTOFF_HAMMING, ground_eng, energy_cutoff)
        state_energy[name] = state_energy_sg
        count = len(state_energy_sg.energy)
        counts.append(count)
        result_names.append(name)

    return result_names, counts, state_energy


def compute_droplets_dwave(path: str, min_df: pd.DataFrame, instance_path: str, metric: Optional[str], output_directory):
    result_names = []
    counts = []
    state_energy = {}

    for filename in tqdm(os.listdir(path)):
        file = os.path.join(path, filename)
        name = filename.split(".")[0]
        if name not in ["001", "002", "003", "004", "005"]:
            continue
        if os.path.isfile(file):
            instance_df = pd.read_csv(file, index_col=0)
            ground_eng = min_df[min_df.index == name]['Ground energy'].values[0]
            state_energy_tuple = get_state_energy_from_dwave(instance_df, CUTOFF_ENERGY, TOPOLOGY_SIZE, ground_eng)
            energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)
            if metric == "connected":
                file_inst = os.path.join(instance_path, name + "_sg.txt")
                graph = create_spin_glass_peps_graph(file_inst)
                state_energy_dw = find_max_set_connected(output_directory, ITERATIONS, state_energy_tuple, CUTOFF_HAMMING, ground_eng, energy_cutoff, graph)
            else:
                state_energy_dw = find_max_set(output_directory, ITERATIONS, state_energy_tuple, CUTOFF_HAMMING, ground_eng, energy_cutoff)
            state_energy[name] = state_energy_dw
            count = len(state_energy_dw.energy)
            result_names.append(name)
            counts.append(count)
    return result_names, counts, state_energy


def count_droplets_in_union(best_found_path: str, state_energy_1: dict, state_energy_2: dict, instance_path: str, metric: Optional[str]):
    result_names = []  # To store instance names
    counts = []  # To store count_states_in_union

    min_df = pd.read_csv(best_found_path, index_col=0)
    min_df.index = min_df.index.map(lambda x: str(x).zfill(3))
    for name, se1 in tqdm(state_energy_1.items()):
        se2 = state_energy_2[name]
        ground_eng = min_df[min_df.index == name]['Ground energy'].values[0]
        energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)
        concatenated_se = create_union(se1, se2)
        if metric == "connected":
            file_inst = os.path.join(instance_path, name + "_sg.txt")
            graph = create_spin_glass_peps_graph(file_inst)
            independent_union = find_max_set_connected(ITERATIONS, concatenated_se, CUTOFF_HAMMING, ground_eng, energy_cutoff, graph)
            count_states_in_union = count_states_connected(independent_union, se2, graph)
        else:
            independent_union = find_max_set(ITERATIONS, concatenated_se, CUTOFF_HAMMING, ground_eng, energy_cutoff)
            count_states_in_union = count_states(independent_union, se2)
            
        counts.append(count_states_in_union / len(independent_union.energy))
        result_names.append(name)

    return result_names, counts


def plot_instance_vs_count(output_directory):
    # Collect data from all CSV files in the output directory
    all_data = []
    names = []
    iterations = []
    for filename in os.listdir(output_directory):
        if filename.endswith(".csv"):
            name_parts = filename.split("_")
            name = name_parts[0]
            iteration = name_parts[1][4:]  # Extract iteration number
            instance_df = pd.read_csv(os.path.join(output_directory, filename))
            all_data.append(instance_df)
            names.append(name)
            iterations.append(iteration)

    if all_data:
        plt.figure(figsize=(10, 6))
        unique_iterations = set(iterations)
        
        markers = ['o', 's', 'D', 'v', '^']  # Add more if needed
        colors = ['b', 'g', 'r', 'c', 'm']   # Add more if needed

        marker_color_dict = dict(zip(unique_iterations, zip(markers, colors)))
        
        legend_labels = []

        for iteration in unique_iterations:
            iteration_data = [data for i, data in enumerate(all_data) if iterations[i] == iteration]
            iteration_names = [name for i, name in enumerate(names) if iterations[i] == iteration]

            # Sort data and names based on instance names
            sorted_data = sorted(zip(iteration_names, iteration_data), key=lambda x: x[0])
            sorted_names, sorted_df = zip(*sorted_data)
            marker, color = marker_color_dict[iteration]  # Get marker and color for this iteration
            label = f"Iteration {iteration}"

            for i, instance_df in enumerate(sorted_df):
                plt.plot(sorted_names[i], instance_df['Count'][0], marker, color=color, label=label)
            
            # Create a custom legend label with marker and color
            legend_labels.append(plt.Line2D([0], [0], marker=marker, color=color, label=label))

        plt.xlabel("Instance")
        plt.ylabel("Count")
        plt.title("Instance vs. Count")
        plt.legend(handles=legend_labels, title="Iterations")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, "instance_vs_count.png"))

if __name__ == '__main__':

    # result_names_dw, counts_dw, se_dw = compute_droplets(dwave_directory, minimum_path, instance_path, output_directory, "DWave", "connected")
    result_names_tn, counts_tn, se_tn = compute_droplets(json_directory, minimum_path, instance_path, output_directory, "SpinGlass", "connected",
                                            beta=BETA, eng=ENG, bd=BD, ms=MAX_STATES)
    # result_names_sbm, counts_sbm, se_sbm = compute_droplets(h5_directory, minimum_path, instance_path, output_directory, "SBM", "connected")

    # plot_instance_vs_count(output_directory)