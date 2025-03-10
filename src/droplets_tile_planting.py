from matplotlib.lines import Line2D

import copy
import os
import json
import random
import time
import ast
import sys

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
from itertools import zip_longest 
from droplets import (filter_states_by_energy, get_state_energy_from_dwave, 
    read_json_files, read_h5_files, xor, hamming_dist, create_spin_glass_peps_graph, connected_hamming_dist,
    filter_states_by_energy, find_droplets_hamming_connected, find_droplets_hamming, find_max_set_connected,
    find_max_set, array_from_dict)

# Instance characteristics
TOPOLOGY = "P16"
TOPOLOGY_SIZE = "16"
L = "40"
BETA = 0.5
HAMMING = 766 #147
APPROX_RATIO = 0.01
ITERATIONS=10
BD = 8
MAX_STATES = 1024 #1024
INST = int(sys.argv[sys.argv.index("--inst") + 1])
# INST = "tile_planting_2D_L_20_p1_0.0_p2_1.0_p3_0.0_inst_1"
script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
root = os.path.dirname(script_dir)
json_directory1 = os.path.join(root, "embedded_tile_planting_simplified", "tn_results", f"{TOPOLOGY}_adv6.4_results", "new", f"{TOPOLOGY}_L{L}_droplets")
dwave_directory = os.path.join(root, "embedded_tile_planting_simplified","embedded_tile_planting_spectra","dwave", f"{TOPOLOGY}_adv6.4_results", f"L_{L}_p2_1")
sbm_directory = os.path.join(root, "embedded_tile_planting_simplified","embedded_tile_planting_spectra","sbm", f"{TOPOLOGY}_adv6.4_results", f"L_{L}_p2_1")
minimum_path = os.path.join(root, "embedded_tile_planting_simplified", "best", f"{TOPOLOGY}_adv6.4_results", f"best_{TOPOLOGY}_2D_L_{L}.csv")
instance_path = os.path.join(root, "embedded_tile_planting_simplified","instances", f"{TOPOLOGY}_adv6.4", "nonzero")
output_directory_union = os.path.join(root, "embedded_tile_planting_simplified", "droplets", f"{TOPOLOGY}_adv6.4", f"results_union_ar01_dr025", f"{INST}")
output_directory_solver = os.path.join(root, "embedded_tile_planting_simplified", "droplets", f"{TOPOLOGY}_adv6.4", f"results_solver_ar01_dr025", f"{INST}")

vector = Union[np.ndarray, list]

# def compare_states(state, states, graph):
#     for (i, st) in enumerate(states):
#         h = connected_hamming_dist(state, st, graph)
#         if h <= HAMMING:
#             return True
#     return False

# def compare_states(state, states, graph):
#     # print("compare states")
#     for (i, st) in enumerate(states):
#         h = connected_hamming_dist(state, st, graph)
#         # print("hamming ", h)
#         if h <= HAMMING:
#             return st, i+1
#     return False

def compare_states(state, states, graph):
    min_distance = float('inf') 
    min_state = None 
    min_index = None 
    
    for i, st in enumerate(states):
        # h = connected_hamming_dist(state, st, graph)
        h = hamming_dist(state, st)
        if h <= HAMMING and h < min_distance:  
            min_state = st  
            min_distance = h  
            min_index = i 

    if min_state is not None:  
        return min_state, min_index + 1  
    else:
        return False, None 


def compute_union(results_folder, output_directory):
    names = []
    counts = []
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Folder '{output_directory}' created successfully.")
    
    for i, filename in enumerate(sorted(os.listdir(results_folder))):
        if filename.endswith(".csv"):
            name = filename.split("_P")[0]
            file_path = os.path.join(results_folder, filename)
            df = pd.read_csv(file_path, index_col=0, delimiter=',', quotechar='"')
            count = df['Count'].max()
        names.append(name)
        counts.append(count)
    
    data = {'Name': names, 'Count': counts}
    df_output = pd.DataFrame(data)
    output_csv_path = os.path.join(output_directory, 'union.csv')
    df_output.to_csv(output_csv_path, index=False)
    return names, counts

def read_all_json_files(directory, beta, df_min) -> dict:
    instance_data = {}
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
        if os.path.isfile(file) and file.endswith(".csv"):
            with open(file, encoding='utf-8') as f:
                json_data = json.load(f)
                if json_data['columns'][json_data['colindex']['lookup']['β']-1][0] == beta:
                    instance_name = json_data['columns'][json_data['colindex']['lookup']['instance']-1][0].split('_P')[0]
                    # instance_name = instance_name.split('_P')[0]

                    if instance_name == "BP":
                        continue
                    else:
                        energy_data = json_data['columns'][json_data['colindex']['lookup']['drop_eng']-1][0]
                        state_data = json_data['columns'][json_data['colindex']['lookup']['ig_states']-1][0]
                        state_data_np = array_from_dict(state_data)
                        energy_data_np = np.array(energy_data)
                        ground_eng = df_min[df_min.index == instance_name]['Energy'].values[0]
                        cutoff_energy = APPROX_RATIO * 2 * np.abs(ground_eng)
                        filtered_states, filtered_energies = filter_states_by_energy(state_data_np, energy_data_np, cutoff_energy, ground_eng)

                        if instance_name in instance_data:
                            existing_state_energy = instance_data[instance_name]
                            new_state_data = np.concatenate((existing_state_energy.state, filtered_states))
                            new_energy_data = np.concatenate((existing_state_energy.energy, filtered_energies))
                            unique_rows, unique_indices = np.unique(new_state_data, axis=0, return_index=True)
                            filtered_energy_data = new_energy_data[unique_indices]
                            instance_data[instance_name] = StateEnergy(unique_rows, filtered_energy_data)                
                        else:
                            instance_data[instance_name] = StateEnergy(filtered_states, filtered_energies)
                else:
                    pass
    return instance_data

def get_state_energy(data: pd.DataFrame, cutoff_energy: float,
                    ground_eng: float) -> namedtuple:
    # ground_eng = df_min[df_min.index == instance_name]['Energy'].values[0]
    # cutoff_energy = APPROX_RATIO * 2 * np.abs(ground_eng)
                        
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    data_filtered = data
    states_dwave = copy.deepcopy(data_filtered)
    states_dwave.drop(["energy"], axis=1, inplace=True)
    
    energies_dwave = data_filtered["energy"].to_numpy()

    # renum = {k: advantage_6_1_to_spinglass_int(int(k), instance_size) for k in states_dwave.columns}
    # states_dwave.rename(columns=renum, inplace=True)

    states_dwave_reordered = states_dwave[sorted(list(states_dwave.columns))]
    states_dwave_reordered = states_dwave_reordered.to_numpy()

    filtered_states, filtered_energies = filter_states_by_energy(states_dwave_reordered, energies_dwave,
                                                                 cutoff_energy, ground_eng)
    filtered_state_energy_tuple = StateEnergy(filtered_states, filtered_energies)

    return filtered_state_energy_tuple


def get_state_energy_from_dwave(data: pd.DataFrame, cutoff_energy: float,
                                instance_size: int, ground_eng: float) -> namedtuple:
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])

    # Sprawdzamy, czy "energy" jest indeksem
    if "energy" not in data.columns:
        data = data.reset_index()

    # Kopia DataFrame
    states_dwave = copy.deepcopy(data)

    # Pobranie energii jako numpy array
    energies_dwave = data["energy"].to_numpy()

    # Konwersja states (ze stringów do macierzy numpy)
    states_dwave["state"] = states_dwave["state"].apply(lambda x: np.array(x.split(";"), dtype=float))
    states_dwave_reordered = np.vstack(states_dwave["state"].to_numpy())  # Konwersja do macierzy 2D

    # Filtracja stanów
    filtered_states, filtered_energies = filter_states_by_energy(states_dwave_reordered, energies_dwave,
                                                                 cutoff_energy, ground_eng)
    
    return StateEnergy(filtered_states, filtered_energies)

def create_independent_set_of_union_states(json_directory1, dwave_directory, h5_directory, best_found_path, instance_path, history_directory, metric: Optional[str], **kwargs):

    min_df = pd.read_csv(best_found_path, index_col=0, delimiter=',', quotechar='"')
    min_df.index = min_df.index.map(lambda x: str(x).zfill(3))
    if not os.path.exists(history_directory):
        os.makedirs(history_directory)
        print(f"Folder '{history_directory}' created successfully.")
    
    beta = kwargs["beta"]
    spinglass_states1 = read_all_json_files(json_directory1, beta, min_df)

    inst = kwargs["inst"]
    # sbm_states = read_h5_files(h5_directory, min_df, APPROX_RATIO, inst)
    sbm_states = {}
    for filename in tqdm(os.listdir(sbm_directory)):
        StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])

        file = os.path.join(sbm_directory, filename)
        name = "_".join(filename.split('_')[2:]).replace('.csv', '')
        name_range = f"inst_{inst}"
        if name_range not in name:
            continue
        if os.path.isfile(file):
            instance_df = pd.read_csv(file, index_col=0, delimiter=',',quotechar='"')
            ground_eng = min_df[min_df.index == name]['Energy'].values[0]
            instance_parameters = f"_beta{beta}_{metric}_best{ground_eng}"
            history_file_name = os.path.join(history_directory, f"{name}.csv")
            energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)
            state_energy_sbm = get_state_energy_from_dwave(instance_df, energy_cutoff, TOPOLOGY_SIZE, ground_eng)
            sbm_states[name] = state_energy_sbm
            
    for filename in tqdm(os.listdir(dwave_directory)):
        StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])

        file = os.path.join(dwave_directory, filename)
        name = "_".join(filename.split('_')[2:]).replace('.csv', '')
        name_range = f"inst_{inst}"
        if name_range not in name:
            continue
        if os.path.isfile(file):
            instance_df = pd.read_csv(file, index_col=0, delimiter=',',quotechar='"')
            ground_eng = min_df[min_df.index == name]['Energy'].values[0]
            instance_parameters = f"_beta{beta}_{metric}_best{ground_eng}"
            history_file_name = os.path.join(history_directory, f"{name}.csv")
            energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)
            state_energy_dw = get_state_energy_from_dwave(instance_df, energy_cutoff, TOPOLOGY_SIZE, ground_eng)
            state_energy_tn1 = spinglass_states1[name]
            state_energy_sbm = sbm_states[name]

            union_data = {}
            for nt in [state_energy_dw, state_energy_tn1, state_energy_sbm]:
                for state, energy in zip_longest(nt.state, nt.energy, fillvalue=None):
                    state = tuple(state) if isinstance(state, np.ndarray) else state
                    if state not in union_data:
                        union_data[state] = energy

            unique_states = list(union_data.keys())
            unique_energies = [union_data[state] for state in unique_states]
            unique_namedtuple = StateEnergy(unique_states, unique_energies)
            
            if metric == "connected":
                file_inst = os.path.join(instance_path, name + ".txt")
                graph = create_spin_glass_peps_graph(file_inst)
                state_energy = find_max_set_connected(ITERATIONS, unique_namedtuple, HAMMING, ground_eng, energy_cutoff, graph, history_file_name)     
            else:
                state_energy = find_max_set(ITERATIONS, unique_namedtuple, HAMMING, ground_eng, energy_cutoff, history_file_name)
    
    return #state_energy


# def create_independent_set_of_union_states(json_directory1, dwave_directory, sbm_directory, best_found_path, instance_path, history_directory, metric: Optional[str], **kwargs):

#     min_df = pd.read_csv(best_found_path, index_col=0, delimiter=',', quotechar='"')
#     min_df.index = min_df.index.map(lambda x: str(x).zfill(3))
    
#     if not os.path.exists(history_directory):
#         os.makedirs(history_directory)
#         print(f"Folder '{history_directory}' created successfully.")
    
#     beta = kwargs["beta"]
#     spinglass_states1 = read_all_json_files(json_directory1, beta, min_df)
#     dw_states = get_state_energy(dwave_directory, beta, min_df)
#     sbm_states1 = get_state_energy(sbm_directory, beta, min_df)

#     # inst = kwargs["inst"]
    
#     for name in tqdm(spinglass_states1.keys()):
#         StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
        
#         ground_eng = min_df.loc[name, 'Energy']
#         instance_parameters = f"_beta{beta}_{metric}_best{ground_eng}"
#         history_file_name = os.path.join(history_directory, f"{name}{instance_parameters}.csv")
#         energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)

#         state_energy_tn1 = spinglass_states1[name]

#         union_data = {}
#         for nt in [state_energy_tn1]:
#             for state, energy in zip_longest(nt.state, nt.energy, fillvalue=None):
#                 state = tuple(state) if isinstance(state, np.ndarray) else state
#                 if state not in union_data:
#                     union_data[state] = energy

#         unique_states = list(union_data.keys())
#         unique_energies = [union_data[state] for state in unique_states]
#         unique_namedtuple = StateEnergy(unique_states, unique_energies)
        
#         if metric == "connected":
#             file_inst = os.path.join(instance_path, name + "_sg.txt")
#             graph = create_spin_glass_peps_graph(file_inst)
#             state_energy = find_max_set_connected(ITERATIONS, unique_namedtuple, HAMMING, ground_eng, energy_cutoff, graph, history_file_name)     
#         else:
#             state_energy = find_max_set(ITERATIONS, unique_namedtuple, HAMMING, ground_eng, energy_cutoff, history_file_name)
    
#     return #state_energy


if __name__ == '__main__':
    result = create_independent_set_of_union_states(json_directory1, dwave_directory, sbm_directory, minimum_path, instance_path, output_directory_union, 
                                                    "nonconnected", beta=BETA, inst=INST)
    compute_union(output_directory_union, output_directory_solver)
    # name, count = find_droplets_in_union(output_directory_union, json_directory1, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SpinGlass",
    #                                 beta=BETA, eng=ENG, bd=BD, ms=MAX_STATES, inst=INST, text="tr16")
