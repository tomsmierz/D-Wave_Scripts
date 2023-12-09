from matplotlib.lines import Line2D

import copy
import os
import json
import random
import time
import ast

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
from droplets_database import (filter_states_by_energy, get_state_energy_from_dwave, 
    read_json_files, read_h5_files, xor, hamming_dist, create_spin_glass_peps_graph, connected_hamming_dist,
    filter_states_by_energy, find_droplets_hamming_connected, find_droplets_hamming, find_max_set_connected,
    find_max_set, array_from_dict)

# Instance characteristics
TOPOLOGY = "pegasus"
INSTANCE_SYMBOL = "P8"
INSTANCE_TYPE = "RCO"
TOPOLOGY_SIZE = 8
BETA = 0.5
HAMMING = 147 #147
APPROX_RATIO = 0.005
ITERATIONS=1
ENG= 40
BD = 8
BD1 = 8 
BD2 = 12
MAX_STATES = 1024
MAX_STATES1 = 256
MAX_STATES2 = 1024

# TOPOLOGY = "pegasus"
# INSTANCE_SYMBOL = "P4"
# INSTANCE_TYPE = "RCO"
# TOPOLOGY_SIZE = 4
# BETA = 0.75
# HAMMING = 27 #147
# APPROX_RATIO = 1e-2
# ITERATIONS=100
# ENG = 7
# BD = 4
# BD1 = 8
# BD2 = 12
# MAX_STATES = 64
# MAX_STATES1 = 256
# MAX_STATES2 = 1024

script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
root = os.path.dirname(script_dir)
json_directory = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "final_bench_truncate2^16")
dwave_directory = os.path.join(root, "energies", f"{TOPOLOGY}_random_aggregated", INSTANCE_SYMBOL, INSTANCE_TYPE, "filtered_files_hd100_i020")
h5_directory = os.path.join(root, "energies", "sbm", f"{TOPOLOGY}_random", INSTANCE_SYMBOL, INSTANCE_TYPE,
                            "SpinGlass", "tmp")
minimum_path = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "minimum_truncated2^16.csv")
instance_path = os.path.join(root, "instances", f"{TOPOLOGY}_random", INSTANCE_SYMBOL, INSTANCE_TYPE)
output_directory_union = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "results_beta05_union_ar005_truncated2^16")
output_directory_solver = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "results_beta05_solver_ar005_truncated2^16")

vector = Union[np.ndarray, list]



def find_droplets_in_union(results_folder, json_directory, dwave_directory, h5_directory, output_directory, best_found_path, instance_path, solver, **kwargs):
    if solver not in ["DWave", "SpinGlass", "SBM", "PT", None]:
        raise ValueError("Solver should be \"DWave\", \"SpinGlass\", \"SBM\", \"PT\" or None")

    min_df = pd.read_csv(best_found_path, index_col=0)
    min_df.index = min_df.index.map(lambda x: str(x).zfill(3))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Folder '{output_directory}' created successfully.")

    if solver == "DWave":
        result_names, counts = compute_droplets_in_union_dwave(results_folder, dwave_directory, min_df, instance_path, output_directory)

    elif solver == "SpinGlass":
        if "beta" not in kwargs or "eng" not in kwargs or "bd" not in kwargs or "ms" not in kwargs:
            raise ValueError("To compute droplets for SpinGlassPEPS parameters beta, eng (cutoff energy in "
                             "SpinGlassPEPS), bd (bond dimension) and ms (max states) are needed")
        beta = kwargs["beta"]
        eng = kwargs["eng"]
        bd = kwargs["bd"]
        ms = kwargs["ms"]
        result_names, counts = compute_droplets_in_union_spinglass(results_folder, json_directory, beta, eng, bd, ms, output_directory, instance_path, min_df)

    elif solver == "SBM":
        result_names, counts = compute_droplets_in_union_sbm(results_folder, h5_directory, output_directory, instance_path, min_df)

    elif solver == "PT":
        raise NotImplementedError()

    else:
        result_names, counts = [], []

    return result_names, counts


def compare_states(state, states, graph):
    for (i, st) in enumerate(states):
        h = connected_hamming_dist(state, st, graph)
        if h <= HAMMING/2:
            return True
    return False

def compute_union(results_folder, output_directory):
    names = []
    counts = []
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Folder '{output_directory}' created successfully.")
    
    for i, filename in enumerate(sorted(os.listdir(results_folder))):
        if filename.endswith(".csv"):
            name = filename.split("_")[0]
            file_path = os.path.join(results_folder, filename)
            df = pd.read_csv(file_path)
            count = df['Count'].max()
        names.append(name)
        counts.append(count)
    
    data = {'Name': names, 'Count': counts}
    df_output = pd.DataFrame(data)
    output_csv_path = os.path.join(output_directory, 'union.csv')
    df_output.to_csv(output_csv_path, index=False)
    return names, counts

            
def compute_droplets_in_union_dwave(results_folder, dwave_directory, min_df, instance_path, output_directory):
    names = []
    counts = []
    
    for i, filename in enumerate(sorted(os.listdir(results_folder))):
        if filename.endswith(".csv"):
            file_path = os.path.join(results_folder, filename)
            df = pd.read_csv(file_path)
            states = df['State'].values[-1]
            states_list = ast.literal_eval(states)

            name = filename.split("_")[0]
            
            file = os.path.join(dwave_directory, f"{name}.csv")
            file_inst = os.path.join(instance_path, name + "_sg.txt")
            graph = create_spin_glass_peps_graph(file_inst)
            if os.path.isfile(file):
                instance_df = pd.read_csv(file, index_col=0)
                ground_eng = min_df[min_df.index == name]['Energy'].values[0]
                energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)
                state_energy_tuple = get_state_energy_from_dwave(instance_df, energy_cutoff, TOPOLOGY_SIZE, ground_eng)
                dw_states = state_energy_tuple.state
                
                count = 0
                for st in states_list:
                    if compare_states(st, dw_states, graph):
                        count += 1
                
                names.append(name)
                counts.append(count)
    data = {'Name': names, 'Count': counts}
    df_output = pd.DataFrame(data)
    output_csv_path = os.path.join(output_directory, 'dwave.csv')
    df_output.to_csv(output_csv_path, index=False)
    return names, counts


def compute_droplets_in_union_spinglass(results_folder, json_directory, beta, eng, bd, ms, output_directory, instance_path, min_df):
    names = []
    counts = []
    spinglass_states = read_json_files(json_directory, beta, eng, bd, ms, min_df, APPROX_RATIO)
    name_range = [f"{i:03d}" for i in range(1, 21)]  # Generate filenames from "001" to "020"

    for filename in sorted(os.listdir(results_folder)):
        if filename.endswith(".csv") and any(name in filename for name in name_range):
            file_path = os.path.join(results_folder, filename)
            df = pd.read_csv(file_path)
            states = df['State'].values[-1]
            states_list = ast.literal_eval(states)

            name = filename.split("_")[0]
            tn_states = spinglass_states[name]
            tn_states = tn_states.state
            file_inst = os.path.join(instance_path, name + "_sg.txt")
            graph = create_spin_glass_peps_graph(file_inst)
            ground_eng = min_df[min_df.index == name]['Energy'].values[0]

            count = 0
            for st in states_list:
                if compare_states(st, tn_states, graph):
                    count += 1
                
            names.append(name)
            counts.append(count)
            
    data = {'Name': names, 'Count': counts}
    df_output = pd.DataFrame(data)
    output_csv_path = os.path.join(output_directory, f'beta{beta}_eng{eng}_bd{bd}_ms{ms}.csv')
    df_output.to_csv(output_csv_path, index=False)
            
    return names, counts


def compute_droplets_in_union_sbm(results_folder, h5_directory, output_directory, instance_path, min_df):
    names = []
    counts = []
    sbm_states  = read_h5_files(h5_directory, min_df, APPROX_RATIO)
    for i, filename in enumerate(sorted(os.listdir(results_folder))):
        if filename.endswith(".csv"):
            file_path = os.path.join(results_folder, filename)
            df = pd.read_csv(file_path)
            states = df['State'].values[-1]
            states_list = ast.literal_eval(states)

            name = filename.split("_")[0]
            file_inst = os.path.join(instance_path, name + "_sg.txt")
            graph = create_spin_glass_peps_graph(file_inst)
            sbm_state = sbm_states[name]
            sbm_state = sbm_state.state
            ground_eng = min_df[min_df.index == name]['Energy'].values[0]

            count = 0
            for st in states_list:
                if compare_states(st, sbm_state, graph):
                    count += 1
                
            names.append(name)
            counts.append(count)
    data = {'Name': names, 'Count': counts}
    df_output = pd.DataFrame(data)
    output_csv_path = os.path.join(output_directory, 'sbm.csv')
    df_output.to_csv(output_csv_path, index=False)
    return names, counts

def read_all_json_files(directory, beta, df_min) -> dict:
    instance_data = {}
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
        if os.path.isfile(file) and file.endswith(".json"):
            with open(file, encoding='utf-8') as f:
                json_data = json.load(f)
                if json_data['columns'][json_data['colindex']['lookup']['β']-1][0] == beta:
                    instance_name = json_data['columns'][json_data['colindex']['lookup']['instance']-1][0].split('_')[0]
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


def create_independent_set_of_union_states(dwave_directory, json_directory, h5_directory, best_found_path, instance_path, history_directory, metric: Optional[str], **kwargs):

    min_df = pd.read_csv(best_found_path, index_col=0)
    min_df.index = min_df.index.map(lambda x: str(x).zfill(3))
    if not os.path.exists(history_directory):
        os.makedirs(history_directory)
        print(f"Folder '{history_directory}' created successfully.")
    
    beta = kwargs["beta"]
    spinglass_states = read_all_json_files(json_directory, beta, min_df)
    # sbm_states = read_h5_files(h5_directory, min_df, APPROX_RATIO)
    for filename in tqdm(os.listdir(dwave_directory)):
        StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])

        file = os.path.join(dwave_directory, filename)
        name = filename.split(".")[0]
        name_range = [f"{i:03d}" for i in range(1, 21)]
        if name not in name_range:
            continue
        if os.path.isfile(file):
            instance_df = pd.read_csv(file, index_col=0)
            ground_eng = min_df[min_df.index == name]['Energy'].values[0]
            instance_parameters = f"_beta{beta}_{metric}_best{ground_eng}"
            history_file_name = os.path.join(history_directory, f"{name}{instance_parameters}.csv")
            energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)
            state_energy_dw = get_state_energy_from_dwave(instance_df, energy_cutoff, TOPOLOGY_SIZE, ground_eng)
            state_energy_tn = spinglass_states[name]
            # state_energy_sbm = sbm_states[name]
            union_data = {}
            # for nt in [state_energy_dw, state_energy_tn, state_energy_sbm]:
            for nt in [state_energy_dw, state_energy_tn]:
                for state, energy in zip_longest(nt.state, nt.energy, fillvalue=None):
                    state = tuple(state) if isinstance(state, np.ndarray) else state
                    if state not in union_data:
                        union_data[state] = energy

            # Create a new named tuple with unique states and corresponding energies
            unique_states = list(union_data.keys())
            unique_energies = [union_data[state] for state in unique_states]
            unique_namedtuple = StateEnergy(unique_states, unique_energies)
            if metric == "connected":
                file_inst = os.path.join(instance_path, name + "_sg.txt")
                graph = create_spin_glass_peps_graph(file_inst)
                state_energy = find_max_set_connected(ITERATIONS, unique_namedtuple, HAMMING, ground_eng, energy_cutoff, graph, history_file_name)     
            else:
                state_energy = find_max_set(ITERATIONS, unique_namedtuple, HAMMING, ground_eng, energy_cutoff, history_file_name)
    
    return #state_energy




if __name__ == '__main__':
    result = create_independent_set_of_union_states(dwave_directory, json_directory, h5_directory, minimum_path, instance_path, output_directory_union, 
                                                    "connected", beta=BETA)
    compute_union(output_directory_union, output_directory_solver)
    name_dw, count_dw = find_droplets_in_union(output_directory_union, json_directory, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "DWave")
    # name_sbm, count_sbm = find_droplets_in_union(output_directory_union, json_directory, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SBM")
    name, count = find_droplets_in_union(output_directory_union, json_directory, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SpinGlass",
                                         beta=BETA, eng=ENG, bd=BD, ms=MAX_STATES)
    # name, count = find_droplets_in_union(output_directory_union, json_directory, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SpinGlass",
    #                                      beta=BETA, eng=ENG, bd=BD, ms=MAX_STATES1)
    # name, count = find_droplets_in_union(output_directory_union, json_directory, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SpinGlass",
    #                                      beta=BETA, eng=ENG, bd=BD, ms=MAX_STATES2)
    # name, count = find_droplets_in_union(output_directory_union, json_directory, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SpinGlass",
    #                                      beta=BETA, eng=ENG, bd=BD1, ms=MAX_STATES)
    # name, count = find_droplets_in_union(output_directory_union, json_directory, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SpinGlass",
    #                                      beta=BETA, eng=ENG, bd=BD1, ms=MAX_STATES1)
    # name, count = find_droplets_in_union(output_directory_union, json_directory, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SpinGlass",
    #                                      beta=BETA, eng=ENG, bd=BD1, ms=MAX_STATES2)    
    # name, count = find_droplets_in_union(output_directory_union, json_directory, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SpinGlass",
    #                                      beta=BETA, eng=ENG, bd=BD2, ms=MAX_STATES)
    # name, count = find_droplets_in_union(output_directory_union, json_directory, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SpinGlass",
    #                                      beta=BETA, eng=ENG, bd=BD2, ms=MAX_STATES1)
    # name, count = find_droplets_in_union(output_directory_union, json_directory, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SpinGlass",
    #                                      beta=BETA, eng=ENG, bd=BD2, ms=MAX_STATES2) 