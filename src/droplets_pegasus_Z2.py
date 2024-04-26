from matplotlib.lines import Line2D

import copy
import os
import json
import random
import time
import ast
import sys

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
from droplets_Z2 import (filter_states_by_energy, get_state_energy_from_dwave, 
    read_json_files, read_h5_files, xor, hamming_dist, create_spin_glass_peps_graph, connected_hamming_dist,
    filter_states_by_energy, find_droplets_hamming_connected, find_droplets_hamming, find_max_set_connected,
    find_max_set, array_from_dict)

# Instance characteristics
TOPOLOGY = "pegasus"
INSTANCE_SYMBOL = "P16"
INSTANCE_TYPE = "RCO"
TOPOLOGY_SIZE = 16
BETA = 0.5
HAMMING = 588 #147
APPROX_RATIO = 0.01
ITERATIONS=2
ENG= 40
BD = 8
BD2 = 8
MAX_STATES = 1024 #1024
INST = int(sys.argv[sys.argv.index("--inst") + 1])
TRUNCATION=16
TRUNCATION20=20

script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
root = os.path.dirname(script_dir)
json_directory1 = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, f"final_bench_truncate2^{TRUNCATION} ")
json_directory2 = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, f"final_bench_truncate2^{TRUNCATION20}")
# json_directory3 = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, f"final_bench_notr")

dwave_directory = os.path.join(root, "energies", f"{TOPOLOGY}_random_aggregated", INSTANCE_SYMBOL, INSTANCE_TYPE, "filtered_files_hd100_i020")
h5_directory = os.path.join(root, "energies", "sbm", f"{TOPOLOGY}_random", INSTANCE_SYMBOL, INSTANCE_TYPE,
                            "filtered_files_hd1300_ar01_i020")
minimum_path = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, f"minimum_truncated2^16_allsolvers.csv")
instance_path = os.path.join(root, "instances", f"{TOPOLOGY}_random", INSTANCE_SYMBOL, INSTANCE_TYPE)
output_directory_union = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, f"results_union_ar01_dr05_allsolvers-new2", f"{INST}")
output_directory_solver = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, f"results_solver_ar01_dr05_allsolvers-new2", f"{INST}")

vector = Union[np.ndarray, list]

def find_droplets_in_union(results_folder, json_directory, dwave_directory, h5_directory, output_directory, best_found_path, instance_path, solver, **kwargs):
    if solver not in ["DWave", "SpinGlass", "SBM", "PT", None]:
        raise ValueError("Solver should be \"DWave\", \"SpinGlass\", \"SBM\", \"PT\" or None")

    min_df = pd.read_csv(best_found_path, index_col=0, delimiter=',', quotechar='"')
    min_df.index = min_df.index.map(lambda x: str(x).zfill(3))
    inst = kwargs["inst"]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Folder '{output_directory}' created successfully.")

    if solver == "DWave":
        result_names, counts = compute_droplets_in_union_dwave(results_folder, dwave_directory, min_df, instance_path, output_directory, inst)

    elif solver == "SpinGlass":
        if "beta" not in kwargs or "eng" not in kwargs or "bd" not in kwargs or "ms" not in kwargs:
            raise ValueError("To compute droplets for SpinGlassPEPS parameters beta, eng (cutoff energy in "
                             "SpinGlassPEPS), bd (bond dimension) and ms (max states) are needed")
        beta = kwargs["beta"]
        eng = kwargs["eng"]
        bd = kwargs["bd"]
        ms = kwargs["ms"]
        text = kwargs["text"]
        result_names, counts = compute_droplets_in_union_spinglass(results_folder, json_directory, beta, eng, bd, ms, output_directory, instance_path, min_df, inst, text)

    elif solver == "SBM":
        result_names, counts = compute_droplets_in_union_sbm(results_folder, h5_directory, output_directory, instance_path, min_df, inst)

    elif solver == "PT":
        raise NotImplementedError()

    else:
        result_names, counts = [], []

    return result_names, counts

def compare_states(state, states, graph):
    min_distance = float('inf') 
    min_state = None 
    min_index = None 
    
    for i, st in enumerate(states):
        h = connected_hamming_dist(state, st, graph)
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
            name = filename.split("_")[0]
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


def compute_droplets_in_union_dwave(results_folder, dwave_directory, min_df, instance_path, output_directory, inst):
    names = []
    counts = []
    sts = []
    idx = []
    indices = []
    name_range = [f"{inst:03d}",] 
    output_csv_path = os.path.join(output_directory, f'dwave.csv')
    name = name_range[0]
    
    for i, filename in enumerate(sorted(os.listdir(results_folder))):
        fileres = os.path.join(results_folder, filename)
        df = pd.read_csv(fileres, index_col=0, delimiter=',', quotechar='"')
        states = df['State'].values[-1]
        states_list = ast.literal_eval(states)

        file = os.path.join(dwave_directory, f"{name}.csv")
        file_inst = os.path.join(instance_path, name + "_sg.txt")
        graph = create_spin_glass_peps_graph(file_inst)
        if os.path.isfile(file):
            instance_df = pd.read_csv(file, index_col=0, delimiter=',',quotechar='"')
            ground_eng = min_df[min_df.index == name]['Energy'].values[0]
            energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)
            state_energy_tuple = get_state_energy_from_dwave(instance_df, energy_cutoff, TOPOLOGY_SIZE, ground_eng)
            dw_states = state_energy_tuple.state
                        
            count = 0
            for st in dw_states:
                result, i = compare_states(st, states_list, graph)
                if result is not False:
                    count += 1
                    sts.append(result)
                    idx.append(i)
                else:
                    print("DW Nie znaleziono dopasowania dla stanu:", st)        
            tablica_numpy = np.array(sts)
            unikalne_wiersze = np.unique(tablica_numpy, axis=0)
            cts = unikalne_wiersze.shape[0]            
            names.append(name)
            counts.append(cts)
            
            tablica_numpy2 = np.array(idx)
            unikalne_wiersze2 = np.unique(tablica_numpy2, axis=0)
            indices.append(unikalne_wiersze2)
            data = {'Name': names, 'Count': counts, 'Indices' : indices}
            df_output = pd.DataFrame(data)
            df_output.to_csv(output_csv_path, index=False, header=not os.path.exists(output_csv_path), mode='a')
    return names, counts


def compute_droplets_in_union_spinglass(results_folder, json_directory, beta, eng, bd, ms, output_directory, instance_path, min_df, inst, text):
    names = []
    counts = []
    sts = []
    idx = []
    indices = []
    spinglass_states = read_json_files(json_directory, beta, eng, bd, ms, min_df, APPROX_RATIO)
    # name_range = [f"{i:03d}" for i in range(1, inst)]  # Generate filenames from "001" to "020"
    name_range = [f"{inst:03d}",] 
    # output_csv_path = os.path.join(output_directory, f'beta{beta}_eng{eng}_bd{bd}_ms{ms}_{text}.csv')
    output_csv_path = os.path.join(output_directory, f'tn_{text}.csv')
    tn_output_path = os.path.join(output_directory, f'tn_states_{text}.txt')  # Ścieżka do pliku z tn_states

    name = name_range[0]
    tn_states = spinglass_states[name]
    tn_states = tn_states.state
    with open(tn_output_path, 'w') as f:
        for state in tn_states:
            f.write(f"{state}\n")
        
    for i, filename in enumerate(sorted(os.listdir(results_folder))):
        fileres = os.path.join(results_folder, filename)
        df = pd.read_csv(fileres, index_col=0, delimiter=',', quotechar='"')
        states = df['State'].values[-1]
        states_list = ast.literal_eval(states)

        file_inst = os.path.join(instance_path, name + "_sg.txt")
        graph = create_spin_glass_peps_graph(file_inst)

        ground_eng = min_df[min_df.index == name]['Energy'].values[0]
       
        count = 0
        for st in tn_states:
            result, i = compare_states(st, states_list, graph)
            if result is not False:
                count += 1
                sts.append(result)
                idx.append(i)
            else:
                print("TN Nie znaleziono dopasowania dla stanu:", st)        
        
        tablica_numpy = np.array(sts)
        unikalne_wiersze = np.unique(tablica_numpy, axis=0)
        cts = unikalne_wiersze.shape[0]            
        names.append(name)
        counts.append(cts)
            
        tablica_numpy2 = np.array(idx)
        unikalne_wiersze2 = np.unique(tablica_numpy2, axis=0)
        indices.append(unikalne_wiersze2)
        data = {'Name': names, 'Count': counts, 'Indices' : indices}
        df_output = pd.DataFrame(data)
        df_output.to_csv(output_csv_path, index=False, header=not os.path.exists(output_csv_path), mode='a')
          
    return names, counts


def compute_droplets_in_union_sbm(results_folder, h5_directory, output_directory, instance_path, min_df, inst):
    names = []
    counts = []
    sts = []
    idx = []
    indices = []
    sbm_states  = read_h5_files(h5_directory, min_df, APPROX_RATIO, inst)
    name_range = [f"{inst:03d}",] 
    name = name_range[0]
    output_csv_path = os.path.join(output_directory, 'sbm.csv')

    for i, filename in enumerate(sorted(os.listdir(results_folder))):
        fileres = os.path.join(results_folder, filename)
        df = pd.read_csv(fileres, index_col=0, delimiter=',', quotechar='"')
        states = df['State'].values[-1]
        states_list = ast.literal_eval(states)

        file_inst = os.path.join(instance_path, name + "_sg.txt")
        graph = create_spin_glass_peps_graph(file_inst)

        sbm_state = sbm_states[name]
        sbm_state = sbm_state.state
        ground_eng = min_df[min_df.index == name]['Energy'].values[0]

        count = 0
        for st in sbm_state:
            result, i = compare_states(st, states_list, graph)
            if result is not False:
                count += 1
                sts.append(result)
                idx.append(i)
            else:
                print("SBM Nie znaleziono dopasowania dla stanu:", st)        
        
        tablica_numpy = np.array(sts)
        unikalne_wiersze = np.unique(tablica_numpy, axis=0)
        cts = unikalne_wiersze.shape[0]            
        names.append(name)
        counts.append(cts)
            
        tablica_numpy2 = np.array(idx)
        unikalne_wiersze2 = np.unique(tablica_numpy2, axis=0)
        indices.append(unikalne_wiersze2)
        data = {'Name': names, 'Count': counts, 'Indices' : indices}
        df_output = pd.DataFrame(data)
        df_output.to_csv(output_csv_path, index=False, header=not os.path.exists(output_csv_path), mode='a')
            
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


# def create_independent_set_of_union_states(dwave_directory, json_directory1, json_directory2, json_directory3, h5_directory, best_found_path, instance_path, history_directory, metric: Optional[str], **kwargs):
def create_independent_set_of_union_states(dwave_directory, json_directory1, h5_directory, best_found_path, instance_path, history_directory, metric: Optional[str], **kwargs):
    min_df = pd.read_csv(best_found_path, index_col=0, delimiter=',', quotechar='"')
    min_df.index = min_df.index.map(lambda x: str(x).zfill(3))
    if not os.path.exists(history_directory):
        os.makedirs(history_directory)
        print(f"Folder '{history_directory}' created successfully.")
    
    beta = kwargs["beta"]
    spinglass_states1 = read_all_json_files(json_directory1, beta, min_df)
    # spinglass_states2 = read_all_json_files(json_directory2, beta, min_df)
    # spinglass_states3 = read_all_json_files(json_directory3, beta, min_df)

    inst = kwargs["inst"]
    sbm_states = read_h5_files(h5_directory, min_df, APPROX_RATIO, inst)
    for filename in tqdm(os.listdir(dwave_directory)):
        StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])

        file = os.path.join(dwave_directory, filename)
        name = filename.split(".")[0]
        name_range = [f"{inst:03d}",]
        if name not in name_range:
            continue
        if os.path.isfile(file):
            instance_df = pd.read_csv(file, index_col=0, delimiter=',',quotechar='"')
            ground_eng = min_df[min_df.index == name]['Energy'].values[0]
            instance_parameters = f"_beta{beta}_{metric}_best{ground_eng}"
            history_file_name = os.path.join(history_directory, f"{name}{instance_parameters}.csv")
            energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)
            state_energy_dw = get_state_energy_from_dwave(instance_df, energy_cutoff, TOPOLOGY_SIZE, ground_eng)
            state_energy_tn1 = spinglass_states1[name]
            # state_energy_tn2 = spinglass_states2[name]
            # state_energy_tn3 = spinglass_states3[name]
            state_energy_sbm = sbm_states[name]

            union_data = {}
            # for nt in [state_energy_dw, state_energy_tn1, state_energy_tn2, state_energy_tn3, state_energy_sbm]:
            for nt in [state_energy_dw, state_energy_tn1, state_energy_sbm]:
            # for nt in [state_energy_dw, state_energy_tn]:
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
    # result = create_independent_set_of_union_states(dwave_directory, json_directory1, json_directory2, json_directory3, h5_directory, minimum_path, instance_path, output_directory_union, 
    #                                                 "connected", beta=BETA, inst=INST)
    result = create_independent_set_of_union_states(dwave_directory, json_directory1, h5_directory, minimum_path, instance_path, output_directory_union, 
                                                    "connected", beta=BETA, inst=INST)
    compute_union(output_directory_union, output_directory_solver)
    name, count = find_droplets_in_union(output_directory_union, json_directory1, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SpinGlass",
                                    beta=BETA, eng=ENG, bd=BD, ms=MAX_STATES, inst=INST, text="tr16")
    # name, count = find_droplets_in_union(output_directory_union, json_directory2, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SpinGlass",
    #                                 beta=BETA, eng=ENG, bd=BD, ms=MAX_STATES, inst=INST, text="tr20")
    # name, count = find_droplets_in_union(output_directory_union, json_directory3, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SpinGlass",
    #                                 beta=BETA, eng=ENG, bd=BD2, ms=MAX_STATES, inst=INST, text="notr")
    name_dw, count_dw = find_droplets_in_union(output_directory_union, json_directory1, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "DWave", inst=INST)
    name_sbm, count_sbm = find_droplets_in_union(output_directory_union, json_directory1, dwave_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SBM", inst=INST)