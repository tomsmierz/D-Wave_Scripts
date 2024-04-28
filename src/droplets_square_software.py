from matplotlib.lines import Line2D

import copy
import os
import json
import random
import time
import ast
import multiprocessing
import sys
import csv
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
from droplets import (xor, hamming_dist, create_spin_glass_peps_graph, connected_hamming_dist,
    find_droplets_hamming_connected, find_droplets_hamming, find_max_set_connected,
    find_max_set)
from read_data import (read_all_json_files, read_all_csv_files, read_json_files, read_h5_files)
# Instance characteristics
BETA = 3.0

HAMMING = 312 #147
APPROX_RATIO = 0.005
ITERATIONS=10
ENG= 20
BD = 16
MAX_STATES = 256
INST = int(sys.argv[sys.argv.index("--inst") + 1])
is_Z2 = False

script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
root = os.path.dirname(script_dir)
json_directory = os.path.join(root, "droplets", "square", "50x50_ground_droplets_ms256")
h5_directory = os.path.join(root, "energies", "sbm", "square", "50x50", "filtered_files_hd200_i020_ar005")
cplex_directory = os.path.join(root, "energies", "cplex", "square_50_results.csv")
minimum_path = os.path.join(root, "droplets", "square", "results", "50x50", "square50_allnew.csv")
instance_path = os.path.join(root, "instances", "square", "square_50x50", "single")
output_directory_union = os.path.join(root, "droplets", "square", "results", "50x50", "union_ar005_R0125_software_python38", f"{INST}")
output_directory_solver = os.path.join(root, "droplets", "square", "results","50x50", "solver_ar005_R0125_software_python38", f"{INST}")


vector = Union[np.ndarray, list]


def find_droplets_in_union(results_folder, json_directory, cplex_directory, h5_directory, output_directory, best_found_path, instance_path, solver, is_Z2: bool = False, **kwargs):
    if solver not in ["DWave", "SpinGlass", "SBM", "CPLEX", "PT", None]:
        raise ValueError("Solver should be \"DWave\", \"SpinGlass\", \"SBM\", \"CPLEX\", \"PT\" or None")
    min_df = pd.read_csv(best_found_path, index_col=0, delimiter=',', quotechar='"')
    min_df.index = min_df.index.map(lambda x: str(x).zfill(3))
    inst = kwargs["inst"]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Folder '{output_directory}' created successfully.")

    if solver == "SpinGlass":
        if "beta" not in kwargs or "eng" not in kwargs or "bd" not in kwargs or "ms" not in kwargs:
            raise ValueError("To compute droplets for SpinGlassPEPS parameters beta, eng (cutoff energy in "
                             "SpinGlassPEPS), bd (bond dimension) and ms (max states) are needed")
        beta = kwargs["beta"]
        eng = kwargs["eng"]
        bd = kwargs["bd"]
        ms = kwargs["ms"]
        result_names, counts = compute_droplets_in_union_spinglass(results_folder, json_directory, beta, eng, bd, ms, output_directory, instance_path, min_df, inst, is_Z2)

    elif solver == "SBM":
        result_names, counts = compute_droplets_in_union_sbm(results_folder, h5_directory, output_directory, instance_path, min_df, inst, is_Z2)

    elif solver == "CPLEX":
        result_names, counts = compute_droplets_in_union_cplex(results_folder, cplex_directory, output_directory, instance_path, min_df, inst, is_Z2)

    elif solver == "DWave":
        raise NotImplementedError()
    
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
            name = filename.split(".")[0] #TODO
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


def compute_droplets_in_union_spinglass(results_folder, json_directory, beta, eng, bd, ms, output_directory, instance_path, min_df, inst, is_Z2: bool = False):
    names = []
    counts = []
    sts = []
    idx = []
    indices = []
    spinglass_states = read_json_files(json_directory, beta, eng, bd, ms, min_df, APPROX_RATIO, is_Z2)
    name_range = [f"{inst:03d}",] 
    output_csv_path = os.path.join(output_directory, f'tn_{beta}.csv')

    name = name_range[0]
    tn_states = spinglass_states[name]
    tn_states = tn_states.state
        
    for i, filename in enumerate(sorted(os.listdir(results_folder))):
        fileres = os.path.join(results_folder, filename)
        df = pd.read_csv(fileres, index_col=0, delimiter=',', quotechar='"')
        states = df['State'].values[-1]
        states_list = ast.literal_eval(states)

        file_inst = os.path.join(instance_path, name + ".txt")
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


def compute_droplets_in_union_cplex(results_folder, cplex_directory, output_directory, instance_path, min_df, inst, is_Z2: bool = False):
    names = []
    counts = []
    sts = []
    idx = []
    indices = []
    cplex_states = read_all_csv_files(cplex_directory, min_df)
    name_range = [f"{inst:03d}",] 
    name = name_range[0]
    output_csv_path = os.path.join(output_directory, f'cplex.csv')

    for filename in sorted(os.listdir(results_folder)):
        file_path = os.path.join(results_folder, filename)
        df = pd.read_csv(file_path, index_col=0, delimiter=',', quotechar='"')
        states = df['State'].values[-1]
        states_list = ast.literal_eval(states)

        cplex_states = cplex_states[name]
        cplex_states = cplex_states.state
        file_inst = os.path.join(instance_path, name + ".txt")
        graph = create_spin_glass_peps_graph(file_inst)
      
        count = 0
        for st in cplex_states:
            result, i = compare_states(st, states_list, graph)
            if result is not False:
                count += 1
                sts.append(result)
                idx.append(i)
            else:
                print("CPLEX Nie znaleziono dopasowania dla stanu:", st)        
        
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


def compute_droplets_in_union_sbm(results_folder, h5_directory, output_directory, instance_path, min_df, inst, is_Z2: bool = False):
    names = []
    counts = []
    sts = []
    idx = []
    indices = []
    sbm_states  = read_h5_files(h5_directory, min_df, APPROX_RATIO, inst, is_Z2)
    name_range = [f"{inst:03d}",] 
    name = name_range[0]
    output_csv_path = os.path.join(output_directory, 'sbm.csv')

    for i, filename in enumerate(sorted(os.listdir(results_folder))):
        fileres = os.path.join(results_folder, filename)
        df = pd.read_csv(fileres, index_col=0, delimiter=',', quotechar='"')
        states = df['State'].values[-1]
        states_list = ast.literal_eval(states)

        file_inst = os.path.join(instance_path, name + ".txt")
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

def create_independent_set_of_union_states(json_directory, cplex_directory, h5_directory, best_found_path, instance_path, history_directory, metric: Optional[str], is_Z2: bool = False, **kwargs):
    min_df = pd.read_csv(best_found_path, index_col=0, delimiter=',', quotechar='"')
    min_df.index = min_df.index.map(lambda x: str(x).zfill(3))
    if not os.path.exists(history_directory):
        os.makedirs(history_directory)
        print(f"Folder '{history_directory}' created successfully.")
    
    beta = kwargs["beta"]

    spinglass_states = read_all_json_files(json_directory, beta, APPROX_RATIO, min_df, is_Z2)
    cplex_states = read_all_csv_files(cplex_directory, min_df)

    inst = kwargs["inst"]
    sbm_states = read_h5_files(h5_directory, min_df, APPROX_RATIO, inst, is_Z2)
    for filename in tqdm(os.listdir(instance_path)):
        StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])

        file = os.path.join(instance_path, filename)
        name = filename.split(".")[0]
        name_range = [f"{inst:03d}",]
        if name not in name_range:
            continue
        if os.path.isfile(file):
            ground_eng = min_df[min_df.index == name]['Energy'].values[0]
            history_file_name = os.path.join(history_directory, f"{name}.csv")
            energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)
            state_energy_tn = spinglass_states[name]

            state_energy_cplex = cplex_states[name]
            state_energy_sbm = sbm_states[name]
            union_data = {}
            for nt in [state_energy_tn, state_energy_sbm, state_energy_cplex]:
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


if __name__ == '__main__':
    result = create_independent_set_of_union_states(json_directory, cplex_directory, h5_directory, minimum_path, instance_path, output_directory_union, 
                                                    "connected", is_Z2, beta=BETA, inst=INST)
    compute_union(output_directory_union, output_directory_solver)
    name, count = find_droplets_in_union(output_directory_union, json_directory, cplex_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SpinGlass", is_Z2,
                                    beta=BETA, eng=ENG, bd=BD, ms=MAX_STATES, inst=INST)
    name_cplex, count_cplex = find_droplets_in_union(output_directory_union, json_directory, cplex_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "CPLEX", is_Z2, inst=INST)
    name_sbm, count_sbm = find_droplets_in_union(output_directory_union, json_directory, cplex_directory, h5_directory, output_directory_solver, minimum_path, instance_path, "SBM", is_Z2, inst=INST)
