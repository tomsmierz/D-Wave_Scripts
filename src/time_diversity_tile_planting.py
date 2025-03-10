from matplotlib.lines import Line2D
from multiprocessing import Pool, cpu_count

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
from droplets import (filter_states_by_energy, 
    read_h5_files, xor, hamming_dist, create_spin_glass_peps_graph, connected_hamming_dist,
    filter_states_by_energy, find_droplets_hamming_connected, find_droplets_hamming, find_max_set_connected,
    find_max_set, array_from_dict)


BETA = 0.5
# m = 200  
n_dw1 = 10
n_dw2 = 100
n_dw3 = 1000
n_dw4 = 10000
n_tn1 = 1
n_tn2 = 2
n_tn4 = 4
n_tn8 = 8
# Instance characteristic
TOPOLOGY = "P16"
TOPOLOGY_SIZE = "16"
L = "40"

# Directories
script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
root = os.path.dirname(script_dir)

dwave_directory = os.path.join(root, "embedded_tile_planting_simplified","embedded_tile_planting_spectra","dwave", f"{TOPOLOGY}_adv6.4_results", f"L_{L}_p2_1")
sbm_directory = os.path.join(root, "embedded_tile_planting_simplified","embedded_tile_planting_spectra","sbm", f"{TOPOLOGY}_adv6.4_results", f"L_{L}_p2_1")
json_directory = os.path.join(root, "embedded_tile_planting_simplified", "tn_results", f"{TOPOLOGY}_adv6.4_results", "new", f"{TOPOLOGY}_L{L}_droplets")
minimum_path = os.path.join(root, "embedded_tile_planting_simplified", "best", f"{TOPOLOGY}_adv6.4_results", f"best_{TOPOLOGY}_2D_L_{L}.csv")
instance_path = os.path.join(root, "embedded_tile_planting_simplified", "instances", f"{TOPOLOGY}_adv6.4", "nonzero")
output_directory_union = os.path.join(root, "embedded_tile_planting_simplified", "droplets", f"{TOPOLOGY}_adv6.4", f"results_union_ar01_dr025")
output_directory_solver = os.path.join(root, "embedded_tile_planting_simplified", "droplets", f"{TOPOLOGY}_adv6.4", f"results_solver_ar01_dr025")
union = os.path.join(root, "embedded_tile_planting_simplified", "droplets", f"{TOPOLOGY}_adv6.4", f"results_solver_ar01_dr025", "merged_union.csv")

HAMMING = 766 #1344 #147
APPROX_RATIO = 0.01
ITERATIONS=10
ENG= 7
BD = 8
MAX_STATES = 1024 #256 #1024
m=20


vector = Union[np.ndarray, list]


def read_json_files(directory, beta, eng, bd, ms, df_min, approx_ratio) -> dict:
    instance_data = {}
    
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])

    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        if os.path.isfile(file) and file.endswith(".csv"):
            with open(file, encoding='utf-8') as f:
                json_data = json.load(f)
                
                if json_data['columns'][json_data['colindex']['lookup']['β']-1][0] == beta and \
                   json_data['columns'][json_data['colindex']['lookup']['bond_dim']-1][0] == bd and \
                   json_data['columns'][json_data['colindex']['lookup']['max_states']-1][0] == ms:

                    instance_name = json_data['columns'][json_data['colindex']['lookup']['instance']-1][0].split('_P')[0]
                    if instance_name == "BP":
                        continue
                    else:
                        energy_data = json_data['columns'][json_data['colindex']['lookup']['drop_eng']-1][0]
                        state_data = json_data['columns'][json_data['colindex']['lookup']['ig_states']-1][0]
                        
                        state_data_np = array_from_dict(state_data)
                        energy_data_np = np.array(energy_data)
                        
                        ground_eng = df_min[df_min.index == instance_name]['Energy'].values[0]
                        energy_cutoff = approx_ratio * 2 * np.abs(ground_eng)
                        
                        filtered_states, filtered_energies = filter_states_by_energy(state_data_np, energy_data_np, energy_cutoff, ground_eng)

                        if instance_name not in instance_data:
                            instance_data[instance_name] = StateEnergy(state=[], energy=[])

                        existing_state_energy = instance_data[instance_name]
                        
                        existing_state_energy.state.append(filtered_states)
                        existing_state_energy.energy.append(filtered_energies)

                        instance_data[instance_name] = StateEnergy(existing_state_energy.state, existing_state_energy.energy)
                    
                else:
                    pass

    return instance_data

def find_droplets_in_union(results_folder, json_directory, dwave_directory, h5_directory, output_directory, best_found_path, instance_path, solver, **kwargs):
    if solver not in ["DWave", "SpinGlass", "SBM", "PT", None]:
        raise ValueError("Solver should be \"DWave\", \"SpinGlass\", \"SBM\", \"PT\" or None")

    min_df = pd.read_csv(best_found_path, index_col=0, delimiter=',', quotechar='"')
    min_df.index = min_df.index.map(lambda x: str(x).zfill(3))
    inst = kwargs["inst"]
    n = kwargs["n"]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Folder '{output_directory}' created successfully.")

    if solver == "DWave":
        result_names, counts = compute_droplets_in_union_dwave(results_folder, dwave_directory, min_df, instance_path, output_directory, inst, n)

    elif solver == "SpinGlass":
        if "beta" not in kwargs or "eng" not in kwargs or "bd" not in kwargs or "ms" not in kwargs:
            raise ValueError("To compute droplets for SpinGlassPEPS parameters beta, eng (cutoff energy in "
                             "SpinGlassPEPS), bd (bond dimension) and ms (max states) are needed")
        beta = kwargs["beta"]
        eng = kwargs["eng"]
        bd = kwargs["bd"]
        ms = kwargs["ms"]
        text = kwargs["text"]
        result_names, counts = compute_droplets_in_union_spinglass(results_folder, json_directory, beta, eng, bd, ms, output_directory, instance_path, min_df, inst, text, n)

    elif solver == "SBM":
        result_names, counts = compute_droplets_in_union_sbm(results_folder, h5_directory, min_df, instance_path, output_directory, inst, n)

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

def get_state_energy_from_sbm(data: pd.DataFrame, cutoff_energy: float,
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

def compute_droplets_in_union_dwave(results_folder, dwave_directory, min_df, instance_path, output_directory, inst, n):
    names = []
    counts = []
    sts = []
    idx = []
    indices = []
    name_range = [inst,]
    name = name_range[0] 
    name_inst = f"tile_planting_2D_L_{L}_p1_0.0_p2_1.0_p3_0.0_inst_{name}"
    
    # output_csv_path = os.path.join(output_directory, 'name', f'dwave.csv')
    results_folder = os.path.join(results_folder, f'{inst}')
    for i, filename in enumerate(sorted(os.listdir(results_folder))):
        fileres = os.path.join(results_folder, filename)
        df = pd.read_csv(fileres, index_col=0, delimiter=',', quotechar='"')
        states = df['State'].values[-1]
        states_list = ast.literal_eval(states)

        file = os.path.join(dwave_directory, f"response_{TOPOLOGY}_tile_planting_2D_L_{L}_p1_0.0_p2_1.0_p3_0.0_inst_{name}.csv")
        file_inst = os.path.join(instance_path,  f"tile_planting_2D_L_{L}_p1_0.0_p2_1.0_p3_0.0_inst_{name}_{TOPOLOGY}_nonzero.txt")
        graph = create_spin_glass_peps_graph(file_inst)
        if os.path.isfile(file):
            instance_df = pd.read_csv(file, index_col=0, delimiter=',',quotechar='"')
            ground_eng = min_df[min_df.index == name_inst]['Energy'].values[0]
            energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)
            state_energy_tuple = get_state_energy_from_dwave(instance_df, energy_cutoff, TOPOLOGY_SIZE, ground_eng)
            dw_states = state_energy_tuple.state
            dw_states_random = random.choices(dw_states, k=n) 
                           
            count = 0
            for st in dw_states_random:
                result, i = compare_states(st, states_list, graph)
                if result is not False:
                    count += 1
                    sts.append(result)
                    idx.append(i)
                # else:
                #     print("DW Nie znaleziono dopasowania dla stanu")        
            tablica_numpy = np.array(sts)
            unikalne_wiersze = np.unique(tablica_numpy, axis=0)
            cts = unikalne_wiersze.shape[0]            
            names.append(name)
            counts.append(cts)
            
            # tablica_numpy2 = np.array(idx)
            # unikalne_wiersze2 = np.unique(tablica_numpy2, axis=0)
            # indices.append(unikalne_wiersze2)
            # data = {'Name': names, 'Count': counts, 'Indices' : indices}
            # df_output = pd.DataFrame(data)
            # df_output.to_csv(output_csv_path, index=False, header=not os.path.exists(output_csv_path), mode='a')
    return names, counts

def compute_droplets_in_union_sbm(results_folder, sbm_directory, min_df, instance_path, output_directory, inst, n):
    names = []
    counts = []
    sts = []
    idx = []
    indices = []
    name_range = [inst,]
    name = name_range[0] 
    name_inst = f"tile_planting_2D_L_{L}_p1_0.0_p2_1.0_p3_0.0_inst_{name}"
    # output_csv_path = os.path.join(output_directory, 'name', f'dwave.csv')
    results_folder = os.path.join(results_folder, f'{inst}')
    for i, filename in enumerate(sorted(os.listdir(results_folder))):
        fileres = os.path.join(results_folder, filename)
        df = pd.read_csv(fileres, index_col=0, delimiter=',', quotechar='"')
        states = df['State'].values[-1]
        states_list = ast.literal_eval(states)

        file = os.path.join(sbm_directory, f"response_{TOPOLOGY}_tile_planting_2D_L_{L}_p1_0.0_p2_1.0_p3_0.0_inst_{name}.csv")
        file_inst = os.path.join(instance_path,  f"tile_planting_2D_L_{L}_p1_0.0_p2_1.0_p3_0.0_inst_{name}_{TOPOLOGY}_nonzero.txt")
        graph = create_spin_glass_peps_graph(file_inst)
        if os.path.isfile(file):
            instance_df = pd.read_csv(file, index_col=0, delimiter=',',quotechar='"')
            ground_eng = min_df[min_df.index == name_inst]['Energy'].values[0]
            energy_cutoff = APPROX_RATIO * 2 * np.abs(ground_eng)
            state_energy_tuple = get_state_energy_from_sbm(instance_df, energy_cutoff, TOPOLOGY_SIZE, ground_eng)
            dw_states = state_energy_tuple.state
            dw_states_random = random.choices(dw_states, k=n) 
                           
            count = 0
            for st in dw_states_random:
                result, i = compare_states(st, states_list, graph)
                if result is not False:
                    count += 1
                    sts.append(result)
                    idx.append(i)
                # else:
                #     print("DW Nie znaleziono dopasowania dla stanu")        
            tablica_numpy = np.array(sts)
            unikalne_wiersze = np.unique(tablica_numpy, axis=0)
            cts = unikalne_wiersze.shape[0]            
            names.append(name)
            counts.append(cts)
            
            # tablica_numpy2 = np.array(idx)
            # unikalne_wiersze2 = np.unique(tablica_numpy2, axis=0)
            # indices.append(unikalne_wiersze2)
            # data = {'Name': names, 'Count': counts, 'Indices' : indices}
            # df_output = pd.DataFrame(data)
            # df_output.to_csv(output_csv_path, index=False, header=not os.path.exists(output_csv_path), mode='a')
    return names, counts


def compute_droplets_in_union_spinglass(results_folder, json_directory, beta, eng, bd, ms, output_directory, instance_path, min_df, inst, text, n):
    names = []
    counts = []
    sts = []
    idx = []
    indices = []
    spinglass_states = read_json_files(json_directory, beta, eng, bd, ms, min_df, APPROX_RATIO)
    name_range = [inst,] 
    name = name_range[0] 
    name_inst = f"tile_planting_2D_L_{L}_p1_0.0_p2_1.0_p3_0.0_inst_{name}"

    # output_csv_path = os.path.join(output_directory, 'name', f'tn_{text}.csv')
    results_folder = os.path.join(results_folder, f'{inst}')
    tn_states = spinglass_states[name_inst]
    tn_states = tn_states.state
    tn_states_random = random.choices(tn_states, k=n) 
    
    for i, filename in enumerate(sorted(os.listdir(results_folder))):
        fileres = os.path.join(results_folder, filename)
        df = pd.read_csv(fileres, index_col=0, delimiter=',', quotechar='"')
        states = df['State'].values[-1]
        states_list = ast.literal_eval(states)

        file_inst = os.path.join(instance_path,  f"tile_planting_2D_L_{L}_p1_0.0_p2_1.0_p3_0.0_inst_{name}_{TOPOLOGY}_nonzero.txt")
        graph = create_spin_glass_peps_graph(file_inst)

        ground_eng = min_df[min_df.index == name_inst]['Energy'].values[0]
       
        count = 0
        for st in tn_states_random:
            for s in st:
                result, i = compare_states(s, states_list, graph)
                if result is not False:
                    count += 1
                    sts.append(result)
                    idx.append(i)
            # else:
            #     print("TN Nie znaleziono dopasowania dla stanu")        
        
        tablica_numpy = np.array(sts)
        unikalne_wiersze = np.unique(tablica_numpy, axis=0)
        cts = unikalne_wiersze.shape[0]            
        names.append(name)
        counts.append(cts)
            
        # tablica_numpy2 = np.array(idx)
        # unikalne_wiersze2 = np.unique(tablica_numpy2, axis=0)
        # indices.append(unikalne_wiersze2)
        # data = {'Name': names, 'Count': counts, 'Indices' : indices}
        # df_output = pd.DataFrame(data)
        # df_output.to_csv(output_csv_path, index=False, header=not os.path.exists(output_csv_path), mode='a')
          
    return names, counts



def get_diversity_sbm(output_directory_union, output_directory_solver, json_directory1, dwave_directory, h5_directory, best_energies, union, instance_path, m, n, **kwargs):
    all_diversity_ratios = []
    medians = []
    for _ in range(m):
        sampled_instance = random.randint(1, 5)
        si = [sampled_instance,] 
        name = si[0]
        df = pd.read_csv(union)
        name = f"tile_planting_2D_L_{L}_p1_0.0_p2_1.0_p3_0.0_inst_{name}.csv"
        union_states = df[df['Name'] == name]
        union_states = union_states["Count"]
        name_sbm, count_sbm = find_droplets_in_union(output_directory_union, json_directory1, dwave_directory, h5_directory, output_directory_solver, best_energies, instance_path, "SBM", n=n, inst=sampled_instance)
        diversity_ratio = count_sbm[0]/union_states
        # print("sbm10 ", count_sbm[0])
        # print("union ", union_states)
        all_diversity_ratios.append(diversity_ratio)
        # print(len(all_diversity_ratios), diversity_ratio)
    for _ in range(100):
        selected_energies = random.sample(all_diversity_ratios, 10)
        median = np.median(selected_energies)
        medians.append(median)

    median = np.median(medians)
    percentile_10 = np.percentile(medians, 2.5)
    percentile_90 = np.percentile(medians, 97.5)
    return median, percentile_10, percentile_90


def get_diversity_dw(output_directory_union, output_directory_solver, json_directory1, dwave_directory, h5_directory, best_energies, union, instance_path, m, n, **kwargs):
    all_diversity_ratios = []
    medians = []
    for _ in range(m):
        sampled_instance = random.randint(1, 5)
        si = [sampled_instance,] 
        name = si[0]
        name = f"tile_planting_2D_L_{L}_p1_0.0_p2_1.0_p3_0.0_inst_{name}.csv"

        df = pd.read_csv(union)
        union_states = df[df['Name'] == name]
        union_states = union_states["Count"]
        name_dw, count_dw = find_droplets_in_union(output_directory_union, json_directory1, dwave_directory, h5_directory, output_directory_solver, best_energies, instance_path, "DWave", n=n, inst=sampled_instance)
        diversity_ratio = count_dw[0]/union_states
        all_diversity_ratios.append(diversity_ratio)
    for _ in range(100):
        selected_energies = random.sample(all_diversity_ratios, 10)
        median = np.median(selected_energies)
        medians.append(median)

    median = np.median(medians)
    percentile_10 = np.percentile(medians, 2.5)
    percentile_90 = np.percentile(medians, 97.5)
    return median, percentile_10, percentile_90



def get_diversity_tn(output_directory_union, output_directory_solver, json_directory1, dwave_directory, h5_directory, best_energies, union, instance_path, m, n, **kwargs):
    all_diversity_ratios = []
    beta = kwargs["beta"]
    eng = kwargs["eng"]
    bd = kwargs["bd"]
    ms = kwargs["ms"]
    text = kwargs["text"]
    medians = []
    for _ in range(m):
        sampled_instance = random.randint(1, 5)
        si = [sampled_instance,] 
        name = si[0]
        name = f"tile_planting_2D_L_{L}_p1_0.0_p2_1.0_p3_0.0_inst_{name}.csv"

        df = pd.read_csv(union)
        union_states = df[df['Name'] == name]
        union_states = union_states["Count"]
        name, count = find_droplets_in_union(output_directory_union, json_directory1, dwave_directory, h5_directory, output_directory_solver, best_energies, instance_path, "SpinGlass",  n=n,
                                    beta=beta, eng=eng, bd=bd, ms=ms, inst=sampled_instance, text=text)
        diversity_ratio = count[0]/union_states

        all_diversity_ratios.append(diversity_ratio)

    for _ in range(100):
        selected_energies = random.sample(all_diversity_ratios, 10)
        median = np.median(selected_energies)
        medians.append(median)

    median = np.median(medians)
    # print(median)
    percentile_10 = np.percentile(medians, 2.5)
    percentile_90 = np.percentile(medians, 97.5)
    return median, percentile_10, percentile_90

def execute_solver(solver, diversity_function, n, output_directory_union, output_directory_solver, json_directory1, dwave_directory, h5_directory, minimum_path, union, instance_path, m):
    if any(prefix in solver for prefix in ["TN16", "TN20", "TN"]):
        median, percentile_10, percentile_90 = diversity_function(
            output_directory_union, output_directory_solver, json_directory1, dwave_directory, h5_directory,
            minimum_path, union, instance_path, m, n, beta=BETA, eng=ENG, bd=BD, ms=MAX_STATES, text="")
    else:
        median, percentile_10, percentile_90 = diversity_function(
            output_directory_union, output_directory_solver, json_directory1, dwave_directory, h5_directory,
            minimum_path, union, instance_path, m, n)
    
    return solver, median, percentile_10, percentile_90

if __name__ == '__main__':
    solvers = {
        'TN_n1': (get_diversity_tn, json_directory, n_tn1),
        'TN_n2': (get_diversity_tn, json_directory, n_tn2),
        'TN_n4': (get_diversity_tn, json_directory, n_tn4),
        'TN_n8': (get_diversity_tn, json_directory, n_tn8),
        'SBM10': (get_diversity_sbm, sbm_directory, n_dw1),
        'SBM100': (get_diversity_sbm, sbm_directory, n_dw2),
        'SBM1000': (get_diversity_sbm, sbm_directory, n_dw3),
        'SBM10000': (get_diversity_sbm, sbm_directory, n_dw4),
        'DWat2000_n10': (get_diversity_dw, dwave_directory, n_dw1),
        'DWat2000_n100': (get_diversity_dw, dwave_directory, n_dw2),
        'DWat2000_n1000': (get_diversity_dw, dwave_directory, n_dw3),
        'DWat2000_n10000': (get_diversity_dw, dwave_directory, n_dw4),
    }

    results = {}
    output_file = f"embedded_tile_planting_simplified/time_to_diversity/{TOPOLOGY}_L{L}/{TOPOLOGY}_L{L}_ar025.txt"

    # Prepare arguments for each solver
    solver_args = [(solver, diversity_function, n, output_directory_union, output_directory_solver, dir,
                    dwave_directory, sbm_directory, minimum_path, union, instance_path, m)
                   for solver, (diversity_function, dir, n) in solvers.items()]

    # Create a pool of workers and execute the solvers in parallel
    with Pool(processes=min(cpu_count(), len(solvers))) as pool:
        solver_results = pool.starmap(execute_solver, solver_args)

    # Write results to the output file
    with open(output_file, "w") as f:
        for solver, median, percentile_10, percentile_90 in solver_results:
            results[solver] = (median, percentile_10, percentile_90)
            f.write(f"{solver}, {median}, {percentile_10}, {percentile_90}\n")
