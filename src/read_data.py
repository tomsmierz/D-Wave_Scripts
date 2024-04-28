import copy
import os
import json
import random
import time
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

vector = Union[np.ndarray, list]


def get_state_energy_from_dwave(data: pd.DataFrame, cutoff_energy: float,
                                instance_size: int, ground_eng: float, is_Z2: bool = False) -> namedtuple:
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    states_dwave = copy.deepcopy(data)
    if instance_size == 16:
        states_dwave.drop(["energy"], axis=1, inplace=True)
    else:
        states_dwave.drop(["energy", "num_occurrences", "annealing_time", "num_reads", "pause_time", "reverse"],
                      axis=1, inplace=True)
    
    energies_dwave = data["energy"].to_numpy()

    renum = {k: advantage_6_1_to_spinglass_int(int(k), instance_size) for k in states_dwave.columns}
    states_dwave.rename(columns=renum, inplace=True)

    states_dwave_reordered = states_dwave[sorted(list(states_dwave.columns))]
    states_dwave_reordered = states_dwave_reordered.to_numpy()

    filtered_states, filtered_energies = filter_states_by_energy(states_dwave_reordered, energies_dwave,
                                                                 cutoff_energy, ground_eng, is_Z2)
    filtered_state_energy_tuple = StateEnergy(filtered_states, filtered_energies)

    return filtered_state_energy_tuple


def get_state_energy_from_dwave_zephyr(data: pd.DataFrame, cutoff_energy: float,
                                instance_size: int, ground_eng: float, is_Z2: bool = False) -> namedtuple:
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    states_dwave = copy.deepcopy(data)
    states_dwave.drop(["energy", "num_occurrences", "annealing_time", "num_reads", "pause_time", "reverse"],
                      axis=1, inplace=True)
    energies_dwave = data["energy"].to_numpy()

    states_dwave_reordered = states_dwave[sorted(list(states_dwave.columns))]
    states_dwave_reordered = states_dwave_reordered.to_numpy()

    filtered_states, filtered_energies = filter_states_by_energy(states_dwave_reordered, energies_dwave,
                                                                 cutoff_energy, ground_eng, is_Z2)
    filtered_state_energy_tuple = StateEnergy(filtered_states, filtered_energies)

    return filtered_state_energy_tuple
import numpy as np

def filter_states_by_energy(states: np.ndarray, energies: np.ndarray, cutoff_energy: float, ground_eng: float, is_Z2: bool = False) -> (np.ndarray, np.ndarray):
    """
    states: Expected to be square matrix, with states in rows
    """
    if cutoff_energy == 0:
        return states, energies
    else:
        mask = energies <= ground_eng + cutoff_energy
        indices = np.where(mask)[0]
        filtered_states = states[indices]
        filtered_energies = energies[indices]

        if is_Z2:
            for state_index in range(filtered_states.shape[0]):
                state = filtered_states[state_index]
                if state[0] != 1:
                    filtered_states[state_index] *= -1
            unique_states, unique_indices = np.unique(filtered_states, axis=0, return_index=True)
            unique_energies = filtered_energies[unique_indices]
            return unique_states, unique_energies
        else:
            return filtered_states, filtered_energies

# def filter_states_by_energy(states: np.ndarray, energies: np.ndarray, cutoff_energy: float, ground_eng: float) -> (np.ndarray, np.ndarray):
#     """
#     states: Expected to be square matrix, with states in rows
#     """
#     if cutoff_energy == 0:
#         return states, energies
#     else:
#         mask = energies <= ground_eng + cutoff_energy
#         indices = np.where(mask)[0]
#         filtered_states = states[indices]
#         filtered_energies = energies[indices]
#         for state_index in range(filtered_states.shape[0]):
#             state = filtered_states[state_index]
#             if state[0] != 1:
#                 filtered_states[state_index] *= -1
#         unique_states, unique_indices = np.unique(filtered_states, axis=0, return_index=True)
#         unique_energies = filtered_energies[unique_indices]
#         return unique_states, unique_energies
#         # return filtered_states, filtered_energies


# def filter_states_by_energy(states: np.ndarray, energies: np.ndarray, cutoff_energy: float, ground_eng: float) -> (np.ndarray, np.ndarray):
#     """
#     states: Expected to be square matrix, with states in rows
#     """
#     if cutoff_energy == 0:
#         return states, energies
#     else:
#         mask = energies <= ground_eng + cutoff_energy
#         indices = np.where(mask)[0]
#         filtered_states = states[indices]
#         filtered_energies = energies[indices]
#         return filtered_states, filtered_energies

def array_from_dict(dict_list):
    num_states = len(dict_list)
    # max_index = max(int(key) for d in dict_list for key in d.keys())
    max_index = len(dict_list[0])
    result_array = np.zeros((num_states, max_index))
    for i, d in enumerate(dict_list):
        for j, (key, value) in enumerate(d.items()):
            result_array[i, j] = value
    return result_array

def read_h5_files(directory, df_min, approx_ratio, inst, is_Z2: bool = False):
    instance_data = {}
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
        if os.path.isfile(file):
            file_extension = os.path.splitext(filename)[-1].lower()
            if file_extension == ".h5":
                f = h5py.File(file, "r")
                instance_name = filename.split(".")[0]
                name_range = [f"{i:03d}" for i in range(1, inst+1)]
                # name_range = [f"{inst:03d}",]

                if instance_name not in name_range:
                    continue
                energies = f['Spectrum']['energies']
                states = f['Spectrum']["states"]
                ground_eng = df_min[df_min.index == instance_name]['Energy'].values[0]
                energy_cutoff = approx_ratio * 2 * np.abs(ground_eng)
                filtered_states, filtered_energies = filter_states_by_energy(states, energies,
                                                                        energy_cutoff, ground_eng, is_Z2)
                instance_data[instance_name] = StateEnergy(filtered_states, filtered_energies)
                # instance_data[instance_name] = StateEnergy(states, energies)
    return instance_data


def read_json_files(directory, beta, eng, bd, ms, df_min, approx_ratio, is_Z2: bool = False) -> dict:
    instance_data = {}
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
        if os.path.isfile(file) and file.endswith(".json"):
            with open(file, encoding='utf-8') as f:
                json_data = json.load(f)
                if json_data['columns'][json_data['colindex']['lookup']['β']-1][0] == beta and \
                        json_data['columns'][json_data['colindex']['lookup']['bond_dim']-1][0] == bd and \
                        json_data['columns'][json_data['colindex']['lookup']['max_states']-1][0] == ms:
                        # json_data['columns'][json_data['colindex']['lookup']['eng']-1][0] == eng and \
                    instance_name = json_data['columns'][json_data['colindex']['lookup']['instance']-1][0].split('_')[0]
                    energy_data = json_data['columns'][json_data['colindex']['lookup']['drop_eng']-1][0]
                    state_data = json_data['columns'][json_data['colindex']['lookup']['ig_states']-1][0]
                    state_data_np = array_from_dict(state_data)
                    energy_data_np = np.array(energy_data)
                    ground_eng = df_min[df_min.index == instance_name]['Energy'].values[0]
                    energy_cutoff = approx_ratio * 2 * np.abs(ground_eng)
                    filtered_states, filtered_energies = filter_states_by_energy(state_data_np, energy_data_np, energy_cutoff, ground_eng, is_Z2)

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


def read_all_json_files(directory, beta, approx_ratio, df_min, is_Z2: bool = False) -> dict:
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
                    cutoff_energy = approx_ratio * 2 * np.abs(ground_eng)
                    filtered_states, filtered_energies = filter_states_by_energy(state_data_np, energy_data_np, cutoff_energy, ground_eng, is_Z2)

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


def read_all_csv_files(file_path, df_min) -> dict:
    instance_data = {}
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])

    if os.path.isfile(file_path) and file_path.endswith(".csv"):
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                instance_name = row[1].split('.')[0]
                energy_value = float(row[2])
                state_data = np.array([int(x) for x in row[3].split(',')])
                    
                if instance_name not in instance_data:
                    instance_data[instance_name] = StateEnergy([], [])
                    
                instance_data[instance_name].state.append(state_data)
                instance_data[instance_name].energy.append(energy_value)
    return instance_data
