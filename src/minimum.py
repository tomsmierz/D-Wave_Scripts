import copy
import os
import json
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple
from renumeration import advantage_6_1_to_spinglass, advantage_6_1_to_spinglass_int
from droplets import array_from_dict
from scipy.spatial.distance import hamming
from typing import Optional, Union
from tqdm import tqdm

# Instance characteristic
TOPOLOGY = "pegasus"
INSTANCE_SYMBOL = "P8"
INSTANCE_TYPE = "CBFM-P"
TOPOLOGY_SIZE = 8

# Directories
cwd = os.getcwd()
json_directory = os.path.join(cwd, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, f"{INSTANCE_SYMBOL}_droplets_i1-2")
dwave_directory = os.path.join(cwd, "energies", f"{TOPOLOGY}_random_aggregated", INSTANCE_SYMBOL, INSTANCE_TYPE)
output_csv = os.path.join(cwd, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "minimum.csv")

def read_json_data(directory) -> dict:
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

def read_csv_data(data: pd.DataFrame, instance_size: int) -> namedtuple:
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    states_dwave = copy.deepcopy(data)
    states_dwave.drop(["energy", "num_occurrences", "annealing_time", "num_reads", "pause_time", "reverse"], axis=1, inplace=True)
    energies_dwave = data["energy"].to_numpy()

    renum = {k: advantage_6_1_to_spinglass_int(int(k), instance_size) - 1 for k in states_dwave.columns}
    states_dwave.rename(columns=renum, inplace=True)

    states_dwave_reordered = states_dwave[sorted(list(states_dwave.columns))]
    states_dwave_reordered = states_dwave_reordered.to_numpy()

    state_energy_tuple = StateEnergy(states_dwave_reordered, energies_dwave)

    return state_energy_tuple

def calculate_and_create_dataframe(dwave_path, spinglass_path, output_csv):
    spinglass_states = read_json_data(spinglass_path)
    result_data = []
    
    for name, state_energy_net in tqdm(spinglass_states.items()):
        # print("instance: ", name)
        instance_df = pd.read_csv(os.path.join(dwave_path, f"{name}.csv"),
                                  index_col=0)
        state_energy_tuple = read_csv_data(instance_df, TOPOLOGY_SIZE)
        ground_eng_dw = state_energy_tuple.energy.min()
        ground_eng_sg = state_energy_net.energy.min()
        ground_eng = min(ground_eng_dw, ground_eng_sg)
        # Create a DataFrame with instance index and minimum value
        result_data.append({'Index': name, 'Ground energy': ground_eng})

    # Create a DataFrame with instance index and minimum value
    df = pd.DataFrame(result_data)
    df_sorted = df.sort_values(by="Index")
    # Write the DataFrame to a CSV file
    df_sorted.to_csv(output_csv, index=False)
    return df


if __name__ == '__main__':

    df = calculate_and_create_dataframe(dwave_directory, json_directory, output_csv)
