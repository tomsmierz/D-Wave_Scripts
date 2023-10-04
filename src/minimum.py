import copy
import os
import json
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple
from renumeration import advantage_6_1_to_spinglass, advantage_6_1_to_spinglass_int
from droplets import read_json_files, get_state_energy_from_dwave
from scipy.spatial.distance import hamming
from typing import Optional, Union
from tqdm import tqdm

# Constants
CUTOFF_ENERGY = 2.01

# Instance characteristic
TOPOLOGY = "pegasus"
INSTANCE_SYMBOL = "P4"
INSTANCE_TYPE = "RCO"
TOPOLOGY_SIZE = 4

# Directories
cwd = os.getcwd()
json_directory = os.path.join(cwd, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, f"{INSTANCE_SYMBOL}_droplets_new")
dwave_directory = os.path.join(cwd, "energies", f"{TOPOLOGY}_random", f"{TOPOLOGY}_random", INSTANCE_SYMBOL, INSTANCE_TYPE)
output_csv = os.path.join(cwd, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "minimum.csv")

def calculate_and_create_dataframe(dwave_path, spinglass_path, output_csv):
    spinglass_states = read_json_files(spinglass_path)
    result_data = []
    
    for name, state_energy_net in tqdm(spinglass_states.items()):
        # print("instance: ", name)
        instance_df = pd.read_csv(os.path.join(dwave_path, f"{name}.csv"),
                                  index_col=0)
        state_energy_tuple = get_state_energy_from_dwave(instance_df, CUTOFF_ENERGY, TOPOLOGY_SIZE)
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
