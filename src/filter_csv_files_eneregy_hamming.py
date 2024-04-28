import os
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from renumeration import advantage_6_1_to_spinglass, advantage_6_1_to_spinglass_int
from read_data import get_state_energy_from_dwave

TOPOLOGY = "pegasus"
INSTANCE_SYMBOL = "P8"
INSTANCE_TYPE = "RCO"
TOPOLOGY_SIZE = 8
APPROX_RATIO = 0.005
HAMMING_CUTOFF = 100
is_Z2 = True
script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
root = os.path.dirname(script_dir)
dwave_directory = os.path.join(root, "energies", f"{TOPOLOGY}_random_aggregated", INSTANCE_SYMBOL, INSTANCE_TYPE)
minimum_path = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "minimum_truncated2^16.csv")
output_directory = os.path.join(root, "energies", f"{TOPOLOGY}_random_aggregated", INSTANCE_SYMBOL, INSTANCE_TYPE, "filtered_files_hd100_i020")


def hamming_distance(state1, state2):
    return np.sum(state1 != state2)

def add_state_to_dataframe(states, state2, unique_states_df, row):
    h_list = [hamming_distance(st, state2) for st in states]
        
    if all(distance > HAMMING_CUTOFF for distance in h_list):
        unique_states_df = pd.concat([unique_states_df, row], ignore_index=True)
        states.append(state2)
            
    return unique_states_df, states

# Load instance data
min_values_df = pd.read_csv(minimum_path, index_col=0)
min_values_df.index = min_values_df.index.map(lambda x: str(x).zfill(3))
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Folder '{output_directory}' created successfully.")

for filename in tqdm(os.listdir(dwave_directory)):
    file = os.path.join(dwave_directory, filename)
    name = filename.split(".")[0]
    # if name not in ["001",]:
    name_range = [f"{i:03d}" for i in range(1, 21)]
    if name not in name_range:
        continue
    if os.path.isfile(file):
        instance_df = pd.read_csv(file, index_col=0)
        
        # Sort states based on energy
        instance_df = instance_df.sort_values(by='energy')
        ground_eng = min_values_df[min_values_df.index == name]['Energy'].values[0]
        filtered_states_eng = instance_df[instance_df['energy'] <= ground_eng + 2 * APPROX_RATIO * np.abs(ground_eng)]
        steng, _ = get_state_energy_from_dwave(filtered_states_eng, 0, TOPOLOGY_SIZE, ground_eng, is_Z2)

        unique_states_df = pd.DataFrame(columns=instance_df.columns)
        steng_filtered = []

        for i, row in tqdm(enumerate(steng)):

            if unique_states_df.empty:
                unique_states_df = pd.concat([unique_states_df, instance_df.iloc[[i]]], ignore_index=True)
                steng_filtered.append(row)
            else:
                unique_states_df, steng_filtered = add_state_to_dataframe(steng_filtered, row, unique_states_df, instance_df.iloc[[i]])

        output_file = os.path.join(output_directory, f'{name}.csv')
        unique_states_df.to_csv(output_file, index=True)
