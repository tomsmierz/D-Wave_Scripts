import os
import json
import csv
import h5py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
CUTOFF_ENERGY = 6 #10 CBFMP, 6 RAU
CUTOFF_HAMMING = 27
ITERATIONS = 25
APPROX_RATIO = 1e-2
BETA = 0.5
ENG = 6 #10 CBFMP, 6 RAU
BD = 4
BD1 = 8 
BD2 = 12
MAX_STATES = 64
MAX_STATES1 = 256
MAX_STATES2 = 1024

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
dwave_directory = os.path.join(root, "energies", f"{TOPOLOGY}_random_aggregated", INSTANCE_SYMBOL, INSTANCE_TYPE)
h5_directory = os.path.join(root, "energies", "sbm", f"{TOPOLOGY}_random", INSTANCE_SYMBOL, INSTANCE_TYPE,
                            "SpinGlass", "tmp")
instance_path = os.path.join(root, "instances", f"{TOPOLOGY}_random", INSTANCE_SYMBOL, INSTANCE_TYPE)
output_directory = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "results")


def lowest_tn(folder_path):
    lowest_energy = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                instance = data['columns'][data['colindex']['lookup']['instance']-1][0].split('_')[0]
                energy = data['columns'][data['colindex']['lookup']['energy']-1][0]
                if instance in lowest_energy:
                    if energy < lowest_energy[instance]:
                        lowest_energy[instance] = energy
                else:
                    lowest_energy[instance] = energy
    return lowest_energy


def lowest_dw(folder_path):
    lowest_energy = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            instance = filename.split('_')[0] 
            with open(file_path, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    try:
                        energy = float(row[-2])  # Assuming the energy is in the penultimate column
                        if instance in lowest_energy:
                            if energy < lowest_energy[instance]:
                                lowest_energy[instance] = energy
                        else:
                            lowest_energy[instance] = energy
                    except (ValueError, IndexError):
                        # Handle cases where energy value is not a valid float or the row lacks the expected number of columns
                        pass
    return lowest_energy


def lowest_sbm(folder_path):
    lowest_energy = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.h5'):
            f = h5py.File(filename, "r")
            instance = filename.split("_")[0]
            energy = f['Spectrum']['energies'][0]
            lowest_energy[instance] = energy
    return lowest_energy


def find_lowest_energy(instances_tn, instances_dw, instances_sbm):
    data_tn = lowest_tn(instances_tn)
    data_dw = lowest_dw(instances_dw)
    data_sbm = lowest_dw(instances_sbm)

    lowest_energy = {} 

    all_instances_data = {**data_tn, **data_dw, **data_sbm}

    for instance, energy in all_instances_data.items():
        if instance in data_tn.keys():
            if instance in lowest_energy:
                if energy < lowest_energy[instance]:
                    lowest_energy[instance] = energy
            else:
                lowest_energy[instance] = energy

    return lowest_energy, data_tn

def plot_instance_energy(instances_tn, instances_dw, instances_sbm):
    lowest_energy, data_tn = find_lowest_energy(instances_tn, instances_dw, instances_sbm)

    # Sort the dictionaries by instance name
    sorted_lowest_energy = dict(sorted(lowest_energy.items()))
    sorted_data_tn = dict(sorted(data_tn.items()))

    instances = list(sorted_lowest_energy.keys())
    energies_best = list(sorted_lowest_energy.values())
    energies_tn = list(sorted_data_tn.values())


    # Calculate the normalized differences
    normalized_differences = [
        (data_tn - best_energy) / abs(best_energy) if best_energy is not None else None
        for best_energy, data_tn in zip(energies_best, energies_tn)
    ]
    
    print(normalized_differences)
    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='red', linestyle='--')  # Add a red horizontal line at y=0
    plt.plot(instances, normalized_differences, 'bo')
    plt.xlabel("Instance Name")
    plt.ylabel("(Energies_TN - Energies_best) / |Energies_best|")
    plt.title("Normalized Energy Differences")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    
if __name__ == '__main__':
    plot_instance_energy(json_directory, dwave_directory, h5_directory)