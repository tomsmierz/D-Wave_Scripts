import os
import random
import h5py
import numpy as np
import csv
import json
import matplotlib.pyplot as plt

# Constants
BETA = 0.5
m = 200  
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
tn_directory = os.path.join(root, "embedded_tile_planting_simplified", "tn_results", f"{TOPOLOGY}_adv6.4_results", "new", f"{TOPOLOGY}_L{L}_droplets")
best_file = os.path.join(root, "embedded_tile_planting_simplified", "best", f"{TOPOLOGY}_adv6.4_results", f"best_{TOPOLOGY}_2D_L_{L}.csv")

def read_energy_from_h5(directory):
    instance_energies = {}
    for filename in os.listdir(directory):
        if filename.endswith('.h5'):
            file_path = os.path.join(directory, filename)
            instance = filename.split('_')[0]
            with h5py.File(file_path, 'r') as file:
                energies = file['Spectrum']['energies'][:]
                instance_energies[instance] = list(energies)
    return instance_energies


def read_energy_from_csv(directory):
    instance_energies = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            # instance = filename.split('_P')[0]
            instance = "_".join(filename.split('_')[2:]).replace('.csv', '')
            with open(file_path, 'r', newline='') as file:
                csv_reader = csv.DictReader(file)                
                if 'energy' not in csv_reader.fieldnames:
                    print(f"Warning: No 'energy' column in {filename}")
                    continue
                for row in csv_reader:
                    try:
                        energy = float(row['energy'])
                        if instance not in instance_energies:
                            instance_energies[instance] = []
                        instance_energies[instance].append(energy)
                    except (ValueError, KeyError):
                        pass 
    return instance_energies

def read_energy_from_json(directory):
    instance_energies = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                instance = data['columns'][data['colindex']['lookup']['instance']-1][0].split('_P')[0]
                energy = data['columns'][data['colindex']['lookup']['energy']-1][0]
                if instance not in instance_energies:
                    instance_energies[instance] = []
                instance_energies[instance].append(energy)
    return instance_energies

import csv

def read_best_energy(best_file):
    best_energies = {}
    
    with open(best_file, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        
        for row in csv_reader:
            if len(row) < 2:
                continue
            try:
                instance = row[0]
                energy = float(row[1])
                best_energies[instance] = energy
            except ValueError:
                pass 

    return best_energies

def get_relative_best_energy_dw_sbm(instance_energies, best_energies, m, n):
    median_samples = []
    
    for _ in range(100):
        medians = []
        for _ in range(m):
            sampled_instance = random.choice(list(instance_energies.keys())) # losowanie z powtórzeniami
            if sampled_instance in best_energies:
                best_energy = best_energies[sampled_instance]
                if n == "all":
                    sampled_energies = instance_energies[sampled_instance]
                else:
                    sampled_energies = random.choices(instance_energies[sampled_instance], k=n) # losowanie z powtórzeniami
                best_energy_sample = min(sampled_energies)
                relative_energy = (best_energy_sample - best_energy) / (2 * abs(best_energy))
                medians.append(relative_energy)
        
        if medians:
            median = np.median(medians)
            # median = np.percentile(medians, 90)
            median_samples.append(median)
    
    if median_samples:
        median = np.median(median_samples)
        percentile_10 = np.percentile(median_samples, 2.5)
        percentile_90 = np.percentile(median_samples, 97.5)
        # median = np.percentile(median_samples, 90)
        # percentile_10 = np.percentile(median_samples, 2.5)
        # percentile_90 = np.percentile(median_samples, 97.5)
        return median, percentile_10, percentile_90
    else:
        return None, None, None
    

def get_relative_best_energy_tn(instance_energies, best_energies, m, n):
    median_samples = []
    
    for _ in range(100):
        medians = []
        for _ in range(m):
            sampled_instance = random.choice(list(instance_energies.keys()))  # Losowanie instancji z powtórzeniami
            if sampled_instance in best_energies:
                best_energy = best_energies[sampled_instance]
                sampled_energies = random.sample(instance_energies[sampled_instance], k=n)
                best_energy_sample = min(sampled_energies)
                relative_energy = (best_energy_sample - best_energy) / (2 * abs(best_energy))
                medians.append(relative_energy)
        
        
        if medians:
            median = np.median(medians)
            # median = np.percentile(medians, 90)
            median_samples.append(median)
    
    if median_samples:
        median = np.median(median_samples)
        percentile_10 = np.percentile(median_samples, 2.5)
        percentile_90 = np.percentile(median_samples, 97.5)
        # median = np.percentile(median_samples, 90)
        # percentile_10 = np.percentile(median_samples, 2.5)
        # percentile_90 = np.percentile(median_samples, 97.5)
        return median, percentile_10, percentile_90
    else:
        return None, None, None
    

if __name__ == '__main__':
    
    solvers = {
        'TN1': (read_energy_from_json, tn_directory, get_relative_best_energy_tn, n_tn1),
        'TN2': (read_energy_from_json, tn_directory, get_relative_best_energy_tn, n_tn2),
        'TN4': (read_energy_from_json, tn_directory, get_relative_best_energy_tn, n_tn4),
        'TN8': (read_energy_from_json, tn_directory, get_relative_best_energy_tn, n_tn8),
        'SBM10': (read_energy_from_csv, sbm_directory, get_relative_best_energy_dw_sbm, n_dw1),
        'SBM100': (read_energy_from_csv, sbm_directory, get_relative_best_energy_dw_sbm, n_dw2),
        'SBM1000': (read_energy_from_csv, sbm_directory, get_relative_best_energy_dw_sbm, n_dw3),
        'SBM10000': (read_energy_from_csv, sbm_directory, get_relative_best_energy_dw_sbm, n_dw4),
        'SBMall': (read_energy_from_csv, sbm_directory, get_relative_best_energy_dw_sbm, "all"),
        'DW10': (read_energy_from_csv, dwave_directory, get_relative_best_energy_dw_sbm, n_dw1),
        'DW100': (read_energy_from_csv, dwave_directory, get_relative_best_energy_dw_sbm, n_dw2),
        'DW1000': (read_energy_from_csv, dwave_directory, get_relative_best_energy_dw_sbm, n_dw3),
        'DW10000': (read_energy_from_csv, dwave_directory, get_relative_best_energy_dw_sbm, n_dw4),
        'DWall': (read_energy_from_csv, dwave_directory, get_relative_best_energy_dw_sbm, "all"),
    }

    results = {}
    output_file = f"embedded_tile_planting_simplified/time_to_energy/{TOPOLOGY}_L{L}.txt"

    with open(output_file, "w") as f:
        for solver, (read_function, directory, rel_eng_function, n) in solvers.items():
            instance_energies = read_function(directory)
            best_energies = read_best_energy(best_file)
            median, percentile_10, percentile_90 = rel_eng_function(instance_energies, best_energies, m, n)
            results[solver] = (median, percentile_10, percentile_90)
            f.write(f"{solver}, {median}, {percentile_10}, {percentile_90}\n")

