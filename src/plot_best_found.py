import os
import json
import csv
import h5py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Instance characteristic
TOPOLOGY = "zephyr"
INSTANCE_SYMBOL = "Z4"
INSTANCE_TYPE = "RAU"
TOPOLOGY_SIZE = 4

# Directories
script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
root = os.path.dirname(script_dir)
json_directory = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "droplets", "final_bench_truncate2^12")
json_directory2 = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "droplets", "final_bench_truncate2^14")
json_directory3 = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "droplets", "final_bench")
dwave_directory = os.path.join(root, "energies", f"{TOPOLOGY}_random_aggregated", INSTANCE_SYMBOL, INSTANCE_TYPE)
h5_directory = os.path.join(root, "energies", "sbm", f"{TOPOLOGY}_random", INSTANCE_SYMBOL, INSTANCE_TYPE)
                            #  "SpinGlass", "tmp")
output_csv = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "best_all.csv")
output_csv_tn12 = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "best_tn12.csv")
output_csv_tn14 = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "best_tn14.csv")
output_csv_tn = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "best_tn.csv")
output_csv_dwave = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "best_dwave.csv")
output_csv_sbm = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "best_sbm.csv")


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
            instance = filename.split('.')[0]
            with open(file_path, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    try:
                        energy = float(row[-6]) #TODO: -1 for P16, -6 for the rest
                        if instance in lowest_energy:
                            if energy < lowest_energy[instance]:
                                lowest_energy[instance] = energy
                        else:
                            lowest_energy[instance] = energy
                    except (ValueError, IndexError):
                        pass
    return lowest_energy


def lowest_sbm(folder_path):
    lowest_energy = {}
    for filename in os.listdir(folder_path):
        file = os.path.join(folder_path, filename)
        if os.path.isfile(file):
            file_extension = os.path.splitext(filename)[-1].lower()
            if file_extension == ".h5":
                f = h5py.File(file, "r")
                instance_name = filename.split(".")[0]
                instance = filename.split("_")[0]
                energy = f['Spectrum']['energies'][0]
                lowest_energy[instance] = energy
    return lowest_energy


def find_lowest_energy(instances_tn12, instances_tn14, instances_tn, instances_dw, instances_sbm, output_csv):
    data_tn12 = lowest_tn(instances_tn12)
    data_tn14 = lowest_tn(instances_tn14)
    data_tn = lowest_tn(instances_tn)
    data_dw = lowest_dw(instances_dw)
    data_sbm = lowest_sbm(instances_sbm)
    # name_range = [f"{i:03d}" for i in range(1, 21)]
    lowest_energy = {} 
    
    all_instances_data = {}
    all_keys = set(data_tn.keys()).union(data_tn12.keys(), data_tn14.keys(), data_dw.keys(), data_sbm.keys())
    # all_keys = set(data_tn.keys()).union(data_dw.keys())

    for key in all_keys:
        # values_list = [data_tn.get(key, None), data_dw.get(key, None)]
        values_list = [data_tn.get(key, None), data_dw.get(key, None), data_sbm.get(key, None)]
        all_instances_data[key] = [value for value in values_list if value is not None]
    # print(all_instances_data)
    for instance, energy in all_instances_data.items():
        if instance in data_tn.keys():
            lowest_energy[instance] = min(energy)

    data_list = [[instance, energy] for instance, energy in lowest_energy.items()]
    df = pd.DataFrame(data_list, columns=["Instance", "Energy"])
    df_sorted = df.sort_values(by="Instance")
    df_sorted.to_csv(output_csv, index=False)
    
    return lowest_energy, data_tn12, data_tn14, data_tn, data_sbm, data_dw



def plot_instance_energy(instances_tn, instances_tn2, instances_tn3, instances_dw, instances_sbm, output_csv, output_csv_tn12, output_csv_tn14, output_csv_tn, output_csv_dwave, output_csv_sbm):
    lowest_energy, data_tn12, data_tn14, data_tn, data_sbm, data_dw  = find_lowest_energy(instances_tn, instances_tn2, instances_tn3, instances_dw, instances_sbm, output_csv)

    sorted_lowest_energy = dict(sorted(lowest_energy.items()))
    sorted_data_tn12 = dict(sorted(data_tn12.items()))
    sorted_data_tn14 = dict(sorted(data_tn14.items()))
    sorted_data_tn = dict(sorted(data_tn.items()))
    sorted_data_sbm = dict(sorted(data_sbm.items()))
    sorted_data_dw = dict(sorted(data_dw.items()))

    instances = list(sorted_lowest_energy.keys())
    energies_best = list(sorted_lowest_energy.values())
    energies_tn12 = list(sorted_data_tn12.values())
    energies_tn14 = list(sorted_data_tn14.values())
    energies_tn = list(sorted_data_tn.values())
    energies_sbm = list(sorted_data_sbm.values())
    energies_dwave = list(sorted_data_dw.values())
    print(energies_sbm)
    print(energies_dwave)
    print(energies_tn)
    normalized_differences_tn12 = [
        (data_tn12 - best_energy) / (2*abs(best_energy)) if best_energy is not None else None
        for best_energy, data_tn12 in zip(energies_best, energies_tn12)
    ]
    normalized_differences_tn14 = [
        (data_tn14 - best_energy) / (2*abs(best_energy)) if best_energy is not None else None
        for best_energy, data_tn14 in zip(energies_best, energies_tn14)
    ]    
    normalized_differences_tn = [
        (data_tn - best_energy) / (2*abs(best_energy)) if best_energy is not None else None
        for best_energy, data_tn in zip(energies_best, energies_tn)
    ]
    normalized_differences_sbm = [
        (energies_sbm - best_energy) / (2*abs(best_energy)) if best_energy is not None else None
        for best_energy, energies_sbm in zip(energies_best, energies_sbm)
    ]
    normalized_differences_dw = [
        (energies_dwave - best_energy) / (2*abs(best_energy)) if best_energy is not None else None
        for best_energy, energies_dwave in zip(energies_best, energies_dwave)
    ]

    result_dict = {f"{i+1:03d}": value for i, value in enumerate(normalized_differences_tn12)}
    df = pd.DataFrame(list(result_dict.items()), columns=["Instance", "Energy"])
    df.to_csv(output_csv_tn12, index=False)
    
    result_dict = {f"{i+1:03d}": value for i, value in enumerate(normalized_differences_tn14)}
    df = pd.DataFrame(list(result_dict.items()), columns=["Instance", "Energy"])
    df.to_csv(output_csv_tn14, index=False)
    
    result_dict = {f"{i+1:03d}": value for i, value in enumerate(normalized_differences_tn)}
    df = pd.DataFrame(list(result_dict.items()), columns=["Instance", "Energy"])
    df.to_csv(output_csv_tn, index=False)
    
    result_dict = {f"{i+1:03d}": value for i, value in enumerate(normalized_differences_dw)}
    df = pd.DataFrame(list(result_dict.items()), columns=["Instance", "Energy"])
    df.to_csv(output_csv_dwave, index=False)
    
    result_dict = {f"{i+1:03d}": value for i, value in enumerate(normalized_differences_sbm)}
    df = pd.DataFrame(list(result_dict.items()), columns=["Instance", "Energy"])
    df.to_csv(output_csv_sbm, index=False)
    # plt.figure(figsize=(10, 6))
    # plt.axhline(0, color='red', linestyle='--')  # Add a red horizontal line at y=0
    # plt.plot(instances, normalized_differences, 'bo', label='TN')
    # plt.plot(instances, normalized_differences_sbm, 'ro', label='SBM')
    # # plt.plot(instances, normalized_differences_dw, 'go', label='DW')

    # plt.xlabel("Instance Name")
    # plt.ylabel("(Energies_TN - Energies_best) / (2*|Energies_best|)")
    # plt.title(f"{INSTANCE_SYMBOL}, {INSTANCE_TYPE}")

    # plt.xticks(rotation=45)
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
    
    
if __name__ == '__main__':
    plot_instance_energy(json_directory, json_directory2, json_directory3, dwave_directory, h5_directory, output_csv, output_csv_tn12, output_csv_tn14, output_csv_tn, output_csv_dwave, output_csv_sbm)