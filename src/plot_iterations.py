import os
import pandas as pd
import matplotlib.pyplot as plt

# Instance characteristic
TOPOLOGY = "pegasus"
INSTANCE_SYMBOL = "P8"
INSTANCE_TYPE = "CBFM-P"
TOPOLOGY_SIZE = 8
script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
root = os.path.dirname(script_dir)
results_folder = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "results_beta05")

dataframes = []

for filename in os.listdir(results_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(results_folder, filename)
        df = pd.read_csv(file_path)
        dataframes.append((df, filename))  

for df, filename in dataframes:
    plt.figure(figsize=(8, 6))
    plt.plot(df['Iterations'], df['Count'], "bo")
    plt.title(f'Counts vs. Iterations - {filename}')
    plt.xlabel('Iterations')
    plt.ylabel('Counts')
    plt.grid()
    plt.savefig(f'{filename}.png')  
    plt.show()
