import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define the folder where the CSV files are stored
# Instance characteristics
TOPOLOGY = "pegasus"
INSTANCE_SYMBOL = "P4"
INSTANCE_TYPE = "CBFM-P"
TOPOLOGY_SIZE = 4
BETA = 0.5
HAMMING = 27 #147 #27
AR = 0.01

script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
root = os.path.dirname(script_dir)
# results_folder = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "results_beta05")
results_folder = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "results_beta05_solver")

max_counts = []
file_labels = []
file_name = []

colors = ['b', 'g', 'r', 'c', 'm', 'k', 'b', 'g', 'r', 'c', 'm', 'k']
# markers = ['o', 's', 'D', '^', 'v', '<', '>', '1', '2', '3', '4']
markers = ['o', '*', 'v', '^', 'd', '<', '>', '1', '2', '3', '4', 'x']
# markers = ['<', '1', '2', '3', '4', 'x']
# sizes = [50, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100]
sizes = [100, 100, 100, 100, 100, 100, 100, 200, 200, 200, 200, 200]

param_to_color = {}
param_to_marker = {}
param_to_size = {}

data_dict = {}

for filename in sorted(os.listdir(results_folder)):
    if filename.endswith(".csv"):
        file_path = os.path.join(results_folder, filename)
        df = pd.read_csv(file_path)
        label = filename  
        data_dict[label] = df

plt.figure(figsize=(12, 6))

for i, (label, df) in enumerate(data_dict.items()):
    plt.scatter(df['Name'], df['Count'], marker=markers[i % len(markers)], color = colors[i % len(colors)], s = sizes[i % len(sizes)],label=label)

plt.xlabel('Instance Index')
plt.ylabel('Counts')
plt.title('Counts in union of solvers')
plt.xticks(rotation=90)  
plt.legend()

plt.tight_layout()
plt.show()
