import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define the folder where the CSV files are stored
# Instance characteristics
TOPOLOGY = "pegasus"
INSTANCE_SYMBOL = "P8"
INSTANCE_TYPE = "RCO"
TOPOLOGY_SIZE = 8
BETA = 0.5
HAMMING = 147 #147 #27
AR = 0.005
TR= "2^16"

script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
root = os.path.dirname(script_dir)
# results_folder = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "results_beta05")
results_folder = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "results_beta05_solver_ar005_truncated2^16")

max_counts = []
file_labels = []
file_name = []
# colors = ['purple', 'blue', 'green', 'lightcoral', 'magenta', 'teal', 'orange', 'blueviolet', 'brown', 'cyan', 'royalblue', 'red']
# colors = ['purple', 'blue', 'green', 'lightcoral', 'magenta', 'teal', 'orange', 'blueviolet', 'brown', 'cyan', 'red']
# colors = ['purple', 'blue', 'green', 'lightcoral', 'magenta', 'teal', 'cyan', 'royalblue', 'red']
colors = ['blue', 'green', 'red']

# markers = ['o', '*', 'v', '^', 'd', '<', '>', '1', '2', '3', '4', 'x']
# markers = ['o', '*', 'v', '^', 'd', '<', '>', '1', '2', '3', 'x']
# markers = ['o', '*', 'v', '^', 'd', '<', '2', '3', 'x']
markers = ['1', '2', 'x']
# markers = ['<', '1', '2', '3', '4', 'x']
# sizes = [50, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100]
sizes = [300, 300, 200, 100, 100, 100, 100, 200, 200, 200, 200, 200]

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
name = [f"{i:03d}" for i in range(1, 21)]

for i, (label, df) in enumerate(data_dict.items()):
    plt.scatter(name, df['Count'], marker=markers[i % len(markers)], color = colors[i % len(colors)], 
                s = sizes[i % len(sizes)],label=label)
    
plt.yscale('symlog')
plt.ylim(bottom=0)

plt.xlabel('Instance Index')
plt.ylabel('Counts')
plt.title(f"{INSTANCE_SYMBOL}, {INSTANCE_TYPE}, beta={BETA}, Hamming={HAMMING}, approx ratio={AR}, truncation={TR}")
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.show()
