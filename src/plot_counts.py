import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define the folder where the CSV files are stored
# Instance characteristics
TOPOLOGY = "pegasus"
INSTANCE_SYMBOL = "P8"
INSTANCE_TYPE = "CBFM-P"
TOPOLOGY_SIZE = 8
BETA = 0.5
HAMMING = 147 #27
AR = 0.001

script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
root = os.path.dirname(script_dir)
results_folder = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "results_beta05")

max_counts = []
file_labels = []
file_name = []

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c']
# markers = ['o', 's', 'D', '^', 'v', '<', '>', '1', '2', '3', '4']
markers = ['o', '*', 'v', '^', 'd', '<', '1', '2', '3', '4', 'x']
sizes = [50, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100]
param_to_color = {}
param_to_marker = {}
param_to_size = {}

file_info = []  # Store filename, instance parameters, color, marker, and max count
for i, filename in enumerate(sorted(os.listdir(results_folder))):  # Sort files by instance names
    if filename.endswith(".csv"):
        file_path = os.path.join(results_folder, filename)
        df = pd.read_csv(file_path)
        max_count = df['Count'].max()  
        max_counts.append(max_count)

        instance_parameters = filename.split('_')[2:5]
        name = filename.split('_')[0]

        file_name.append(name)
        file_labels.append(instance_parameters)

        if tuple(instance_parameters) not in param_to_color:
            color = colors[len(param_to_color) % len(colors)]
            marker = markers[len(param_to_marker) % len(markers)]
            size = sizes[len(param_to_size) % len(sizes)]
            param_to_color[tuple(instance_parameters)] = color
            param_to_marker[tuple(instance_parameters)] = marker
            param_to_size[tuple(instance_parameters)] = size

        color = param_to_color[tuple(instance_parameters)]
        marker = param_to_marker[tuple(instance_parameters)]
        size = param_to_size[tuple(instance_parameters)]

        file_info.append((name, instance_parameters, color, marker, max_count, size))

legend_handles = []
for param_set, color in param_to_color.items():
    marker = param_to_marker[param_set]
    legend_handles.append(Line2D([0], [0], color=color, marker=marker, label=param_set))

plt.figure(figsize=(12, 6))  # Set the figure size

for info in file_info:
    plt.scatter(info[0], info[4], marker=info[3], color=info[2], label=info[1], s=info[5])

plt.legend(handles=legend_handles, title="Instance Parameters", numpoints=1)

plt.xlabel('Instance index')
plt.ylabel('Independent Droplets')
plt.title(f"{INSTANCE_SYMBOL}, {INSTANCE_TYPE}, beta={BETA}, Hamming={HAMMING}, approx ratio={AR}")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
