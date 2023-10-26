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
HAMMING = 27
AR = 0.01

# Define the folder where the CSV files are stored
script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
root = os.path.dirname(script_dir)
results_folder = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "results")

# Initialize empty lists to store the maximum counts and corresponding labels (file names)
max_counts = []
file_labels = []
file_name = []

# Define a list of colors and markers for differentiation
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c']
markers = ['o', 's', 'D', '^', 'v', '<', '>', '1', '2', '3', '4']

# Create a mapping from parameter sets to colors and markers
param_to_color = {}
param_to_marker = {}

# Loop through CSV files in the folder
file_info = []  # Store filename, instance parameters, color, marker, and max count
for i, filename in enumerate(sorted(os.listdir(results_folder))):  # Sort files by instance names
    if filename.endswith(".csv"):
        file_path = os.path.join(results_folder, filename)
        df = pd.read_csv(file_path)
        max_count = df['Count'].max()  # Find the maximum "Counts" in the file
        max_counts.append(max_count)

        # Extract the instance parameters from the filename
        instance_parameters = filename.split('_')[2:5]
        name = filename.split('_')[0]

        # Append the file name, instance parameters, color, marker, and max count
        file_name.append(name)
        file_labels.append(instance_parameters)

        # If this parameter set is not in the mapping, assign a color and marker
        if tuple(instance_parameters) not in param_to_color:
            color = colors[len(param_to_color) % len(colors)]
            marker = markers[len(param_to_marker) % len(markers)]
            param_to_color[tuple(instance_parameters)] = color
            param_to_marker[tuple(instance_parameters)] = marker

        color = param_to_color[tuple(instance_parameters)]
        marker = param_to_marker[tuple(instance_parameters)]

        # Append data to file_info list
        file_info.append((name, instance_parameters, color, marker, max_count))

# Create a legend to label the data
legend_handles = []
for param_set, color in param_to_color.items():
    marker = param_to_marker[param_set]
    legend_handles.append(Line2D([0], [0], color=color, marker=marker, label=param_set))

plt.figure(figsize=(12, 6))  # Set the figure size

# Plot the data points with correct colors and markers
for info in file_info:
    plt.plot(info[0], info[4], marker=info[3], color=info[2], label=info[1])

plt.legend(handles=legend_handles, title="Instance Parameters", numpoints=1)

# Set x-axis and y-axis labels and title
plt.xlabel('Instance index')
plt.ylabel('Independent Droplets')
plt.title(f"{INSTANCE_SYMBOL}, {INSTANCE_TYPE}, beta={BETA}, Hamming={HAMMING}, approx ratio={AR}")

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
