import os
import pandas as pd
import matplotlib.pyplot as plt

# Instance characteristic
TOPOLOGY = "pegasus"
INSTANCE_SYMBOL = "P4"
INSTANCE_TYPE = "CBFM-P"
TOPOLOGY_SIZE = 4
# Define the folder where the CSV files are stored
script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
root = os.path.dirname(script_dir)
results_folder = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, "results")


# Initialize an empty list to store dataframes and file names
dataframes = []

# Loop through CSV files in the folder
for filename in os.listdir(results_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(results_folder, filename)
        df = pd.read_csv(file_path)
        dataframes.append((df, filename))  # Store both the dataframe and the file name

# Create a plot for each dataframe and save as an image
for df, filename in dataframes:
    plt.figure(figsize=(8, 6))
    plt.plot(df['Iterations'], df['Count'], "bo")
    plt.title(f'Counts vs. Iterations - {filename}')
    plt.xlabel('Iterations')
    plt.ylabel('Counts')
    plt.grid()
    plt.savefig(f'{filename}.png')  # Save the figure as a PNG file with the same name as the CSV file
    plt.show()
