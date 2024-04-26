import pandas as pd
import os

# TOPOLOGY = "pegasus"
# INSTANCE_SYMBOL = "P16"
# INSTANCE_TYPE = "RCO"
# TOPOLOGY_SIZE = 16

script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
root = os.path.dirname(script_dir)
# output_directory_solver = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, f"results_solver_ar01_dr025_allsolvers-new2")
# output_directory_merged = os.path.join(root, "droplets", INSTANCE_SYMBOL, INSTANCE_TYPE, f"results_solver_ar01_dr025_allsolvers-new2")

output_directory_solver = os.path.join(root, "droplets", "square", "results", "50x50", "solver_ar005_R0125_software")
output_directory_merged = os.path.join(root, "droplets", "square", "results", "50x50", "solver_ar005_R0125_software")


filename = 'union.csv'
merged_df = pd.DataFrame()

for subdir in os.listdir(output_directory_solver):
    subdir_path = os.path.join(output_directory_solver, subdir)
    if os.path.isdir(subdir_path):
        file_path = os.path.join(subdir_path, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            merged_df = pd.concat([merged_df, df], ignore_index=True)

merged_df.sort_values(by='Name', inplace=True)
output_file = os.path.join(output_directory_solver, 'merged_union.csv')

merged_df.to_csv(output_file, index=False)
