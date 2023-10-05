import os
import pandas as pd
from tqdm import tqdm

cwd = os.getcwd()


def aggregate_dwave(topology: str, topology_size: int, type: str):
    symbol = f"P{topology_size}" if topology == "pegasus" else f"Z{topology_size}"
    dwave_path = os.path.join(cwd, "..", "energies", f"{topology}_random", symbol, type)
    save_path = os.path.join(cwd, "..", "energies", "aggregated", f"{topology}_random", symbol, type)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    current_instance = "000"
    master_df = pd.DataFrame()
    for filename in tqdm(os.listdir(dwave_path), desc=f"{symbol} {type}: "):
        file = os.path.join(dwave_path, filename)
        if os.path.isfile(file):

            filename = filename[:-4]
            instance_params = filename.split("_")
            instance_name = instance_params[0]
            if instance_name != current_instance:
                if not master_df.empty:
                    master_df = master_df.drop_duplicates()
                    master_df = master_df.sort_values(by="energy", ignore_index=True)
                    master_df.to_csv(os.path.join(save_path, f"{current_instance}.csv"))
                current_instance = instance_name
                master_df = pd.DataFrame()

            temp_df = pd.read_csv(file, index_col=0)
            temp_df["annealing_time"] = [instance_params[1] for _ in range(len(temp_df))]
            temp_df["num_reads"] = [instance_params[2] for _ in range(len(temp_df))]
            temp_df["pause_time"] = [instance_params[3] if len(instance_params) > 3 else 0
                                     for _ in range(len(temp_df))]
            temp_df["reverse"] = [True if len(instance_params) > 4 else False for _ in range(len(temp_df))]
            master_df = pd.concat([master_df, temp_df], ignore_index=True)
        # save last instance
        master_df = master_df.sort_values(by="energy", ignore_index=True)
        master_df.to_csv(os.path.join(save_path, f"{current_instance}.csv"))


if __name__ == '__main__':

    for size in [4, 8, 12, 16]:
        for t in ["AC3", "CBFM-P", "RAU", "RCO"]:
            aggregate_dwave("pegasus", size, t)







