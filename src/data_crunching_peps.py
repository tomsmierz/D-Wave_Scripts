import pandas as pd
import os
import re
from tqdm import tqdm

cwd = os.getcwd()

if __name__ == "__main__":
    folder = "RCO"
    size = "Z4"
    topology = "zephyr"
    path = os.path.join(cwd, f"..\\energies\\PEPS\\{size}")
    num_of_diff_runs = 1

    best_energy = []
    inst_name = []

    tr = pd.read_csv(os.path.join(path, "RCO_tr2^10.csv"), sep=";")
    instances = tr["instance"].unique()
    energy = []
    for i in instances:
        df = tr
        best_energy = df[df["instance"] == i].energy.min()
        energy.append(best_energy)

    data = {"instance": instances, "best_energy": energy}
    agregated = pd.DataFrame(data)
    print(agregated)
    agregated.to_csv(os.path.join(cwd, "..", "energies", "aggregated", size, "RCO_TR10.csv"), sep=";")

