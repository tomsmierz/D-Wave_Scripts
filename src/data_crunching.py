import pandas as pd
import os
import re
from tqdm import tqdm

cwd = os.getcwd()

if __name__ == "__main__":
    folder = "RCO"
    size = "Z3"
    path = os.path.join(cwd, f"..\\energies\\zephyr_random\\{size}\\{folder}")

    best_dwave = []
    prob_best = []
    inst_name = []

    num_reads = 445  # depends on problem class
    pattern = re.compile(".{3}_2000_445\.csv")
    for file in tqdm(os.listdir(path)):
        match = pattern.fullmatch(file)
        if match:

            df = pd.read_csv(os.path.join(path, file))
            df_min = df[df.energy == df.energy.min()]

            inst = file[:3]
            inst_name.append(inst)

            best = df.energy.min()
            best_dwave.append(best)

            prob = sum(df_min.num_occurrences)/num_reads
            prob = round(100*prob, 2)
            prob_best.append(prob)

    data = {"instance": inst_name, "best_dwave": best_dwave, "probability_dwave [%]": prob_best}
    aggregated_results = pd.DataFrame(data)
    save_path = os.path.join(cwd, f"..\\energies\\aggregated\\{size}")
    name = f"{folder}_dwave.csv"
    aggregated_results.to_csv(os.path.join(save_path, name), sep=";")
    print(aggregated_results)
