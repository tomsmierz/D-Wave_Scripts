import pandas as pd
import os
import re
from tqdm import tqdm

cwd = os.getcwd()

if __name__ == "__main__":
    folder = "CBFM-P"
    size = "P8"
    topology = "pegasus"
    path = os.path.join(cwd, f"..\\energies\\{topology}_random\\{size}\\{folder}")
    num_of_diff_runs = 1

    best_dwave = []
    avg_dwave = []
    median_dave = []
    prob_best = []
    inst_name = []

    for i in tqdm(range(1, 101)):
        num = f"00{i}"[-3::]
        c = 0
        pattern = re.compile(num)
        min_values = []
        probs = []
        for file in os.listdir(path):
            match = pattern.match(file)
            if match:
                df = pd.read_csv(os.path.join(path, file))
                df_min = df[df.energy == df.energy.min()]
                prob = df_min.num_occurrences.sum() / df.num_occurrences.sum()
                prob = round(100*prob, 2)

                min_values.append(df_min.energy[0])
                probs.append(prob)

                c += 1
                if c >= num_of_diff_runs:
                    inst_name.append(num)
                    best = min(min_values)
                    best_index = min_values.index(best)
                    prob_of_min = probs[best_index]
                    best_dwave.append(best)
                    prob_best.append(prob_of_min)
                    break

    data = {"instance": inst_name, "best_dwave": best_dwave, "probability_dwave [%]": prob_best}
    aggregated_results = pd.DataFrame(data)
    save_path = os.path.join(cwd, f"..\\energies\\aggregated\\{size}")
    name = f"{folder}_dwave.csv"
    aggregated_results.to_csv(os.path.join(save_path, name), sep=";")
    print(aggregated_results)
