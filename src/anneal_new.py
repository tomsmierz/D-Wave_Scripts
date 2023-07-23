from dwave.system import DWaveSampler
from tqdm import tqdm
import numpy as np
import pickle
import os

cwd = os.getcwd()

if __name__ == '__main__':
    token = "jszS-a65db66e0d37bf62c83f45adad63cd25c5f8dc53"
    topology = "pegasus"
    sizes = [8, 12, 16] if topology == "pegasus" else [2, 3, 4]
    sampler = (
        DWaveSampler(solver="Advantage_system6.2", token=token)
        if topology == "pegasus"
        else DWaveSampler(solver="Advantage2_prototype1.1", token=token)
    )
    symbol = "P" if topology == "pegasus" else "Z"
    QCP = 0.3 if topology == "pegasus" else 0.25
    for size in sizes:
        for i in tqdm(range(1, 101)):
            instance_number = f"00{i}"[-3:]
            for category in ["CBFM-P"]:  # ["AC3", "RCO", "RAU", "CBFM-P"]:
                with open(
                    f"../instances/{topology}_random/{symbol}{size}/{category}/{instance_number}_dv.pkl",
                    "rb",
                ) as f:
                    h, J = pickle.load(f)

                annealing_time = 2000
                pause = 200
                anneal_schedule = [[0, 0], [(annealing_time - pause)/2, QCP], [(annealing_time + pause)/2, QCP],
                                  [annealing_time, 1]]
                num_reads = 445
                sample_set = sampler.sample_ising(
                    h,
                    J,
                    anneal_schedule=anneal_schedule,
                    num_reads=num_reads,
                    auto_scale=False,
                    label=f"{topology} {symbol}{size} {instance_number} {category} {annealing_time}",
                )
                df = sample_set.to_pandas_dataframe()
                df.to_csv(
                    f"../energies/{topology}_random/{symbol}{size}/{category}/"
                    f"{instance_number}_{annealing_time}_{num_reads}_{pause}.csv"
                )


