from dwave.system import DWaveSampler
import dwave.inspector

import pickle
import pandas
from tqdm import tqdm

"""
annealing_time_range = [0.5, 2000.0] microseconds
problem_run_duration_range = [0.0, 1000000.0] microseconds
"""


# topology = "pegasus"
# symbol = "P" if topology == "pegasus" else "Z"
# sampler = DWaveSampler(solver="Advantage_system6.1")  # DWaveSampler(solver="Advantage2_prototype1.1")  #

if __name__ == "__main__":
    sampler = DWaveSampler(solver="Advantage_system6.1")

    topology = "pegasus"
    symbol = "P"
    size = 8
    category = "CBFM-P"
    i = 2
    instance_number = f"00{i}"[-3:]
    with open(f"../instances/{topology}_random/{symbol}{size}/{category}/{instance_number}_dv.pkl", "rb") as f:
        h, J = pickle.load(f)
    sample_set = sampler.sample_ising(h, J, num_reads=10)
    print(sample_set)



    #                                                       label=f"{topology} {symbol}{size} {instance_number} {category} {annealing_time}")
    # for topology in ["pegasus"]: #["pegasus", "zephyr"]:
    #     sizes = [8] # [4, 8, 12, 16] if topology == "pegasus" else [2, 3, 4]
    #     sampler = DWaveSampler(solver="Advantage_system6.1") if topology == "pegasus" else DWaveSampler(
    #         solver="Advantage2_prototype1.1")
    #     symbol = "P" if topology == "pegasus" else "Z"
    #     for j in sizes:
    #         size = j
    #         for i in tqdm(range(1, 101)):
    #             instance_number = f"00{i}"[-3:]
    #             for category in ["RAU"]:  # ["AC3", "RCO", "RAU", "CBFM-P"]:
    #                 with open(f"../instances/{topology}_random/{symbol}{size}/{category}/{instance_number}_dv.pkl", "rb") as f:
    #                     h, J = pickle.load(f)
    #
    #                 for annealing_time in [2000]:  # [2, 200, 2000]:
    #                     num_reads = 445 if annealing_time == 2000 else 2500 if annealing_time == 200 else 6000
    #                     sample_set = sampler.sample_ising(h, J, annealing_time=annealing_time, num_reads=num_reads, auto_scale=False,
    #                                                       label=f"{topology} {symbol}{size} {instance_number} {category} {annealing_time}")
    #
    #                     df = sample_set.to_pandas_dataframe()
    #                     df.to_csv(f"../energies/{topology}_random/{symbol}{size}/{category}/{instance_number}_{annealing_time}_{num_reads}.csv")


