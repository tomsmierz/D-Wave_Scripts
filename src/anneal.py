from dwave.system import DWaveSampler
import dwave.inspector
import pickle
import pandas

"""
annealing_time_range = [0.5, 2000.0] microseconds
problem_run_duration_range = [0.0, 1000000.0] microseconds
"""


topology = "zephyr"
symbol = "Z"
sampler = DWaveSampler(solver="Advantage2_prototype1.1")  # DWaveSampler(solver="Advantage_system6.1")


instance_number = "001"
size = 1
for category in ["RAU"]:  # ["AC3", "RCO", "RAU"]:
    with open(f"../instances/{topology}_random/{symbol}{size}/{category}/{instance_number}_dv.pkl", "rb") as f:
        h, J = pickle.load(f)

    for annealing_time in [2, 200, 2000]:
        num_reads = 300 if annealing_time == 2000 else 1000
        sample_set = sampler.sample_ising(h, J, annealing_time=annealing_time, num_reads=num_reads, auto_scale=False,
                                          label=f"{topology} {symbol}{size} {instance_number} {category} {annealing_time}")
        print(sample_set.info)
        df = sample_set.to_pandas_dataframe()
        df.to_csv(f"../energies/{topology}_random/{symbol}{size}/{category}/{instance_number}_{annealing_time}_{num_reads}.csv")


