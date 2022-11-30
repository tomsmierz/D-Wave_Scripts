from dwave.system import DWaveSampler
import dwave.inspector
import pickle
import pandas

"""
annealing_time_range = [0.5, 2000.0] microseconds
problem_run_duration_range = [0.0, 1000000.0] microseconds
"""

sampler = DWaveSampler(solver="Advantage_system6.1")

for category in ["AC3", "RCO", "RAU"]:
    with open(f"../instances/pegasus_random/P4/{category}/001_dv.pkl", "rb") as f:
        h, J = pickle.load(f)

    for annealing_time in [2, 200, 2000]:
        num_reads = 300 if annealing_time == 2000 else 1000
        sample_set = sampler.sample_ising(h, J, annealing_time=annealing_time, num_reads=num_reads, auto_scale=False)
        df = sample_set.to_pandas_dataframe()
        df.to_csv(f"../energies/pegasus_random/P4/{category}/001_{annealing_time}_{num_reads}.csv")


