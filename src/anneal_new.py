from dwave.system import DWaveSampler, LazyFixedEmbeddingComposite
from tqdm import tqdm
import numpy as np
import pickle
import os

rng = np.random.default_rng()
cwd = os.getcwd()

if __name__ == '__main__':
    token = "jszS-a65db66e0d37bf62c83f45adad63cd25c5f8dc53"
    for topology in ["pegasus", "zephyr"]:

        sizes = [4, 8, 12, "16_6.2"] if topology == "pegasus" else [2, 3, 4] #[4, 8, 12]
        sampler = (
            DWaveSampler(solver="Advantage_system6.2", token=token)
            if topology == "pegasus"
            else DWaveSampler(solver="Advantage2_prototype1.1", token=token)
        )
        symbol = "P" if topology == "pegasus" else "Z"
        QCP = 0.3 if topology == "pegasus" else 0.25
        for size in sizes:
            for i in tqdm(range(1, 101), desc=f"{topology} {size} {i}"):
                instance_number = f"00{i}"[-3:]
                categories = ["AC3", "RCO", "RAU"]
                if topology == "pegasus":
                    categories += ["CBFM-P"]
                for category in categories:
                    with open(
                        f"../instances/{topology}_random/{symbol}{size}/{category}/{instance_number}_dv.pkl",
                        "rb",
                    ) as f:
                        h, J = pickle.load(f)

                    annealing_time = 2000
                    # pause = 200
                    pauses = [100, 300, 400, 600, 700, 800, 900, 1100, 1200, 1300, 1400, 1600, 1700, 1800, 1900]
                    if size == "16_6.2":
                        pauses += [200, 500, 1000, 1500]
                    for pause in pauses:
                        anneal_schedule = [[0, 0], [(annealing_time - pause)/2, QCP], [(annealing_time + pause)/2, QCP],
                                          [annealing_time, 1]]
                        reverse_anneal = [[0, 1], [(annealing_time - pause)/2, QCP], [(annealing_time + pause)/2, QCP],
                                          [annealing_time, 1]]
                        num_reads = 445
                        initial_state = {i: rng.choice([-1, 1]) for i in h.keys()}
                        for anneal in ["forward", "reverse"]:
                            sample_set = sampler.sample_ising(
                                h,
                                J,
                                anneal_schedule=anneal_schedule if anneal == "forward" else reverse_anneal,
                                num_reads=num_reads,
                                auto_scale=False,
                                label=f"{topology} {size} {instance_number} {category} {annealing_time}",
                                initial_state=initial_state if anneal == "reverse" else None,
                                reinitialize_state=True if anneal == "reverse" else False
                            )
                            df = sample_set.to_pandas_dataframe()
                            an = "" if anneal == "forward" else "_reverse"
                            df.to_csv(
                                f"../energies/{topology}_random/{symbol}{size}/{category}/"
                                f"{instance_number}_{annealing_time}_{num_reads}_{pause}{an}.csv"
                            )


