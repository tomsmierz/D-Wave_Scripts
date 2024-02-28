import h5py
import os
import pandas as pd
import h5py
import json
import pickle

USER_HOME_DIR = os.path.expanduser('~')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
solutions = os.path.join(USER_HOME_DIR, "Desktop", "sdiag", "square_diag_50x50", "tmp")
instances = os.path.join(ROOT, "instances", "square_diag", "square_diag_50x50")


if __name__ == '__main__':
    err = 0
    for filename in os.listdir(solutions):
        file = os.path.join(solutions, filename)
        name, _ = filename.split(".")
        if os.path.isfile(file) and filename.endswith(".h5"):
            try:
                with h5py.File(file, 'r') as file:
                    # Iterate through the groups and datasets in the file
                    spectrum = file["Spectrum"]
                    energies = spectrum["energies"]
                    states = spectrum["states"]

                    best_energy = energies[0]
                    best_state = states[0]
                    for filename2 in os.listdir(instances):
                        file2 = os.path.join(instances, filename2)
                        if os.path.isfile(file2) and filename2.endswith('.txt'):
                            name2, _ = filename2.split(".")
                            if name2 == name:
                                inst = pd.read_csv(file2, sep=" ", header=None,
                                                   names=["x", "y", "v"], comment="#")
                                en = 0
                                for row in inst.itertuples():
                                    s_i = best_state[int(row.x) - 1]
                                    s_j = best_state[int(row.y) - 1]
                                    J_ij = row.v
                                    en += s_i * s_j * J_ij
                                if en == best_energy:
                                    print(name, "energies are matching")
                                else:
                                    print(name, "energies are not matching, difference: ", abs(en-best_energy))

            except Exception as e:
                err += 1
                print("error: ", filename, e)

    print("number of errored files: ", err)

