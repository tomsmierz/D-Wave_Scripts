import os
import pandas as pd
import h5py
import json
import pickle

USER_HOME_DIR = os.path.expanduser('~')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
solutions = os.path.join(USER_HOME_DIR, r"Desktop\square")
instances = os.path.join(ROOT, r"instances\pegasus_random\P8\RCO")


if __name__ == '__main__':
    err = 0
    for filename in os.listdir(solutions):
        file = os.path.join(solutions, filename)
        if os.path.isfile(file):
            try:
                with open(file) as f:
                    data = json.load(f)
                    name = data["columns"][0][0][0:-7]
                    energy = data["columns"][4][0]
                    state = data["columns"][15][0][0]
                    for filename2 in os.listdir(instances):
                        file2 = os.path.join(instances, filename2)
                        if os.path.isfile(file2):
                            if not filename2.endswith('.pkl'):
                                name2, _ = filename2.split("_")
                                if name2 == name:
                                    inst = pd.read_csv(file2, sep=" ", header=None,
                                                       names=["x", "y", "v"], comment="#")
                                    en = 0
                                    for row in inst.itertuples():
                                        en += state[str(row.x)] * state[str(row.y)] * row.v
                                    if en == energy:
                                        print(name, "good")
                                    else:
                                        print(name, en, energy, abs(en-energy))
            except:
                print(filename)
                err += 1

    print(err)







