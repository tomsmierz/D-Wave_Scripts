import pandas as pd
import h5py

path = r"C:\Users\tsmierzchalski\PycharmProjects\D-Wave_Scripts\energies\sbm\pegasus_random\P4\CBFM-P\SpinGlass\tmp\001_sg.h5"
f = h5py.File(path, "r")

spectrum = f['Spectrum']
energies = spectrum['energies']
states = spectrum["states"]


print(states[0])
def array_to_dict(array):
    return {i + 1: array[i] for i in range(len(array))}

print(array_to_dict(states[0]))

instance_path = r"C:\Users\tsmierzchalski\PycharmProjects\D-Wave_Scripts\instances\pegasus_random\P4\CBFM-P\001_sg.txt"
instance = pd.read_csv(instance_path, header=None, comment="#", sep=" ", names=["s1", "s2", "v"])
print(instance)

def energy(instance: pd.DataFrame, sol):
    e = 0
    for row in instance.itertuples():
        if row.s1 == row.s2:
            e += sol[row.s1] * row.v
        else:
            e += sol[row.s1] * sol[row.s2] * row.v
    return e

print(states.shape)