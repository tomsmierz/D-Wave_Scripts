import os
import json
import pandas as pd
from tqdm import tqdm

USER_HOME_DIR = os.path.expanduser('~')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json_source = os.path.join(ROOT, "energies", "sbm", "square", "50x50x1")
csv_destination = os.path.join(ROOT, "energies", "aggregated", "square", "50x50")


if __name__ == '__main__':
    for filename in tqdm(os.listdir(json_source)):
        file = os.path.join(json_source, filename)
        try:
            if os.path.isfile(file):
                with open(file) as f:
                    data = json.load(f)
                    tmp = pd.DataFrame()
                    # lookup = data["colindex"]["lookup"]
                    # names = data["colindex"]["names"]
                    name = data["columns"][0][0]  # get string
                    name = [name.rsplit(".", 1)[0]]
                    beta = data["columns"][1]
                    layout = data["columns"][2]
                    strategy = data["columns"][3]
                    sparsity = data["columns"][4]
                    transform = data["columns"][5]
                    energy = data["columns"][6]
                    energies = data["columns"][12][0]
                    ig_states = data["columns"][8][0]

                    states = pd.DataFrame(ig_states)
                    states = states.reindex(columns=sorted(states.columns, key=lambda x: int(x)))
                    states = states.drop_duplicates()

                    tmp["instance"] = name * len(states)
                    tmp["beta"] = beta * len(states)
                    tmp["layout"] = layout * len(states)
                    tmp["strategy"] = strategy * len(states)
                    tmp["sparsity"] = sparsity * len(states)
                    tmp["transform"] = transform * len(states)
                    tmp["energy"] = energies

                    save_name = name[0] + ".csv"

                    if save_name in os.listdir(csv_destination):
                        df = pd.read_csv(os.path.join(csv_destination, save_name), index_col=0)
                    else:
                        df = pd.DataFrame()
                    tmp = pd.concat([tmp, states], axis=1)
                    df = pd.concat([df, tmp], axis=0, ignore_index=True)
                    df.to_csv(os.path.join(csv_destination, save_name))
        except:
            print(filename)



