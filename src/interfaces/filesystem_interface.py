import pickle
from pathlib import Path


def write_dwave_file(device, mapping, bias, couplings, output_path: Path):
    if device is not None:
        couplings_dv = {
            (mapping(edge[0]), mapping(edge[1])): value
            for edge, value in couplings.items()
        }
        bias_dv = {mapping(node): value for node, value in bias.items()}
        data = [bias_dv, couplings_dv]
    else:
        data = [bias, couplings]

    with output_path.open(mode="wb") as f:
        pickle.dump(data, f)


def write_spin_glass_file(bias_sg, couplings_sg, target_path: Path):
    bias_sg = dict(sorted(bias_sg.items()))

    with target_path.open("w") as f:
        f.write("# \n")
        for node, value in bias_sg.items():
            f.write(f"{str(node)} {str(node)} {str(value)}" + "\n")
        for edge, value in couplings_sg.items():
            f.write(f"{str(edge[0])} {str(edge[1])} {str(value)}" + "\n")