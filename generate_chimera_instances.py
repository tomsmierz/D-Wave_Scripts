import dwave_networkx as dnx  # type: ignore
import argparse

from os import path


def generate_chimera_instance(size: int, out: str):
    chimera = dnx.chimera_graph(size, size)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--size", type=int, choices=[i+1 for i in range(16)], default=4,
                        help="Size of the chimera graph. Default is 4 (C4).")
    parser.add_argument("-N", "--number", type=int, default=1,
                        help="Number of instances to be generated.")

    args = parser.parse_args()

    generate_chimera_instance(args.size, "")
