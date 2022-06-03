import dwave_networkx as dnx  # type: ignore
import argparse

from os import path


def generate_chimera_instance(number: int, size: int, out: str):
    chimera = dnx.chimera_graph(size, size)
    for _ in range(number):
        pass
        #  p[-3:]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--size", type=int, choices=[i+1 for i in range(16)], default=4,
                        help="Size of the chimera graph. Default is 4 (C4).")
    parser.add_argument("-N", "--number", type=int, default=1,
                        help="Number of instances to be generated. Default is 1, Maximum 999.")

    args = parser.parse_args()

    if args.number and args.number >= 1000:
        parser.error("Maximum number of generated instances is 999.")

    generate_chimera_instance(args.number, args.size, "")
