import random
import networkx as nx
import matplotlib.pyplot as plt

M = 5  # Change m to the desired lattice size
output_file = "ising_instance5x5a.txt"  # Change the output file name if needed


def is_not_pair_in_ising_data(spin1, spin2, ising_data):
    for entry in ising_data:
        if (entry[0] == spin1 and entry[1] == spin2) or (entry[0] == spin2 and entry[1] == spin1):
            return False
    return True

def create_ising_instance_graph(m, output_file):
    ising_data = []
    for i in range(m):
        for j in range(m):
            node = i * m + j

            spins = [node * 4+1, node * 4 + 2, node * 4 + 3, node * 4 + 4]
            for spin1 in spins:
                for spin2 in spins:
                    if spin1 != spin2 and is_not_pair_in_ising_data(spin1, spin2, ising_data):
                        Jij = random.uniform(-1, 1)
                        ising_data.append((spin1, spin2, Jij))

            if i > 0:
                neighbor = ((i - 1) * m + j) * 4 +1
                for spin1 in spins:
                    for spin2 in range(neighbor, neighbor + 4):
                        if is_not_pair_in_ising_data(spin1, spin2, ising_data):
                            Jij = random.uniform(-1, 1)
                            ising_data.append((spin1, spin2, Jij))
            if j > 0:
                neighbor = (i * m + (j - 1)) * 4 +1
                for spin1 in spins:
                    for spin2 in range(neighbor, neighbor + 4):
                        if is_not_pair_in_ising_data(spin1, spin2, ising_data):
                            Jij = random.uniform(-1, 1)
                            ising_data.append((spin1, spin2, Jij))

    with open(output_file, 'w') as file:
        for entry in ising_data:
            spin1, spin2, Jij = entry
            file.write(f"{spin1} {spin2} {Jij}\n")

    return

if __name__ == '__main__':
    ising_graph = create_ising_instance_graph(M, output_file)

