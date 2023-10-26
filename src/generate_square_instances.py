import random
import networkx as nx
import matplotlib.pyplot as plt

M = 5  # Change m to the desired lattice size
output_file = "ising_instance5x5.txt"  # Change the output file name if needed


def is_not_pair_in_ising_data(spin1, spin2, ising_data):
    for entry in ising_data:
        if (entry[0] == spin1 and entry[1] == spin2) or (entry[0] == spin2 and entry[1] == spin1):
            return False
    return True

def create_ising_instance_graph(m, output_file):
    G = nx.Graph()

    # Create a list to store the Ising instance data
    ising_data = []

    # Generate nodes for the spins
    for i in range(m):
        for j in range(m):
            node = i * m + j

            # Generate four spins within each unit cell
            spins = [node * 4+1, node * 4 + 2, node * 4 + 3, node * 4 + 4]
            # Add couplings within the unit cell
            for spin1 in spins:
                for spin2 in spins:
                    if spin1 != spin2 and is_not_pair_in_ising_data(spin1, spin2, ising_data):
                        Jij = random.uniform(-1, 1)
                        G.add_edge(spin1, spin2, weight=Jij)
                        ising_data.append((spin1, spin2, Jij))

            # Connect spins of adjacent nodes
            if i > 0:
                neighbor = ((i - 1) * m + j) * 4 +1
                for spin1 in spins:
                    for spin2 in range(neighbor, neighbor + 4):
                        if is_not_pair_in_ising_data(spin1, spin2, ising_data):
                            Jij = random.uniform(-1, 1)
                            G.add_edge(spin1, spin2, weight=Jij)
                            ising_data.append((spin1, spin2, Jij))
            if j > 0:
                neighbor = (i * m + (j - 1)) * 4 +1
                for spin1 in spins:
                    for spin2 in range(neighbor, neighbor + 4):
                        if is_not_pair_in_ising_data(spin1, spin2, ising_data):
                            Jij = random.uniform(-1, 1)
                            G.add_edge(spin1, spin2, weight=Jij)
                            ising_data.append((spin1, spin2, Jij))

    # Write the data to the output file
    with open(output_file, 'w') as file:
        for entry in ising_data:
            spin1, spin2, Jij = entry
            file.write(f"{spin1} {spin2} {Jij}\n")

    return G

if __name__ == '__main__':
    # ising_data=[(1,2,3), (1,4,7)]

    # a= is_not_pair_in_ising_data(1, 3, ising_data)
    # print(a)
    ising_graph = create_ising_instance_graph(M, output_file)

    # pos = nx.spring_layout(ising_graph)  # Layout for the plot
    # edge_labels = {(u, v): d['weight'] for u, v, d in ising_graph.edges(data=True)}
    # nx.draw(ising_graph, pos, with_labels=True, node_size=200, font_size=10)
    # nx.draw_networkx_edge_labels(ising_graph, pos, edge_labels=edge_labels)
    # plt.axis('off')
    # plt.show()
