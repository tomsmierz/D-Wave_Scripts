import numpy as np
import os

def generate_spin_instances(N):
    spins = []

    for i in range(N):
        for j in range(N):
            if i < N - 1:  # Interaction with spin below
                Jij = np.random.uniform(-1, 1)
                spins.append((i * N + j + 1, (i + 1) * N + j + 1, Jij))
            if j < N - 1:  # Interaction with spin to the right
                Jij = np.random.uniform(-1, 1)
                spins.append((i * N + j + 1, i * N + j + 2, Jij))
            if i < N - 1 and j < N - 1:  # Interaction with spin diagonally below and to the right
                Jij = np.random.uniform(-1, 1)
                spins.append((i * N + j + 1, (i + 1) * N + j + 2, Jij))
            if i < N - 1 and j > 0:  # Interaction with spin diagonally below and to the left
                Jij = np.random.uniform(-1, 1)
                spins.append((i * N + j + 1, (i + 1) * N + j, Jij))

    return spins

def generate_spin_instances_bias(N):
    spins = []

    for i in range(N):
        for j in range(N):
            if i < N - 1:  # Interaction with spin below
                Jij = np.random.uniform(-1, 1)
                spins.append((i * N + j + 1, (i + 1) * N + j + 1, Jij))
            if j < N - 1:  # Interaction with spin to the right
                Jij = np.random.uniform(-1, 1)
                spins.append((i * N + j + 1, i * N + j + 2, Jij))
            if i < N - 1 and j < N - 1:  # Interaction with spin diagonally below and to the right
                Jij = np.random.uniform(-1, 1)
                spins.append((i * N + j + 1, (i + 1) * N + j + 2, Jij))
            if i < N - 1 and j > 0:  # Interaction with spin diagonally below and to the left
                Jij = np.random.uniform(-1, 1)
                spins.append((i * N + j + 1, (i + 1) * N + j, Jij))
            
            Jii = np.random.uniform(-0.1, 0.1)
            spins.append((i * N + j + 1, i * N + j + 1, Jii))

    return spins


def save_spin_instances_to_file(N, instances, folder, filename):
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w') as file:
        for instance in instances:
            file.write(f"{instance[0]} {instance[1]} {instance[2]}\n")

if __name__ == "__main__":
    N_values = [20, 30, 40, 50]
    num_instances_per_size = 100

    for N in N_values:
        folder_name = f"square_diag_bias_{N}x{N}"
        os.makedirs(folder_name, exist_ok=True)

        for instance_num in range(1, num_instances_per_size + 1):
            instances = generate_spin_instances_bias(N)
            filename = f"{str(instance_num).zfill(3)}.txt"
            save_spin_instances_to_file(N, instances, folder_name, filename)
            print(f"Square instances for N={N}, instance {instance_num} saved to {folder_name}/{filename}")