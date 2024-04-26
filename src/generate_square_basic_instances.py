import numpy as np
import os

def generate_spin_instances(N):
    spins = []

    for i in range(N):
        for j in range(N):
            if i < N - 1:
                Jij = np.random.uniform(-1, 1)
                spins.append((i * N + j + 1, (i + 1) * N + j + 1, Jij))  # Spin na pozycji (i, j) oddziałuje z spinem poniżej
            if j < N - 1:
                Jij = np.random.uniform(-1, 1)
                spins.append((i * N + j + 1, i * N + j + 2, Jij))  # Spin na pozycji (i, j) oddziałuje z spinem obok

    return spins

def generate_spin_instances_bias(N):
    spins = []
    local_fields = []

    for i in range(N):
        for j in range(N):
            # Dodawanie spinów oddziałujących z sąsiadami
            if i < N - 1:
                Jij = np.random.uniform(-1, 1)
                spins.append((i * N + j + 1, (i + 1) * N + j + 1, Jij))  # Spin na pozycji (i, j) oddziałuje z spinem poniżej
            if j < N - 1:
                Jij = np.random.uniform(-1, 1)
                spins.append((i * N + j + 1, i * N + j + 2, Jij))  # Spin na pozycji (i, j) oddziałuje z spinem obok

            # Generowanie losowego pola lokalnego z przedziału [-0.1, 0.1] dla każdego spinu
            Jii = np.random.uniform(-0.1, 0.1)
            spins.append((i * N + j + 1, i * N + j + 1, Jii))

    return spins

def save_spin_instances_to_file(N, instances, folder, filename):
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w') as file:
        for instance in instances:
            file.write(f"{instance[0]} {instance[1]} {instance[2]}\n")
        # for instance in instances:
        #     for spin in instance:
        #         file.write(f"{spin[0]} {spin[1]} {spin[2]}\n")

if __name__ == "__main__":
    N_values = [20, 30, 40, 50]
    num_instances_per_size = 100

    for N in N_values:
        folder_name = f"square_bias_{N}x{N}"
        os.makedirs(folder_name, exist_ok=True)

        for instance_num in range(1, num_instances_per_size + 1):
            instances = generate_spin_instances_bias(N)
            filename = f"{str(instance_num).zfill(3)}.txt"
            save_spin_instances_to_file(N, instances, folder_name, filename)
            print(f"Square instances for N={N}, instance {instance_num} saved to {folder_name}/{filename}")