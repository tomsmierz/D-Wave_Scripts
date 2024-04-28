import copy
import os
import json
import random
import time

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dwave_networkx as dnx
import networkx as nx

from collections import namedtuple
from renumeration import advantage_6_1_to_spinglass, advantage_6_1_to_spinglass_int
from scipy.spatial.distance import hamming
from typing import Optional, Union
from tqdm import tqdm
from itertools import zip_longest 

vector = Union[np.ndarray, list]

def xor(v1: vector, v2: vector) -> vector:
    assert len(v1) == len(v2)
    return [0 if v1[i] == v2[i] else 1 for i in range(len(v1))]

def hamming_dist(v1: vector, v2: vector) -> int:
    return hamming(v1, v2) * len(v1)

def create_spin_glass_peps_graph(file: str) -> nx.Graph:
    df = pd.read_csv(file, sep=" ", names=["v", "w", "J"], comment="#")
    edges = []
    nodes = []
    for row in df.itertuples():
        if row.v != row.w and row.J != 0:
            edges.append((row.v,row.w))
        elif row.v == row.w:
            nodes.append(row.v)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def connected_hamming_dist(state1: vector, state2: vector, graph) -> int:
    xor_state = xor(state1, state2)
    nodes = []
    for (i,node) in enumerate(graph.nodes):
        if xor_state[i]:
            nodes.append(node)
    if nodes == []:
        largest_cc = []
    else:
        subgraph = nx.subgraph(graph, nodes)
        largest_cc = max(nx.connected_components(subgraph), key=len)
    return len(largest_cc)

def find_droplets_hamming_connected(graph: nx.Graph, state_energy_tuple: namedtuple, hamming_cutoff: int,
                                    ground_eng: float, energy_cutoff: float, permutation: Optional[list] = None,
                                    checkpoint: Optional[namedtuple] = None):
    accepted_states = [] if not checkpoint else [list(checkpoint.state[j]) for j in
                                                 range(len(checkpoint.energy))]
    accepted_energies = [] if not checkpoint else list(checkpoint.energy)
    AcceptedStateEnergy = namedtuple('AcceptedStateEnergy', ['state', 'energy'])
    if permutation is not None:
        perm_states = [state_energy_tuple.state[i] for i in permutation]
        perm_energies = [state_energy_tuple.energy[i] for i in permutation]
    else:
        perm_states = state_energy_tuple.state
        perm_energies = state_energy_tuple.energy
    for idx, state in tqdm(enumerate(perm_states)):
        if not accepted_states:
            accepted_states.append(state)
            accepted_energies.append(perm_energies[idx])
        elif any(np.array_equal(state, accepted_state) for accepted_state in accepted_states):
            pass
        else:
            h_list = []
            for drop in accepted_states:
                h = connected_hamming_dist(state, drop, graph)
                h_list.append(h)
                eng = (abs(perm_energies[idx] - ground_eng))
            if eng <= energy_cutoff and all(h >= hamming_cutoff for h in h_list):
                accepted_states.append(state)
                accepted_energies.append(perm_energies[idx])
    return AcceptedStateEnergy(np.array(accepted_states), np.array(accepted_energies))


def find_droplets_hamming(state_energy_tuple: namedtuple, hamming_cutoff: int, ground_eng: float,
                          energy_cutoff: float, permutation: Optional[list] = None):
    accepted_states = [] 
    accepted_energies = [] 
    AcceptedStateEnergy = namedtuple('AcceptedStateEnergy', ['state', 'energy'])

    if permutation is not None:
        perm_states = [state_energy_tuple.state[i] for i in permutation]
        perm_energies = [state_energy_tuple.energy[i] for i in permutation]
    else:
        perm_states = state_energy_tuple.state
        perm_energies = state_energy_tuple.energy
    for idx, state in tqdm(enumerate(perm_states)):
        if not accepted_states:
            accepted_states.append(state)
            accepted_energies.append(perm_energies[idx])
        elif any(np.array_equal(state, accepted_state) for accepted_state in accepted_states):
            pass
        else:
            h_list = []
            for drop in accepted_states:
                h = hamming_dist(state, drop)
                h_list.append(h)
                eng = (abs(perm_energies[idx] - ground_eng))
            if eng <= energy_cutoff and all(h >= hamming_cutoff for h in h_list):
                accepted_states.append(state)
                accepted_energies.append(perm_energies[idx])
    return AcceptedStateEnergy(np.array(accepted_states), np.array(accepted_energies))


def find_max_set_connected(iterations: int, state_energy_tuple: namedtuple, hamming_cutoff: int,
                 ground_eng: float, energy_cutoff: float, graph: nx.Graph, file_path: str):
    set_size = 0
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    permutation = list(range(len(state_energy_tuple.energy)))
    if os.path.exists(file_path):
        result_df = pd.read_csv(file_path, sep=",", quotechar='"')
        max_iterations_stored = result_df['Iterations'].max()
        
        if max_iterations_stored >= iterations:
            max_iterations_row = result_df[result_df['Iterations'] == iterations]
            max_state = np.array(eval(max_iterations_row['State'].values[0]))
            max_energy = np.array(eval(max_iterations_row['Energy'].values[0]))
            new_state_energy_tuple = StateEnergy(max_state, max_energy)
        else:
            max_iterations_row = result_df[result_df['Iterations'] == max_iterations_stored]
            max_state = np.array(eval(max_iterations_row['State'].values[0]))
            max_energy = np.array(eval(max_iterations_row['Energy'].values[0]))
            new_state_energy_tuple = StateEnergy(max_state, max_energy)
            for i in range(max_iterations_stored+1, iterations+1):
                random.shuffle(permutation)
                accepted_state_energy_tuple = find_droplets_hamming_connected(graph, state_energy_tuple, hamming_cutoff,
                                                                ground_eng, energy_cutoff, permutation, new_state_energy_tuple)
                set_size = max_iterations_row['Count'].values[0]
                if len(accepted_state_energy_tuple.state) > set_size:
                    new_state_energy_tuple = StateEnergy(accepted_state_energy_tuple.state, accepted_state_energy_tuple.energy)
                new_data = {'Iterations': i, 'Energy': [list(new_state_energy_tuple.energy)],
                            'State': [[list(new_state_energy_tuple.state[j])
                                   for j in range(len(new_state_energy_tuple.energy))]],
                            'Count': len(new_state_energy_tuple.energy)}
                result_df = pd.concat([result_df, pd.DataFrame(new_data)], ignore_index=True)
                result_df.to_csv(file_path, index=False)
    else:
        result_df = pd.DataFrame(columns=['Iterations', 'Count', 'Energy', 'State'])
        for i in range(iterations):
            random.shuffle(permutation)
            accepted_state_energy_tuple = find_droplets_hamming_connected(graph, state_energy_tuple, hamming_cutoff,
                                                                ground_eng, energy_cutoff, permutation)
            if len(accepted_state_energy_tuple.state) > set_size or len(accepted_state_energy_tuple.state)==0:
                new_state_energy_tuple = StateEnergy(accepted_state_energy_tuple.state, accepted_state_energy_tuple.energy)
                set_size = len(accepted_state_energy_tuple.state)
            new_data = {'Iterations': i+1, 'Energy': [list(new_state_energy_tuple.energy)],
                        'State': [[list(new_state_energy_tuple.state[j])
                                   for j in range(len(new_state_energy_tuple.energy))]],
                        'Count': len(new_state_energy_tuple.energy)}
            result_df = pd.concat([result_df, pd.DataFrame(new_data)], ignore_index=True)
            result_df.to_csv(file_path, index=False)
    return new_state_energy_tuple


def find_max_set(iterations: int, state_energy_tuple: namedtuple, hamming_cutoff: int,
                 ground_eng: float, energy_cutoff: float, file_path: str):
    set_size = 0
    StateEnergy = namedtuple('StateEnergy', ['state', 'energy'])
    permutation = list(range(len(state_energy_tuple.energy)))

    if os.path.exists(file_path):
        result_df = pd.read_csv(file_path, index_col=0, sep=",", quotechar='"')
        max_iterations_stored = result_df['Iterations'].max()
        
        if max_iterations_stored >= iterations:
            max_iterations_row = result_df[result_df['Iterations'] == iterations]
            max_state = np.array(eval(max_iterations_row['State'].values[0]))
            max_energy = np.array(eval(max_iterations_row['Energy'].values[0]))
            new_state_energy_tuple = StateEnergy(max_state, max_energy)
        else:
            max_iterations_row = result_df[result_df['Iterations'] == max_iterations_stored]
            max_state = np.array(eval(max_iterations_row['State'].values[0]))
            max_energy = np.array(eval(max_iterations_row['Energy'].values[0]))
            new_state_energy_tuple = StateEnergy(max_state, max_energy)
            for i in range(max_iterations_stored+1, iterations+1):
                random.shuffle(permutation)
                accepted_state_energy_tuple = find_droplets_hamming(state_energy_tuple, hamming_cutoff,
                                                                ground_eng, energy_cutoff, permutation, new_state_energy_tuple)
                set_size = max_iterations_row['Count'].values[0]
                if len(accepted_state_energy_tuple.state) > set_size:
                    new_state_energy_tuple = StateEnergy(accepted_state_energy_tuple.state, accepted_state_energy_tuple.energy)
                new_data = {'Iterations': i, 'Energy': [list(new_state_energy_tuple.energy)],
                            'State': [[list(new_state_energy_tuple.state[j])
                                   for j in range(len(new_state_energy_tuple.energy))]],
                            'Count': len(new_state_energy_tuple.energy)}
                result_df = pd.concat([result_df, pd.DataFrame(new_data)], ignore_index=True)
                result_df.to_csv(file_path, index=False)
    else:
        result_df = pd.DataFrame(columns=['Iterations', 'Count', 'Energy', 'State'])
        for i in range(iterations):
            random.shuffle(permutation)
            accepted_state_energy_tuple = find_droplets_hamming(state_energy_tuple, hamming_cutoff,
                                                                ground_eng, energy_cutoff, permutation)
            if len(accepted_state_energy_tuple.state) > set_size:
                new_state_energy_tuple = StateEnergy(accepted_state_energy_tuple.state, accepted_state_energy_tuple.energy)
                set_size = len(accepted_state_energy_tuple.state)
            new_data = {'Iterations': i+1, 'Energy': [list(new_state_energy_tuple.energy)],
                        'State': [[list(new_state_energy_tuple.state[j])
                                   for j in range(len(new_state_energy_tuple.energy))]],
                        'Count': len(new_state_energy_tuple.energy)}
            result_df = pd.concat([result_df, pd.DataFrame(new_data)], ignore_index=True)
            result_df.to_csv(file_path, index=False)
    return new_state_energy_tuple
