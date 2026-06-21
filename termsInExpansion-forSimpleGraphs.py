import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import scipy.linalg as la
from scipy.linalg import expm
import networkx as nx
from collections import defaultdict
import ast
from sklearn.metrics.cluster import normalized_mutual_info_score
import time
import pynauty

def get_orbits_from_file(original_filename):
    orbits_filename = original_filename + "_orbits.txt"
    orbits = []

    with open(orbits_filename) as f:
        num_of_orbits = int(next(f).split()[0])
        for i in range(num_of_orbits):
            orbits.append(ast.literal_eval(next(f)))
    
    return orbits

def turn_membership_array_into_clustering_array(array, num_of_nodes, num_of_classes):
    clustering = np.zeros(num_of_nodes)

    for i in range(len(array)):
        clustering[array[i]] = i

    return clustering

def compute_orbits(G):

    nodes = list(G.nodes())

    mapping = {v: i for i, v in enumerate(nodes)}
    inverse = {i: v for v, i in mapping.items()}

    adjacency = {
        mapping[v]: [mapping[u] for u in G.neighbors(v)]
        for v in nodes
    }

    g = pynauty.Graph(
        number_of_vertices=len(nodes),
        directed=False,
        adjacency_dict=adjacency
    )

    _, _, _, orbit_labels, _ = pynauty.autgrp(g)

    orbit_dict = {}
    for vertex, orbit_id in enumerate(orbit_labels):
        orbit_dict.setdefault(orbit_id, []).append(inverse[vertex])

    return list(orbit_dict.values())


if __name__ == "__main__":
    start_time = time.perf_counter()

    filename  = "smallGraphsFromAmer/graph-01"

    with open(filename) as f:
        num_of_nodes = int(next(f).split()[0])
        G = nx.parse_edgelist(f, nodetype=int)

    G.add_nodes_from(range(num_of_nodes))
    laplacianMatrix = nx.laplacian_matrix(G, nodelist=range(num_of_nodes)).toarray()

    adjacency = {v: list(G.neighbors(v)) for v in G.nodes()}

    g = pynauty.Graph(
        number_of_vertices=G.number_of_nodes(),
        directed=False,
        adjacency_dict=adjacency,
    )

    orbits = compute_orbits(G)
    orbits_clustering = turn_membership_array_into_clustering_array(orbits, num_of_nodes, len(orbits))

    print("orbits: ", orbits)

    eigenvals, eigenvecs = np.linalg.eigh(laplacianMatrix)
    eigensolver_time = time.perf_counter()
    eigenvalues = np.real(eigenvals)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    round_digits = 9
    resolution = 1e9
    eigenvalues = np.round(eigenvalues, round_digits)
    lambda_fit = np.unique(np.sort(eigenvalues))

    Ti = np.arange(1 / np.max(lambda_fit), 1 / np.min(lambda_fit), 0.1)

    # For the diagonal only
    coeffs = np.zeros((num_of_nodes, len(Ti)))
    for i in range(len(Ti)):
        exp_lambda = np.diag(np.exp( - eigenvals * Ti[i]))
        matrix_exponential = eigenvecs @ exp_lambda @ eigenvecs.T
        for j in range(num_of_nodes):
            coeffs[j][i] = matrix_exponential[j][j]

    # For the whole row
    # coeffs = np.zeros((num_of_nodes, len(Ti), num_of_nodes))

    # for i in range(len(Ti)):
    #     exp_lambda = np.diag(np.exp( - eigenvals * Ti[i]))
    #     matrix_exponential = eigenvecs @ exp_lambda @ eigenvecs.T
    #     for j in range(num_of_nodes):
    #         coeffs[j][i] = np.sort(matrix_exponential[j])

    coeffs = np.trunc(coeffs * resolution) / resolution

    groups = defaultdict(list)

    # For the diagonal only
    for i, row in enumerate(coeffs): 
        groups[tuple(row)].append(i)

    # For the whole row
    # for i, row in enumerate(coeffs):
    #     key = tuple(map(tuple, row))
    #     groups[key].append(i)
            
    equivalence_classes = list(groups.values())
    print("equivalence classes: ", equivalence_classes)

    equivalence_classes_clustering = turn_membership_array_into_clustering_array(equivalence_classes, num_of_nodes, len(equivalence_classes))

    # print("NMI (original): ", normalized_mutual_info_score(orbits_clustering[:136], equivalence_classes_clustering[:136]))
    print("NMI (all): ", normalized_mutual_info_score(orbits_clustering, equivalence_classes_clustering))

    print("unique equivalence classes: ", len(np.unique(equivalence_classes_clustering)))