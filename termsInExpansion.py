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

if __name__ == "__main__":
    start_time = time.perf_counter()

    filename  = "ags/ag2-08__02"
    with open(filename) as f:
        num_of_nodes = int(next(f).split()[0])
        G = nx.parse_edgelist(f, nodetype=int)

    G.add_nodes_from(range(num_of_nodes))
    laplacianMatrix = nx.laplacian_matrix(G, nodelist=range(num_of_nodes)).toarray()

    orbits = get_orbits_from_file(filename)
    orbits_clustering = turn_membership_array_into_clustering_array(orbits, num_of_nodes, len(orbits))

    eigenvals, eigenvecs = np.linalg.eigh(laplacianMatrix)
    eigensolver_time = time.perf_counter()
    eigenvalues = np.real(eigenvals)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    round_digits = 12
    resolution = 1e10
    eigenvalues = np.round(eigenvalues, round_digits)
    lambda_fit = np.unique(np.sort(eigenvalues))

    Ti = np.arange(1 / np.max(lambda_fit), 1 / np.min(lambda_fit), 0.1)

    coeffs = np.zeros((num_of_nodes, len(Ti), num_of_nodes))

    for i in range(len(Ti)):
        exp_lambda = np.diag(np.exp( - eigenvals * Ti[i]))
        matrix_exponential = eigenvecs @ exp_lambda @ eigenvecs.T
        for j in range(num_of_nodes):
            coeffs[j][i] = np.sort(matrix_exponential[j])

    coeffs = np.trunc(coeffs * resolution) / resolution
    # print(coeffs, coeffs.shape)

    groups = defaultdict(list)
    # signatures = []
    # max_depth_of_BFS = 0

    # for root in range(len(coeffs)):
    #     dist = nx.single_source_shortest_path_length(G, root, cutoff=max_depth_of_BFS)
    #     sig = []

    #     for d in range(max_depth_of_BFS + 1):
    #         layer_nodes = [node for node, distance in dist.items() if distance == d]
    #         layer_coeffs = [tuple(coeffs[node]) for node in layer_nodes]
    #         layer_coeffs = np.sort(np.concatenate(layer_coeffs))
    #         sig.append(layer_coeffs)

    #     signatures.append(sig)

    # for i, sig in enumerate(signatures):
    #     key = tuple(tuple(layer) for layer in sig)
    #     groups[key].append(i)
            
    # equivalence_classes = list(groups.values())

    # print(coeffs, coeffs.shape)

    print(coeffs[0], coeffs[0].shape)
    for i in range(num_of_nodes):
        groups[coeffs[i].tobytes()].append(i)

    maximum_BFS_depth = 5

    for v in G.nodes():

        dist = nx.single_source_shortest_path_length(G, v, cutoff=maximum_BFS_depth)

        layers = [[] for _ in range(maximum_BFS_depth + 1)]

        for u, d in dist.items():
            layers[d].append(coeffs[u].tobytes())

        signature = tuple(tuple(sorted(layer)) for layer in layers)

        groups[signature].append(v)

    # for i, row in enumerate(coeffs):
    #     key = tuple(map(tuple, row))
    #     print(key)
    #     groups[key].append(i)
            
    equivalence_classes = list(groups.values())
    print(equivalence_classes)

    equivalence_classes_clustering = turn_membership_array_into_clustering_array(equivalence_classes, num_of_nodes, len(equivalence_classes))

    print("NMI (original): ", normalized_mutual_info_score(orbits_clustering[:136], equivalence_classes_clustering[:136]))
    print("NMI (all): ", normalized_mutual_info_score(orbits_clustering, equivalence_classes_clustering))

    # # print(np.unique(equivalence_classes_clustering[:136]))
    # print("Equivalence classes: ", len(np.unique(equivalence_classes_clustering)))

    # end_time = time.perf_counter()
    # print(f"Eigensolver Execution time: {(eigensolver_time - start_time):.6f} seconds")
    # print(f"Total Execution time: {(end_time - start_time):.6f} seconds")