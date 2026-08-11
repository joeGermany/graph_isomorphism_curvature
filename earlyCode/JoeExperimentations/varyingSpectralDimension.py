import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import scipy.linalg as la
from scipy.linalg import expm
import csv
import networkx as nx
from collections import defaultdict
import mpmath as mp
import sys
import ast
from sklearn.metrics.cluster import normalized_mutual_info_score
import time

def create_chebyshev(a, b, n):
    k = np.arange(1, n + 2)  # 1, 2, ..., n+1
    angles = (2 * k - 1) * np.pi / (2 * (n + 1))
    x = (a + b) / 2 + (b - a) / 2 * np.cos(angles)
    return np.sort(x)

def curvature_dim_given(G, laplacianMatrix, dim, orderinexpan, start_coefficient_for_curvature=1, end_coefficient_for_curvature=1, rounding_array=[1e6], bfs_signatures=False, max_depth_of_BFS=0):
    # Numpy solver
    eigenvals, eigenvecs = np.linalg.eigh(laplacianMatrix)
    eigensolver_time = time.perf_counter()
    eigenvalues = np.real(eigenvals)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # Scipy solver
    # eigenvalues = la.eigvalsh(laplacianMatrix, lower=False)
    # eigenvalues[np.isclose(eigenvalues, 0, atol=1e-15)] = 0.0
    # eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # mpmath solver (insanely slow)
    # mp.mp.dps = 15 # decimal places
    # laplacianMatrix = mp.matrix(laplacianMatrix.tolist())
    # eigenvalues, _ = mp.eigsy(laplacianMatrix)
    # eigenvalues = [ev for ev in eigenvalues if ev > mp.mpf('1e-10')]

    N = len(eigenvalues)

    round_digits = 12
    eigenvalues = np.round(eigenvalues, round_digits)
    lambda_fit = np.unique(np.sort(eigenvalues))

    print("dimension ", dim)

    Ti = np.arange(1 / np.max(lambda_fit), 1 / np.min(lambda_fit), 0.001)

    X = np.zeros((len(Ti), orderinexpan))

    for i in range(0, orderinexpan):
        X[:, i] = (Ti ** i) / (4 * np.pi * Ti)**(dim / 2)

    K1diag = np.zeros((laplacianMatrix.shape[0], len(Ti)))

    for i in range(len(Ti)):
        # Ker1 = expm(- laplacianMatrix * Ti[i])
        # K1diag[:, i] = np.diag(Ker1)
        exp_lambda = np.exp( - eigenvals * Ti[i])
        K1diag[:, i] = np.sum((eigenvecs**2) * exp_lambda, axis=1)

    X_pinv = np.linalg.pinv(X)

    vr = np.zeros(laplacianMatrix.shape[0])
    full_coeffs = np.zeros((laplacianMatrix.shape[0], orderinexpan))

    for j in range(laplacianMatrix.shape[0]):
        Y = K1diag[j, :]
        coeffs = X_pinv @ Y
        full_coeffs[j, :] = coeffs

    full_coeffs = full_coeffs[:, start_coefficient_for_curvature:end_coefficient_for_curvature]
    full_coeffs_unrounded = full_coeffs.copy()
    for i in range(end_coefficient_for_curvature - start_coefficient_for_curvature):
        full_coeffs[:, i] = np.trunc(full_coeffs[:, i] * rounding_array[i]) / rounding_array[i]
    # full_coeffs = np.round(full_coeffs, decimals=9)

    # print(full_coeffs)

    groups = defaultdict(list)
    if bfs_signatures == False:
        # print(np.unique(full_coeffs, axis=0))
        # print(len(np.unique(full_coeffs, axis=0)))

        for i, row in enumerate(full_coeffs):
            groups[tuple(row)].append(i)

    else:
        signatures = []

        for root in range(len(full_coeffs)):
            dist = nx.single_source_shortest_path_length(G, root, cutoff=max_depth_of_BFS)
            sig = []

            for d in range(max_depth_of_BFS + 1):
                layer_nodes = [node for node, distance in dist.items() if distance == d]
                layer_coeffs = [tuple(full_coeffs[node]) for node in layer_nodes]
                layer_coeffs.sort()
                sig.append(layer_coeffs)

            signatures.append(sig)

        # print(np.array(signatures[66][1]) - np.array(signatures[73][1]))
        # print("sigs", signatures[66] - signatures[73])

        for i, sig in enumerate(signatures):
            key = tuple(tuple(layer) for layer in sig)
            groups[key].append(i)

    equivalence_classes = list(groups.values())

    # print(equivalence_classes)

    return equivalence_classes, eigensolver_time

def get_orbits_from_file(original_filename):
    orbits_filename = original_filename + "_orbits.txt"
    orbits = []

    with open(orbits_filename) as f:
        num_of_orbits = int(next(f).split()[0])
        for i in range(num_of_orbits):
            orbits.append(ast.literal_eval(next(f)))
    
    return orbits

# sorry for the weird function name
def turn_membership_array_into_clustering_array(array, num_of_nodes, num_of_classes):
    clustering = np.zeros(num_of_nodes)

    for i in range(len(array)):
        clustering[array[i]] = i

    return clustering

if __name__ == "__main__":
    start_time = time.perf_counter()
    
    # original = "ags/ag2-08-original"
    # with open(original) as f:
    #     first_line = next(f).split()
    #     num_of_nodes = int(first_line[0])
    #     num_of_edges = int(first_line[1])
    #     G = nx.parse_edgelist(f, nodetype=int)

    # G.add_nodes_from(range(num_of_nodes))
    # # G = subdivide_edges(G, 2, connection_rule=connect_path)
    # k=1
    # G = subdivide_every_edge(G, k)
    # G = attach_clique(G, 64, 3, True)
    # G = attach_clique(G, 65, 4, True)
    # G = attach_clique(G, 9, 5, True)
    # print(G.number_of_edges())
    # num_of_nodes = G.number_of_nodes()
    # L = nx.laplacian_matrix(G).toarray()
    # print(L)

    # filename = "pp-5-1/pp-5-1__02"
    filename = "ags/ag2-08__02"
    with open(filename) as f:
        num_of_nodes = int(next(f).split()[0])
        G = nx.parse_edgelist(f, nodetype=int)

    G.add_nodes_from(range(num_of_nodes))
    L = nx.laplacian_matrix(G, nodelist=range(num_of_nodes)).toarray()

    # print(np.array_equal(L, L2))
    # indices = np.where(L != L2)
    # print("Indices of differences:", indices)

    # print(np.nonzero(L[1]))
    # print(L2[1][146])

    orbits = get_orbits_from_file(filename)
    orbits_clustering = turn_membership_array_into_clustering_array(orbits, num_of_nodes, len(orbits))
    # for i in range(len(orbits)):
    #     print(len(orbits[i]))
    orderinexpan = np.max(np.diag(L)).astype(int)
    start_coefficient_for_curvature = 1
    end_coefficient_for_curvature = 6
    rounding_array = [1e6] * (end_coefficient_for_curvature - start_coefficient_for_curvature)
    bfs_signatures = False
    max_depth_of_BFS = 3

    dims = np.arange(0, 1, 0.01)
    nmis = []
    unique_classes_for_originals = []
    for i in range(len(dims)):
        equivalence_classes, eigensolver_time = curvature_dim_given(G, L, dims[i], orderinexpan, 
            start_coefficient_for_curvature=start_coefficient_for_curvature, 
            end_coefficient_for_curvature=end_coefficient_for_curvature, 
            rounding_array=rounding_array, 
            bfs_signatures=bfs_signatures, 
            max_depth_of_BFS=max_depth_of_BFS
        )
        equivalence_classes_clustering = turn_membership_array_into_clustering_array(equivalence_classes, num_of_nodes, len(equivalence_classes))
        # print(equivalence_classes[6])

        # # put [:136] if you only want to compare the original vertices for the ag's
        # print("NMI = ", normalized_mutual_info_score(orbits_clustering[:136], equivalence_classes_clustering[:136]))
        # print("uniques ", len(np.unique(equivalence_classes_clustering[:136])))

        nmis.append(normalized_mutual_info_score(orbits_clustering[:136], equivalence_classes_clustering[:136]))
        unique_classes_for_originals.append(len(np.unique(equivalence_classes_clustering[:136])))

    print("NMIs: ", nmis)
    print("Unique Classes: ", unique_classes_for_originals)

    end_time = time.perf_counter()
    print(f"Eigensolver Execution time: {(eigensolver_time - start_time):.6f} seconds")
    print(f"Total Execution time: {(end_time - start_time):.6f} seconds")