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

def curvature(G, laplacianMatrix, orderinexpan, start_coefficient_for_curvature=1, end_coefficient_for_curvature=1, rounding_array=[1e6], bfs_signatures=False, max_depth_of_BFS=0):
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

    round_digits = 9
    eigenvalues = np.round(eigenvalues, round_digits)
    lambda_fit = np.unique(np.sort(eigenvalues))
    N_fit = np.zeros(len(lambda_fit))

    for i in range(len(lambda_fit)):
        c = np.where(lambda_fit < lambda_fit[i])[0]
        N_fit[i] = (len(c)+1) / len(lambda_fit)

    mask = N_fit > 0
    x = np.log(lambda_fit[mask])
    y = np.log(N_fit[mask])

    # plt.scatter(lambda_fit[mask], N_fit[mask])

    # Plot log-log graph
    # plt.scatter(x, y, s=10)
    # plt.xlabel("log(lambda)")
    # plt.ylabel("log(N_fit)")
    # plt.show()

    lowpoints = 4
    nbpoints = len(x)
    rsq = np.zeros(nbpoints - lowpoints + 1)
    slope = []

    for i in range(lowpoints, nbpoints + 1):
        fit = linregress(x[:i], y[:i])
        rsq[i - lowpoints] = fit.rvalue**2
        slope.append(2*fit.slope)
    
    best = np.argmax(rsq)
    best_n = lowpoints + best

    fit = linregress(x[:best_n], y[:best_n])

    slope = fit.slope
    dim = 2*slope
    print("dimension ", dim)

    lambda_critical = np.exp(x[best_n - 1])

    Ti = np.arange(1 / np.max(lambda_fit), 1 / np.min(lambda_fit), 0.001)
    print(1 / np.max(lambda_fit), 1 / lambda_critical)

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
    for i in range(end_coefficient_for_curvature - start_coefficient_for_curvature):
        full_coeffs[:, i] = np.trunc(full_coeffs[:, i] * rounding_array[i]) / rounding_array[i]
    # full_coeffs = np.round(full_coeffs, decimals=9)

    print(full_coeffs)

    groups = defaultdict(list)
    if bfs_signatures == False:
        print(np.unique(full_coeffs, axis=0))
        print(len(np.unique(full_coeffs, axis=0)))

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
        
        for i, sig in enumerate(signatures):
            key = tuple(tuple(layer) for layer in sig)
            groups[key].append(i)
            
    equivalence_classes = list(groups.values())

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

    filename = "ags/ag2-08__02"
    with open(filename) as f:
        num_of_nodes = int(next(f).split()[0])
        G = nx.parse_edgelist(f, nodetype=int)

    G.add_nodes_from(range(num_of_nodes))
    L = nx.laplacian_matrix(G, nodelist=range(num_of_nodes)).toarray()

    orbits = get_orbits_from_file(filename)
    orbits_clustering = turn_membership_array_into_clustering_array(orbits, num_of_nodes, len(orbits))

    orderinexpan = 6 #np.max(np.diag(laplacianMatrix)).astype(int)
    start_coefficient_for_curvature = 1
    end_coefficient_for_curvature = 6
    rounding_array = [1e6] * (end_coefficient_for_curvature - start_coefficient_for_curvature)
    bfs_signatures = True
    max_depth_of_BFS = 5
    equivalence_classes, eigensolver_time = curvature(G, L, orderinexpan, 
        start_coefficient_for_curvature=start_coefficient_for_curvature, 
        end_coefficient_for_curvature=end_coefficient_for_curvature, 
        rounding_array=rounding_array, 
        bfs_signatures=bfs_signatures, 
        max_depth_of_BFS=max_depth_of_BFS
    )
    equivalence_classes_clustering = turn_membership_array_into_clustering_array(equivalence_classes, num_of_nodes, len(equivalence_classes))

    # put [:136] if you only want to compare the original vertices for the ag's
    print(normalized_mutual_info_score(orbits_clustering[:136], equivalence_classes_clustering[:136]))
    print(len(np.unique(equivalence_classes_clustering[:136])))

    end_time = time.perf_counter()
    print(f"Eigensolver Execution time: {(eigensolver_time - start_time):.6f} seconds")
    print(f"Total Execution time: {(end_time - start_time):.6f} seconds")