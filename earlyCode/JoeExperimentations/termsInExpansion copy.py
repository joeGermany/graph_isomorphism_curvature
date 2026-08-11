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

    # filename  = "ags/ag2-08__00"
    filename = "CHH_cc(3-3)_198-2-0indexed"
    with open(filename) as f:
        num_of_nodes = int(next(f).split()[0])
        G = nx.parse_edgelist(f, nodetype=int)

    G.add_nodes_from(range(num_of_nodes))
    laplacianMatrix = nx.laplacian_matrix(G, nodelist=range(num_of_nodes)).toarray()

    # B = nx.incidence_matrix(G, oriented=True).tocsr()
    # edge_laplacian = B.T @ B
    # print(edge_laplacian.toarray())

    orbits = compute_orbits(G)
    print(orbits)
    orbits_clustering = turn_membership_array_into_clustering_array(orbits, num_of_nodes, len(orbits))

    eigenvals, eigenvecs = np.linalg.eigh(laplacianMatrix)
    idx = eigenvals.argsort()
    sorted_eigenvalues = eigenvals[idx]
    sorted_eigenvectors = eigenvecs[:, idx]

    sorted_eigenvectors = sorted_eigenvectors[:, 0:2]

    eigenvector_norms = np.zeros((num_of_nodes, num_of_nodes))
    for i in range(num_of_nodes):
        # this avoids the diagonal since it is meaningless
        for j in range(0, i):
            eigenvector_norms[i][j] = np.linalg.norm(sorted_eigenvectors[i, :] - sorted_eigenvectors[j, :])
            eigenvector_norms[j][i] = eigenvector_norms[i][j]

    print(eigenvector_norms[16][60])
    print(eigenvector_norms[16][82])
    print(eigenvector_norms[16][126])
    print(eigenvector_norms[60][82])
    print(eigenvector_norms[60][126])
    print(eigenvector_norms[60][148])
    print(eigenvector_norms[148][82])
    print(eigenvector_norms[148][126])

    print(np.sort(eigenvector_norms[16]) - np.sort(eigenvector_norms[60]))
    print(np.sort(eigenvector_norms[60]) - np.sort(eigenvector_norms[148]))
    print(np.sort(eigenvector_norms[16]) - np.sort(eigenvector_norms[148]))
    print(np.sort(eigenvector_norms[16]) - np.sort(eigenvector_norms[82]))
    print(np.sort(eigenvector_norms[148]) - np.sort(eigenvector_norms[82]))
    # print(np.sort(eigenvector_norms[16]) - np.sort(eigenvector_norms[82]))

    # 16, 82, 126, 60 /// 148
    resolution = 1e10

    Ti = np.arange(0.1, 10, 0.05)

    coeffs = np.zeros((num_of_nodes, len(Ti), num_of_nodes))

    for i in range(len(Ti)):
        exp_lambda = np.diag(np.exp( - eigenvals * Ti[i]))
        matrix_exponential = eigenvecs @ exp_lambda @ eigenvecs.T
        for j in range(num_of_nodes):
            coeffs[j][i] = np.sort(matrix_exponential[j])

    coeffs = np.trunc(coeffs * resolution) / resolution

    groups = defaultdict(list)

    maximum_BFS_depth = 5

    for v in G.nodes():
        dist = nx.single_source_shortest_path_length(G, v, cutoff=maximum_BFS_depth)
        neighbors = [[] for _ in range(maximum_BFS_depth + 1)]
        values_for_neighbors = [[] for _ in range(maximum_BFS_depth + 1)]

        for u, d in dist.items():
            neighbors[d].append(u)
            values_for_neighbors[d].append(eigenvector_norms[v][u])

        for i in range(1, len(neighbors)):
            idx = np.argsort(values_for_neighbors[i])
            neighbors[i] = np.asarray(neighbors[i])[idx]

        layers = [[] for _ in range(maximum_BFS_depth + 1)]

        for i in range(len(neighbors)):
            for j in range(len(neighbors[i])):
                layers[i].append(coeffs[neighbors[i][j]].tobytes())

        signature = tuple(tuple(sorted(layer)) for layer in layers)
        groups[signature].append(v)
            
    equivalence_classes = list(groups.values())
    equivalence_classes_clustering = turn_membership_array_into_clustering_array(equivalence_classes, num_of_nodes, len(equivalence_classes))
    print(equivalence_classes)

    print("NMI (all): ", normalized_mutual_info_score(orbits_clustering, equivalence_classes_clustering))