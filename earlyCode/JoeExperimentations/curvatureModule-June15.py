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

    round_digits = 12
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
    plt.scatter(x, y, s=10)
    plt.xlabel("log(lambda)")
    plt.ylabel("log(N_fit)")
    plt.show()

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

    print("best", best_n)

    fit = linregress(x[:best_n], y[:best_n])

    slope = fit.slope
    dim = 2*slope
    print("dimension ", dim)

    lambda_critical = np.exp(x[best_n - 1])

    Ti = np.arange(1 / np.max(lambda_fit), 1 / np.min(lambda_fit), 0.001)
    # Ti = np.arange(1 / np.max(lambda_fit), 7.05, 0.001)
    # Ti = np.arange(0.2, 1.529, 0.0005)
    # Ti = create_chebyshev(1 / np.max(lambda_fit), 7, 2000)
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
    full_coeffs_unrounded = full_coeffs.copy()
    for i in range(end_coefficient_for_curvature - start_coefficient_for_curvature):
        full_coeffs[:, i] = np.trunc(full_coeffs[:, i] * rounding_array[i]) / rounding_array[i]
    # full_coeffs = np.round(full_coeffs, decimals=9)

    print(full_coeffs)

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

        for i, sig in enumerate(signatures):
            key = tuple(tuple(layer) for layer in sig)
            groups[key].append(i)

    equivalence_classes = list(groups.values())

    print(equivalence_classes)

    # num = len(equivalence_classes[6])
    # print("num: ", num)
    # differences_array = np.zeros((num, num, 5))
    # print(differences_array.shape)
    # for i in range(num):
    #     for j in range(num):
    #         differences_array[i][j] = full_coeffs_unrounded[equivalence_classes[6][i]] - full_coeffs_unrounded[equivalence_classes[6][j]]

    # norms = np.linalg.norm(differences_array, axis=2)
    # print(norms[0][1], norms[0][7])
    # print(np.max(differences_array))
    # print("same", differences_array[0][1])
    # print("outside", differences_array[0][7])

    return equivalence_classes, eigensolver_time

# For subdividing edges:
## For connecting adjacent nodes 
# def connect_path(H, u, v, new_nodes):
#     vertices = [u] + new_nodes + [v]
#     print(vertices)

#     for i in range(len(vertices) - 1):
#         H.add_edge(vertices[i], vertices[i + 1])

# def connect_path_power_law_esque(H, u, v, new_nodes):
#     vertices = [u] + new_nodes + [v]

#     for i in range(len(vertices) - 1):
#         H.add_edge(vertices[i], vertices[i + 1])

#     for w in new_nodes:
#         H.add_edge(u, w)
#         H.add_edge(v, w)

# def subdivide_edges(G, m, connection_rule=connect_path):
#     H = nx.Graph()
#     H.add_nodes_from(G.nodes())
#     next_node = max(G.nodes()) + 1

#     for u, v in G.edges():
#         new_nodes = list(range(next_node, next_node + m))
#         next_node += m
#         H.add_nodes_from(new_nodes)
#         connection_rule(H, u, v, new_nodes)

#     return H

def subdivide_every_edge(G, k):
    n = G.number_of_nodes()

    out = nx.Graph()
    out.add_nodes_from(range(n))

    # Collect original edges canonically (a < b)
    edges = []
    for u, v in G.edges():
        if u == v:
            continue

        a = min(u, v)
        b = max(u, v)
        edges.append((a, b))

    edges.sort()

    if k == 0:
        out.add_edges_from(edges)
        return out

    next_new = n

    for u, v in edges:
        prev = u

        for _ in range(k):
            x = next_new
            next_new += 1
            out.add_node(x)
            out.add_edge(prev, x)
            prev = x

        out.add_edge(prev, v)

    return out

import networkx as nx


def subdivide_every_edge_power_law_esque(G, k):
    n = G.number_of_nodes()

    out = nx.Graph()
    out.add_nodes_from(range(n))

    edges = []
    for u, v in G.edges():
        if u == v:
            continue

        a = min(u, v)
        b = max(u, v)
        edges.append((a, b))

    edges.sort()

    if k == 0:
        out.add_edges_from(edges)
        return out

    next_new = n

    for u, v in edges:
        prev = u
        new_nodes = []

        for _ in range(k):
            x = next_new
            next_new += 1

            out.add_node(x)
            new_nodes.append(x)

            out.add_edge(prev, x)
            prev = x

        out.add_edge(prev, v)

        # connect u and v to every inserted node
        for x in new_nodes:
            # print(new_nodes, u, v)
            out.add_edge(u, x)
            out.add_edge(v, x)

    return out

import networkx as nx


def attach_clique(G, attach_vertex, clique_size, connect_attach_to_all):
    if clique_size < 0:
        raise ValueError("clique_size must be >= 0")

    if attach_vertex not in G:
        raise ValueError("attach_vertex is not in the graph")

    n = G.number_of_nodes()

    out = nx.Graph()
    out.add_nodes_from(G.nodes())
    out.add_edges_from(G.edges())

    # Add new clique vertices
    new_verts = []
    for i in range(clique_size):
        v_new = n + i
        new_verts.append(v_new)
        out.add_node(v_new)

    # Add clique internal edges
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            out.add_edge(new_verts[i], new_verts[j])

    # Connect clique to attach_vertex
    if connect_attach_to_all:
        for v_new in new_verts:
            out.add_edge(attach_vertex, v_new)
    elif clique_size > 0:
        out.add_edge(attach_vertex, new_verts[0])

    return out

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
    # filename = "ags/ag2-08__00"
    filename = "CHH_cc(3-3)_198-2-0indexed"
    # filename = "smallGraphsFromAmer/graph-01"

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
    start_coefficient_for_curvature = 0
    end_coefficient_for_curvature = 6
    rounding_array = [1e6] * (end_coefficient_for_curvature - start_coefficient_for_curvature)
    bfs_signatures = False
    max_depth_of_BFS = 10
    equivalence_classes, eigensolver_time = curvature(G, L, orderinexpan, 
        start_coefficient_for_curvature=start_coefficient_for_curvature, 
        end_coefficient_for_curvature=end_coefficient_for_curvature, 
        rounding_array=rounding_array, 
        bfs_signatures=bfs_signatures, 
        max_depth_of_BFS=max_depth_of_BFS
    )
    equivalence_classes_clustering = turn_membership_array_into_clustering_array(equivalence_classes, num_of_nodes, len(equivalence_classes))
    # print(equivalence_classes[6])

    # # # put [:136] if you only want to compare the original vertices for the ag's
    print("NMI = ", normalized_mutual_info_score(orbits_clustering, equivalence_classes_clustering))
    print("uniques ", len(np.unique(equivalence_classes_clustering)))

    # end_time = time.perf_counter()
    # print(f"Eigensolver Execution time: {(eigensolver_time - start_time):.6f} seconds")
    # print(f"Total Execution time: {(end_time - start_time):.6f} seconds")