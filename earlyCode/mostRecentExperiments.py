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
from collections import Counter
from pyvis.network import Network

def build_cycle_with_cliques(m, attach_vertices, clique_sizes):

    if len(attach_vertices) != len(clique_sizes):
        raise ValueError("attach_vertices and clique_sizes must have same length")

    N = m + sum(clique_sizes)
    A = np.zeros((N, N), dtype=int)

    for i in range(m):
        j = (i + 1) % m
        A[i, j] = 1
        A[j, i] = 1

    next_node = m

    for v, size in zip(attach_vertices, clique_sizes):

        # New clique vertices
        clique_nodes = list(range(next_node, next_node + size))

        # Make them a clique
        for i in range(size):
            for j in range(i + 1, size):
                u = clique_nodes[i]
                w = clique_nodes[j]
                A[u, w] = 1
                A[w, u] = 1

        # Attach every clique vertex to the cycle vertex
        for u in clique_nodes:
            A[v, u] = 1
            A[u, v] = 1

        next_node += size

    return A

def build_wheel_with_cliques(m, attach_vertices, clique_sizes):

    if len(attach_vertices) != len(clique_sizes):
        raise ValueError("attach_vertices and clique_sizes must have same length")

    N = m + 1 + sum(clique_sizes)
    A = np.zeros((N, N), dtype=int)

    # cycle
    for i in range(m):
        j = (i + 1) % m
        A[i, j] = 1
        A[j, i] = 1

    # add universal vertex
    for i in range(m):
        A[i, m] = 1
        A[m, i] = 1

    next_node = m + 1

    for v, size in zip(attach_vertices, clique_sizes):

        # New clique vertices
        clique_nodes = list(range(next_node, next_node + size))

        # Make them a clique
        for i in range(size):
            for j in range(i + 1, size):
                u = clique_nodes[i]
                w = clique_nodes[j]
                A[u, w] = 1
                A[w, u] = 1

        # Attach every clique vertex to the cycle vertex
        for u in clique_nodes:
            A[v, u] = 1
            A[u, v] = 1

        next_node += size

    return A

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
    # filename = "CHH_cc(3-3)_198-2-0indexed"
    filename = "my_sts/sts-13_0indexed"
    with open(filename) as f:
        num_of_nodes = int(next(f).split()[0])
        G = nx.parse_edgelist(f, nodetype=int)

    G.add_nodes_from(range(num_of_nodes))

    # B = nx.incidence_matrix(G, oriented=True).tocsr()
    # edge_laplacian = B.T @ B
    # print(edge_laplacian.toarray())

    # num_of_nodes = 84
    # # G = nx.from_numpy_array(build_cycle_with_cliques(num_of_nodes, [0], [4]))
    # G = nx.from_numpy_array(build_wheel_with_cliques(num_of_nodes, [0], [40]))
    # num_of_nodes = G.number_of_nodes()
    # laplacianMatrix = nx.laplacian_matrix(G, nodelist=range(num_of_nodes)).toarray()

    # # filename  = "examples/example_1_0indexed"
    # filename = "examples/example_3"
    # # filename = "pg2-5/pg2-5__xx"
    # with open(filename) as f:
    #     num_of_nodes = int(next(f).split()[0])
    #     G = nx.parse_edgelist(f, nodetype=int)

    # G.add_nodes_from(range(num_of_nodes))
     
    
    # G = nx.frucht_graph()

    # G = nx.LCF_graph(12, [4, -4, 5, -5, 5, -5, 5, -5, 4, -4, 5, -5], 1)

    # folkman_edges = [
    #     (0, 1), (0, 3), (0, 4), (0, 5), (1, 2), (1, 11), (1, 12), (2, 3),
    #     (2, 12), (2, 19), (3, 4), (3, 13), (4, 5), (4, 14), (5, 6), (5, 15),
    #     (6, 7), (6, 13), (6, 15), (7, 8), (7, 12), (7, 18), (8, 9), (8, 14),
    #     (8, 19), (9, 10), (9, 13), (9, 15), (10, 11), (10, 14), (10, 18), 
    #     (11, 16), (11, 19), (12, 17), (13, 17), (14, 16), (15, 18), (16, 17), 
    #     (16, 18), (17, 19)
    # ]
    # herschel_edges = [
    #     (0, 1), (0, 7), (0, 9), (0, 10),
    #     (1, 2), (1, 8),
    #     (2, 3), (2, 9),
    #     (3, 4), (3, 8),
    #     (4, 5), (4, 9), (4, 10),
    #     (5, 6), (5, 8),
    #     (6, 7), (6, 10),
    #     (7, 8)
    # ]
    # G = nx.Graph()
    # G.add_edges_from(herschel_edges)
    num_of_nodes = G.number_of_nodes()

    laplacianMatrix = nx.laplacian_matrix(G, nodelist=range(num_of_nodes)).toarray()

    # B = nx.incidence_matrix(G, oriented=True).tocsr()
    # edge_laplacian = (B.T @ B).toarray()
    # print(edge_laplacian)
    
    orbits = compute_orbits(G)
    print("orbits", orbits)
    
    print("THE PRINTS IS PRINTING")
    # laplacianMatrix = laplacianMatrix.astype(np.uint64)

    # L = laplacianMatrix
    # n = L.shape[0]
    # # Compute powers
    # powers = []
    # M = np.eye(n)
    # for k in range(26):
    #     powers.append(M.copy())
    #     M = M @ L
    # i, j = 0, 3
    # # Column signatures across all powers
    # sig_i = np.array([[powers[k][i, l] for k in range(26)] for l in range(n)])
    # sig_j = np.array([[powers[k][j, l] for k in range(26)] for l in range(n)])
    # # Find permutation mapping row j -> row i
    # perm = []
    # used = np.zeros(n, dtype=bool)
    # for l in range(n):
    #     matches = np.where(
    #         np.all(np.isclose(sig_j, sig_i[l]), axis=1) & (~used)
    #     )[0]
    #     if len(matches) == 0:
    #         print("No permutation found")
    #         break
    #     perm.append(matches[0])
    #     used[matches[0]] = True
    # perm = np.array(perm)

    # counts = Counter(map(tuple, sig_j))
    # print([c for c in counts.values() if c > 1])

    # # Check all powers
    # for k in range(26):
    #     print(k, np.allclose(powers[k][j, perm], powers[k][i]))

    # perms = []

    # for t in [0.01, 0.1, 1, 5, 10]:
    #     H = expm(-L*t)
    #     perms.append(np.argsort(H[0], kind="stable"))

    # for p in perms:
    #     print(np.array_equal(p, perms[0]))

    # # check if the same permutation works for powers
    # for k in range(26):
    #     print(k, np.allclose(powers[k][3, P], powers[k][0]))

    # perm = []
    # used = np.zeros(L.shape[1], dtype=bool)

    # for x in L[0]:
    #     for j in range(L.shape[1]):
    #         if not used[j] and np.isclose(L[3, j], x):
    #             perm.append(j)
    #             used[j] = True
    #             break

    # perm = np.array(perm)
    # # print(perm)
    # print(Counter(L[0]))
    # # print(np.allclose(L[0], L[3][perm]))
    # current = np.eye(L.shape[0])
    # for k in range(1, 26):
    #     current = current @ L
    #     ok = np.allclose(current[3][perm], current[0])
    #     print(k, ok)

    # for i in range(0, 26):
    #     dicts = []
    #     perms = []
    #     matrix = np.linalg.matrix_power(laplacianMatrix, i)
    #     print("The power now is", i)
    #     for j in range(matrix.shape[0]):
    #         values, counts = np.unique(matrix[j], return_counts=True)
    #         counts_dict = dict(zip(values, counts))
    #         dicts.append(counts_dict)
    #         perms.append(np.argsort(matrix[j]))

    #     print(dicts)
    #     # print(perms)

    #     for j in range(1, len(dicts)):
    #         if dicts[0] != dicts[j]:
    #             print("BOOM BOOM")

        # matrix = np.linalg.matrix_power(nx.adjacency_matrix(G, nodelist=range(num_of_nodes)).toarray(), i)
        # matrix_2 = np.diagonal(np.linalg.matrix_power(edge_laplacian, i))
        # print(matrix[1][1], matrix[10][10])
        # print("HERE")
        # print(matrix_2)















    orbits_clustering = turn_membership_array_into_clustering_array(orbits, num_of_nodes, len(orbits))

    eigenvals, eigenvecs = np.linalg.eigh(laplacianMatrix)

    eigenvalues = np.real(eigenvals)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    round_digits = 9
    eigenvalues = np.round(eigenvalues, round_digits)
    lambda_fit = np.unique(np.sort(eigenvalues))
    print("lambda uno: ", lambda_fit[0])
    print(1 / np.max(lambda_fit), 1 / np.min(lambda_fit))

    # idx = eigenvals.argsort()
    # sorted_eigenvalues = eigenvals[idx]
    # sorted_eigenvectors = eigenvecs[:, idx]

    # sorted_eigenvectors = sorted_eigenvectors[:, :2]

    # eigenvector_norms = np.zeros((num_of_nodes, num_of_nodes))
    # for i in range(num_of_nodes):
    #     # this avoids the diagonal since it is meaningless
    #     for j in range(0, i):
    #         eigenvector_norms[i][j] = np.linalg.norm(sorted_eigenvectors[i, :] - sorted_eigenvectors[j, :])
    #         eigenvector_norms[j][i] = eigenvector_norms[i][j]

    resolution = 1e9

    Ti = np.arange(0.001, 0.7, 0.01)

    # coeffs = np.zeros((num_of_nodes, len(Ti)))
    # for i in range(len(Ti)):
    #     exp_lambda = np.diag(np.exp( - eigenvals * Ti[i]))
    #     matrix_exponential = eigenvecs @ exp_lambda @ eigenvecs.T
    #     # print(matrix_exponential.diagonal())
    #     for j in range(num_of_nodes):
    #         coeffs[j][i] = matrix_exponential[j][j]

    # here
    arrays = np.zeros((2, len(Ti), 26))
    coeffs = np.zeros((num_of_nodes, len(Ti), num_of_nodes))
    for i in range(len(Ti)):
        exp_lambda = np.diag(np.exp( - eigenvals * Ti[i]))
        matrix_exponential = eigenvecs @ exp_lambda @ eigenvecs.T
        # print(matrix_exponential.shape)
        # arrays[0].append(matrix_exponential[0][0])
        # arrays[1].append(matrix_exponential[3][3])
        arrays[0][i] = matrix_exponential[0]
        arrays[1][i] = matrix_exponential[3]
        for j in range(num_of_nodes):
            coeffs[j][i] = np.sort(matrix_exponential[j])

    # print(arrays[0, 120, :])
    # print(arrays[1, 120, :])


    # IMPORTANT PART: TO PROVE THAT BFS SIGNATURES AREN'T ENOUGH FOR STS-13: 
    dist_0 = nx.single_source_shortest_path_length(G, 0, cutoff=10)
    dist_3 = nx.single_source_shortest_path_length(G, 3, cutoff=10)
    dist_6 = nx.single_source_shortest_path_length(G, 6, cutoff=10)

    # print(dist_0) 
    # print(dist_3)

    # layers = [[] for _ in range(10 + 1)]
    bfs_for_node_0 = [[] for _ in range(3)]
    bfs_for_node_3 = [[] for _ in range(3)]
    bfs_for_node_6 = [[] for _ in range(3)]

    for u, d in dist_0.items():
        bfs_for_node_0[d].append(u)

    for u, d in dist_3.items():
        bfs_for_node_3[d].append(u)

    for u, d in dist_6.items():
        bfs_for_node_6[d].append(u)

    for i in range(len(bfs_for_node_0)):
        bfs_for_node_0[i] = np.sort(bfs_for_node_0[i])
        bfs_for_node_3[i] = np.sort(bfs_for_node_3[i])
        bfs_for_node_6[i] = np.sort(bfs_for_node_6[i])

    print(bfs_for_node_0)
    print(bfs_for_node_3)
    print(bfs_for_node_6)

    # time_value = 0.1
    # time_index = int(time_value / 0.01)
    # print(time_index)
    # print(arrays[0, time_index, :], arrays[1, time_index, :])

    # print("now for bfs signatures for time = 0.1...")
    # sig_0 = [[] for _ in range(3)]
    # sig_3 = [[] for _ in range(3)]

    # for i in range(len(bfs_for_node_0)):
    #     for j in range(len(bfs_for_node_0[i])):
    #         sig_0[i].append(arrays[0, time_index, bfs_for_node_0[i][j]])
    #         sig_3[i].append(arrays[1, time_index, bfs_for_node_3[i][j]])
    
    # print(sig_0)
    # print(sig_3)



    # Investigating the Edge Laplacian:
    B = nx.incidence_matrix(G, oriented=True).toarray()
    # edge_laplacian = (B.T @ B).toarray()

    print(B[0])


    # Specifically for time index = 120 

        # layers[d].append(coeffs[u].tobytes())

    # print(arrays[0, 120, :] - arrays[1, 120, :][perm])

    perm = [3,0,1,2,4,5,7,9,6,8,12,10,13,11,14,15,18,21,22,16,23,17,19,20,25,24]

    print("yiha")
    for i in range(len(Ti)):
        permuted_difference = arrays[0, i, :] - arrays[1, i, :][perm]

        check = permuted_difference[permuted_difference > 1e-14]
        if len(check) != 0:
            print(check)

    print("now for powers...")
    # laplacianMatrix = laplacianMatrix.astype(np.uint64)
    L = laplacianMatrix.astype(object)
    for k in range(26):
        L_k = np.linalg.matrix_power(L, k)
        # print(L_k[0])
        print(np.array_equal(L_k[0], L_k[3][perm]))

    for i in range(0, 26):
        plt.plot(Ti, arrays[0, :, i], color="red", linestyle=":", linewidth=3)
        plt.plot(Ti, arrays[1, :, i], color="blue", linestyle="--", linewidth=1)

    # The diagonals [0,0] and [3,3] of the heat kernel
    # plt.plot(arrays[0, :, 0], color="red", linestyle=":", linewidth=3)
    # plt.plot(arrays[1, :, 3], color="blue", linestyle="--", linewidth=1)

    print(arrays[0, ])

    plt.show()

    coeffs = np.trunc(coeffs * resolution) / resolution

    groups = defaultdict(list)

    # for i, row in enumerate(coeffs): 
    #     groups[tuple(row)].append(i)

    # here
    for i, row in enumerate(coeffs):
        key = tuple(map(tuple, row))
        groups[key].append(i)

    # maximum_BFS_depth = 0
    # for v in G.nodes():
    #     dist = nx.single_source_shortest_path_length(G, v, cutoff=maximum_BFS_depth)
    #     layers = [[] for _ in range(maximum_BFS_depth + 1)]
    #     for u, d in dist.items():
    #         layers[d].append(coeffs[u].tobytes())
    #     signature = tuple(tuple(sorted(layer)) for layer in layers)
    #     groups[signature].append(v)
            
    equivalence_classes = list(groups.values())
    equivalence_classes_clustering = turn_membership_array_into_clustering_array(equivalence_classes, num_of_nodes, len(equivalence_classes))
    print("equivalence classes", equivalence_classes)
    print(len(equivalence_classes))

    # print(eigenvecs[:, 0], eigenvecs[:, 1])
    print(np.where(eigenvecs[:, 5] > 0))

    print("NMI (all): ", normalized_mutual_info_score(orbits_clustering, equivalence_classes_clustering))




    for node in G.nodes():
        G.nodes[node]['label'] = f"Node_{node}"

    nx.write_graphml(G, "sts-13.graphml")


    # net = Network(
    #     height="700px", 
    #     width="100%", 
    #     bgcolor="#1a1a1a",     # Sleek dark background
    #     font_color="#ffffff"   # White text labels
    # )

    # # 2. Load your structure
    # net.from_nx(G)

    # # 3. Apply uniform styling
    # for node in net.nodes:
    #     node_id = node['id']
    #     degree = G.degree(node_id)
        
    #     # Scale appearance dynamically
    #     node['size'] = 15 + (degree * 2)      
    #     node['color'] = '#00adb5' if degree > 2 else '#393e46' 
    #     node['borderWidth'] = 2
    #     node['borderColor'] = '#eeeeee'
    #     node['title'] = f"Node {node_id}<br>Connections: {degree}" 

    # # 4. Tweak physics layout options
    # net.set_options("""
    # var options = {
    # "physics": {
    #     "barnesHut": {
    #     "gravitationalConstant": -8000,
    #     "centralGravity": 0.3,
    #     "springLength": 95,
    #     "springConstant": 0.04
    #     },
    #     "minVelocity": 0.75
    # }
    # }
    # """)

    # # 5. Save the file and automatically open it in your web browser
    # net.show("my_beautiful_graph.html", notebook=False)

    # plt.figure(figsize=(8, 6))
    # pos = nx.spring_layout(G, seed=42) 
    # nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=600, edge_color="gray", font_size=12, font_weight="bold")
    # plt.show()