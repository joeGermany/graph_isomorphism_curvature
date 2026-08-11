import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from sklearn.metrics.cluster import normalized_mutual_info_score
import pynauty

# G = nx.frucht_graph()
# filename  = "ags/ag2-08-original"
filename = "examples/example_1"
# filename = "CHH_cc(3-3)_198-2-0indexed"
with open(filename) as f:
    num_of_nodes = int(next(f).split()[0])
    G = nx.parse_edgelist(f, nodetype=int)

G.add_nodes_from(range(num_of_nodes))


matcher = nx.isomorphism.GraphMatcher(G, G)
automorphisms = list(matcher.isomorphisms_iter())

print(f"Found {len(automorphisms)} structure-preserving permutations:")
for mapping in automorphisms:
    print(mapping)

chosen_permutation = automorphisms[3]
# chosen_permutation = dict(sorted(chosen_permutation.items()))

permuted_G = nx.relabel_nodes(G, chosen_permutation, copy=True)

laplacianMatrix_1 = nx.laplacian_matrix(G, nodelist=range(num_of_nodes)).toarray()
laplacianMatrix_2 = nx.laplacian_matrix(permuted_G, nodelist=range(num_of_nodes)).toarray()

# print(list(permuted_G.nodes()))

eigenvals_1, eigenvecs_1 = np.linalg.eigh(laplacianMatrix_1)
idx = eigenvals_1.argsort()
sorted_eigenvalues_1 = eigenvals_1[idx]
sorted_eigenvectors_1 = eigenvecs_1[:, idx]


eigenvals_2, eigenvecs_2 = np.linalg.eigh(laplacianMatrix_2)
idx = eigenvals_2.argsort()
sorted_eigenvalues_2 = eigenvals_2[idx]
sorted_eigenvectors_2 = eigenvecs_2[:, idx]

print(laplacianMatrix_1 - laplacianMatrix_2)

print("first ", eigenvecs_1[7, :])
print("second ", eigenvecs_2[11, :])
print(eigenvecs_2[11, :] - eigenvecs_1[7, :])

print(chosen_permutation)