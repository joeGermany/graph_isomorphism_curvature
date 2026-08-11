import networkx as nx
import random
import numpy as np

def generate_isomorphic_graphs(n=10, p=0.3, seed=None):
    rng = random.Random(seed)

    G = nx.gnp_random_graph(n, p, seed=seed)

    nodes = list(G.nodes())
    permuted = nodes.copy()
    rng.shuffle(permuted)

    mapping = dict(zip(nodes, permuted))
    H = nx.relabel_nodes(G, mapping)

    return G, H, mapping

G, H, mapping = generate_isomorphic_graphs(n=12, p=0.25, seed=42)

A_G = nx.to_numpy_array(G, nodelist=sorted(G.nodes()), dtype=int)
A_H = nx.to_numpy_array(H, nodelist=sorted(H.nodes()), dtype=int)

print(A_G)
print(A_H)

for u in sorted(mapping):
    print(f"{u} -> {mapping[u]}")



laplacianMatrix_1 = nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).toarray()
laplacianMatrix_2 = nx.laplacian_matrix(H, nodelist=sorted(H.nodes())).toarray()


eigenvals_1, eigenvecs_1 = np.linalg.eigh(laplacianMatrix_1)
idx = eigenvals_1.argsort()
sorted_eigenvalues_1 = eigenvals_1[idx]
sorted_eigenvectors_1 = eigenvecs_1[:, idx]


eigenvals_2, eigenvecs_2 = np.linalg.eigh(laplacianMatrix_2)
idx = eigenvals_2.argsort()
sorted_eigenvalues_2 = eigenvals_2[idx]
sorted_eigenvectors_2 = eigenvecs_2[:, idx]

indx = [7,5,2,8,9,6,11,3,4,0,1,10]
eigenvecs_1 = eigenvecs_1[indx, :]
# print("first ", eigenvecs_1[indx, :])
# print("second ", eigenvecs_2[7, :])
print(eigenvecs_2 - eigenvecs_1)
