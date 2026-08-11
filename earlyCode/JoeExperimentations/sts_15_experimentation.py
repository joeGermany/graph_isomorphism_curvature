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
from tqdm import tqdm

if __name__ == "__main__":
    filename = "my_sts/sts-15_0indexed"

    with open(filename) as f:
        num_of_nodes = int(next(f).split()[0])
        G = nx.parse_edgelist(f, nodetype=int)

    G.add_nodes_from(range(num_of_nodes))
    num_of_nodes = G.number_of_nodes()

    laplacianMatrix = nx.laplacian_matrix(G, nodelist=range(num_of_nodes)).toarray()
    
    eigenvals, eigenvecs = np.linalg.eigh(laplacianMatrix)

    eigenvalues = np.real(eigenvals)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    round_digits = 9
    eigenvalues = np.round(eigenvalues, round_digits)
    lambda_fit = np.unique(np.sort(eigenvalues))
    print("lambda uno: ", lambda_fit[0])
    print(1 / np.max(lambda_fit), 1 / np.min(lambda_fit))

    resolution = 1e12

    Ti = np.arange(0.001, 10, 0.01)

    B = nx.incidence_matrix(G, oriented=True).toarray()
    edge_laplacian = B.T @ B
    print(edge_laplacian.shape)

    edge_indices_0 = np.where(B[0, :] != 0)[0]
    edge_indices_3 = np.where(B[3, :] != 0)[0]

    print(edge_indices_0)
    print(edge_indices_3)

    # connected_edge_rows = edge_laplacian[edge_indices_0, :]

    # edge_list = list(G.edges())
    # for idx, row in zip(edge_indices_0, connected_edge_rows):
    #     print(f"Edge {edge_list[idx]}: {row}")



    # arrays = np.empty((2, 315, 315), dtype=object)

    # all_powers = []

    # edge_L = edge_laplacian.astype(object)
    # current_power = edge_L.copy()
    # all_powers.append(current_power)

    # for k in tqdm(range(2, 316)):
    #     current_power = np.dot(current_power, edge_L)
    #     all_powers.append(current_power)
    #     # print(current_power[0])
    #     elements_0, counts_0 = np.unique(current_power[0], return_counts=True)
    #     elements_3, counts_3 = np.unique(current_power[3], return_counts=True)
    #     if np.array_equal(elements_0, elements_3) and np.array_equal(counts_0, counts_3):
    #         print(k, "is good")
    #     else:
    #         print(k, "is bad")
    #     # print(L_k.shape)
    #     # arrays[0][k] = L_k[0]
    #     # arrays[1][k] = L_k[3]

    # print(arrays)

    # # here
    # arrays = np.zeros((2, len(Ti), 26))
    # coeffs = np.zeros((num_of_nodes, len(Ti), num_of_nodes))
    # for i in range(len(Ti)):
    #     exp_lambda = np.diag(np.exp( - eigenvals * Ti[i]))
    #     matrix_exponential = eigenvecs @ exp_lambda @ eigenvecs.T
    #     arrays[0][i] = matrix_exponential[0]
    #     arrays[1][i] = matrix_exponential[3]
    #     for j in range(num_of_nodes):
    #         coeffs[j][i] = np.sort(matrix_exponential[j])


    # perm = [3,0,1,2,4,5,7,9,6,8,12,10,13,11,14,15,18,21,22,16,23,17,19,20,25,24]

    # print("yiha")
    # for i in range(len(Ti)):
    #     permuted_difference = arrays[0, i, :] - arrays[1, i, :][perm]

    #     check = permuted_difference[permuted_difference > 1e-14]
    #     if len(check) != 0:
    #         print(check)

    # print("now for powers...")
    # L = laplacianMatrix.astype(object)
    # for k in range(26):
    #     L_k = np.linalg.matrix_power(L, k)
    #     # print(L_k[0])
    #     print(np.array_equal(L_k[0], L_k[3][perm]))
