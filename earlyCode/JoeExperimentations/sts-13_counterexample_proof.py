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

if __name__ == "__main__":
    filename = "my_sts/sts-13_0indexed"

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

    resolution = 1e9

    Ti = np.arange(0.001, 10, 0.01)

    # here
    arrays = np.zeros((2, len(Ti), 26))
    coeffs = np.zeros((num_of_nodes, len(Ti), num_of_nodes))
    for i in range(len(Ti)):
        exp_lambda = np.diag(np.exp( - eigenvals * Ti[i]))
        matrix_exponential = eigenvecs @ exp_lambda @ eigenvecs.T
        arrays[0][i] = matrix_exponential[0]
        arrays[1][i] = matrix_exponential[3]
        for j in range(num_of_nodes):
            coeffs[j][i] = np.sort(matrix_exponential[j])


    perm = [3,0,1,2,4,5,7,9,6,8,12,10,13,11,14,15,18,21,22,16,23,17,19,20,25,24]

    print("yiha")
    for i in range(len(Ti)):
        permuted_difference = arrays[0, i, :] - arrays[1, i, :][perm]

        check = permuted_difference[permuted_difference > 1e-14]
        if len(check) != 0:
            print(check)

    print("now for powers...")
    L = laplacianMatrix.astype(object)
    for k in range(26):
        L_k = np.linalg.matrix_power(L, k)
        # print(L_k[0])
        print(np.array_equal(L_k[0], L_k[3][perm]))
