import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import scipy.linalg as la
from scipy.linalg import expm
import csv
import networkx as nx
from collections import defaultdict

def curvature(laplacianMatrix):
    # Numpy solver
    eigenvalues = np.linalg.eigvalsh(laplacianMatrix)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # Scipy solver
    # eigenvalues = la.eigvalsh(laplacianMatrix, lower=False)
    # eigenvalues[np.isclose(eigenvalues, 0, atol=1e-15)] = 0.0
    # eigenvalues = eigenvalues[eigenvalues > 1e-10]

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

    fit = linregress(x[:best_n], y[:best_n])

    slope = fit.slope
    dim = 2*slope
    print("dimension ", dim)

    lambda_critical = x[best_n - 1]

    Ti = np.arange(1 / np.max(lambda_fit), 1 / np.exp(lambda_critical), 0.001)

    # max degree of whole graph
    orderinexpan = np.max(np.diag(laplacianMatrix)).astype(int)

    X = np.zeros((len(Ti), orderinexpan))

    for i in range(0, orderinexpan):
        X[:, i] = (Ti ** i) / (4 * np.pi * Ti)**(dim / 2)

    K1diag = np.zeros((laplacianMatrix.shape[0], len(Ti)))

    for i in range(len(Ti)):
        Ker1 = expm(- laplacianMatrix * Ti[i])
        K1diag[:, i] = np.diag(Ker1)

    X_pinv = np.linalg.pinv(X)

    vr = np.zeros(laplacianMatrix.shape[0])

    for j in range(laplacianMatrix.shape[0]):
        Y = K1diag[j, :]
        coeffs = X_pinv @ Y

        vr[j] = coeffs[1]

    resolution = 1e8
    vr = np.trunc(vr * resolution) / resolution

    print(np.unique(vr))
    print(len(np.unique(vr)))

    # To get equivalence classes based on u_1:
    # groups = defaultdict(list)
    # for i, value in enumerate(vr):
    #     groups[value].append(i)

    # for key, values in groups.items():
    #     print(values)

    return dim, vr

G = nx.read_edgelist("ags/ag2-08__01", nodetype=int)
L = nx.laplacian_matrix(G).toarray()
curvature(L)