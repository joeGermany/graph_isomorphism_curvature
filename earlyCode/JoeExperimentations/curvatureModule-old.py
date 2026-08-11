import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import scipy.linalg as la
from scipy.linalg import expm

def fileToAdjacency(fileDirectory):
    with open(fileDirectory, "r") as f:
        lines = f.readlines()

    firstLine = lines[0].strip().split()
    num_of_vertices = int(firstLine[0])
    num_of_edges = int(firstLine[1])

    adjacencyMatrix = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float64)

    for i in range(1, len(lines)):
        connections = lines[i].strip().split()
        end1 = int(connections[0])
        end2 = int(connections[1])

        adjacencyMatrix[end1][end2] = 1
        adjacencyMatrix[end2][end1] = 1

    return adjacencyMatrix

def adjacencyToLaplacian(adjacencyMatrix):
    return np.diag(np.sum(adjacencyMatrix, axis=1)) - adjacencyMatrix

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

    # eigenvalues = np.round(eigenvalues, 10)
    lambda_fit = np.unique(np.sort(eigenvalues))

    print(N, len(lambda_fit))
    # print(lambda_fit[1] - lambda_fit[0])
    N_fit = np.zeros(len(lambda_fit))

    for i in range(len(lambda_fit)):
        c = np.where(lambda_fit < lambda_fit[i])[0]
        N_fit[i] = len(c) / len(lambda_fit)

    mask = N_fit > 0
    x = np.log(lambda_fit[mask])
    y = np.log(N_fit[mask])

    # print(x)

    plt.scatter(x, y, s=10)
    plt.xlabel("log(lambda)")
    plt.ylabel("log(N_fit)")
    plt.show()

    lowpoints = 2
    nbpoints = len(x)
    rsq = np.zeros(nbpoints - lowpoints + 1)

    for i in range(lowpoints, nbpoints + 1):
        fit = linregress(x[:i], y[:i])
        rsq[i - lowpoints] = fit.rvalue**2
    
    best = np.argmax(rsq)
    best_n = lowpoints + best

    fit = linregress(x[:best_n], y[:best_n])

    slope = fit.slope
    dim = 2*slope
    print(dim)

    # lambda_critical = x[best_n - 1]

    # Ti = np.arange(1 / np.max(lambda_fit), lambda_critical, 0.001)
    # print(1 / np.max(lambda_fit), lambda_critical)
    # orderinexpan = np.max(np.diag(laplacianMatrix)).astype(int)

    # X = np.zeros((len(Ti), orderinexpan))

    # X[:, 0] = 1 / (4 * np.pi * Ti)**(dim / 2)

    # for i in range(2, orderinexpan + 1):
    #     X[:, i - 1] = (Ti ** (i - 1)) / (4 * np.pi * Ti)**(dim / 2)

    # print(X)

    # K1diag = np.zeros((laplacianMatrix.shape[0], len(Ti)))

    # for c, t in enumerate(Ti):
    #     Ker1 = expm( - laplacianMatrix * t)
    #     K1diag[:, c] = np.diag(Ker1)

    # X_pinv = np.linalg.pinv(X)

    # vr = np.zeros(laplacianMatrix.shape[0])

    # for j in range(laplacianMatrix.shape[0]):
    #     Y = K1diag[j, :]
    #     coeffs = X_pinv @ Y

    #     vr[j] = coeffs[1]

    # print(vr)
    # return dim, vr

adjacencyMatrix = fileToAdjacency("ags/ag2-08__00")
laplacianMatrix = adjacencyToLaplacian(adjacencyMatrix)
curvature(laplacianMatrix)

# sparse_matrix = sparse.coo_matrix(laplacianMatrix)
# print(sparse_matrix)