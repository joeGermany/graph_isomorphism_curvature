import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import networkx as nx
from pathlib import Path
import ast
from collections import defaultdict
from sklearn.metrics.cluster import normalized_mutual_info_score

try:
    import cupy as cp
except Exception:
    cp = None



def gpu_is_available():
    if cp is None:
        return False

    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def compute_laplacian_eigendecomposition(laplacianMatrix, prefer_gpu=True):
    use_gpu = prefer_gpu and gpu_is_available()

    if use_gpu:
        print("Using GPU solver: CuPy / cuSOLVER")

        laplacian_gpu = cp.asarray(laplacianMatrix, dtype=cp.float64)
        eigenvalues_gpu, eigenvectors_gpu = cp.linalg.eigh(laplacian_gpu)

        cp.cuda.Device().synchronize()

        eigenvalues_cpu = cp.asnumpy(eigenvalues_gpu)
        eigenvalues_cpu = np.real(eigenvalues_cpu)
        nonzero_eigenvalues_cpu = eigenvalues_cpu[eigenvalues_cpu > 1e-10]

        return eigenvalues_cpu, nonzero_eigenvalues_cpu, eigenvalues_gpu, eigenvectors_gpu, "gpu"

    else:
        print("GPU not available. Using CPU solver: NumPy")

        laplacian_cpu = np.asarray(laplacianMatrix, dtype=np.float64)
        eigenvalues_cpu, eigenvectors_cpu = np.linalg.eigh(laplacian_cpu)

        eigenvalues_cpu = np.real(eigenvalues_cpu)
        nonzero_eigenvalues_cpu = eigenvalues_cpu[eigenvalues_cpu > 1e-10]

        return eigenvalues_cpu, nonzero_eigenvalues_cpu, eigenvalues_cpu, eigenvectors_cpu, "cpu"

def analyze_eigenvalue_distribution(eigenvalues):

    round_digits = 9
    eigenvalues = np.round(eigenvalues, round_digits)
    lambda_fit = np.unique(np.sort(eigenvalues))

    print("\nUnique rounded nonzero eigenvalues:")
    for value in lambda_fit:
        print(value)
    print("Number of unique nonzero eigenvalues:", len(lambda_fit))

    N_fit = np.zeros(len(lambda_fit))

    for i in range(len(lambda_fit)):
        c = np.where(lambda_fit < lambda_fit[i])[0]
        N_fit[i] = (len(c) + 1) / len(lambda_fit)

    mask = N_fit > 0
    x = np.log(lambda_fit[mask])
    y = np.log(N_fit[mask])

    plt.scatter(x, y, s=10)
    plt.xlabel("log(lambda)")
    plt.ylabel("log(N_fit)")
    plt.show()

    lowpoints = 4
    nbpoints = len(x)

    if nbpoints < lowpoints:
        print("Not enough points to compute dimension.")
        return lambda_fit, None, None

    rsq = np.zeros(nbpoints - lowpoints + 1)

    for i in range(lowpoints, nbpoints + 1):
        fit = linregress(x[:i], y[:i])
        rsq[i - lowpoints] = fit.rvalue ** 2

    best = np.argmax(rsq)
    best_n = lowpoints + best

    fit = linregress(x[:best_n], y[:best_n])

    slope = fit.slope
    dim = 2 * slope

    print("\nDimension:", dim)

    lambda_critical = np.exp(x[best_n - 1])

    t_start = 1 / np.max(lambda_fit)
    t_end = 1 / lambda_critical

    Ti = np.arange(t_start, t_end, 0.001)

    print("lambda_critical:", lambda_critical)
    print("1 / max(lambda):", t_start)
    print("1 / lambda_critical:", t_end)
    print("Number of time points:", len(Ti))

    return lambda_fit, dim, Ti


def compute_heat_kernel_diagonal(lambda_all, V, Ti, backend):
    if Ti is None or len(Ti) == 0:
        raise ValueError("Ti is empty. Cannot compute heat-kernel diagonal.")

    if backend == "gpu":
        print("Computing heat-kernel diagonal on GPU")

        Ti_gpu = cp.asarray(Ti, dtype=cp.float64)
        V2_gpu = V ** 2

        exp_terms_gpu = cp.exp(
            -Ti_gpu[:, None] * lambda_all[None, :]
        )

        Kdiag_gpu = exp_terms_gpu @ V2_gpu.T

        K1diag_gpu = Kdiag_gpu.T

        cp.cuda.Device().synchronize()

        K1diag = cp.asnumpy(K1diag_gpu)

    else:
        print("Computing heat-kernel diagonal on CPU using NumPy")

        lambda_all = np.asarray(lambda_all, dtype=np.float64)

        V = np.asarray(V, dtype=np.float64)
        V2 = V ** 2

        exp_terms = np.exp(
            -Ti[:, None] * lambda_all[None, :]
        )

        Kdiag = exp_terms @ V2.T
        K1diag = Kdiag.T

    print("\nK1diag shape:", K1diag.shape)

    return K1diag

def compute_equivalence_classes_from_K1diag(K1diag, Ti, dim, laplacianMatrix, fullVector=False):
    orderinexpan = 6

    X = np.zeros((len(Ti), orderinexpan))

    for i in range(0, orderinexpan):
        X[:, i] = (Ti ** i) / (4 * np.pi * Ti) ** (dim / 2)

    X_pinv = np.linalg.pinv(X)

    vr = np.zeros(laplacianMatrix.shape[0])
    full_coeffs = np.zeros((laplacianMatrix.shape[0], orderinexpan))

    for j in range(laplacianMatrix.shape[0]):
        Y = K1diag[j, :]

        coeffs = X_pinv @ Y

        if fullVector:
            full_coeffs[j, :] = coeffs
        else:
            vr[j] = coeffs[1]

    resolution = 1e8
    groups = defaultdict(list)

    if fullVector:
        full_coeffs = full_coeffs[:, 1:6]

        full_coeffs[:, 0] = np.trunc(full_coeffs[:, 0] * resolution) / resolution
        full_coeffs[:, 1] = np.trunc(full_coeffs[:, 1] * 1e6) / 1e6
        full_coeffs[:, 2] = np.trunc(full_coeffs[:, 2] * 1e5) / 1e5
        full_coeffs[:, 3] = np.trunc(full_coeffs[:, 3] * 1e5) / 1e5
        full_coeffs[:, 4] = np.trunc(full_coeffs[:, 4] * 1e4) / 1e4

        print("\nUnique full coefficient vectors:")
        print(np.unique(full_coeffs, axis=0))
        print("Number of unique coefficient vectors:", len(np.unique(full_coeffs, axis=0)))

        for i, row in enumerate(full_coeffs):
            groups[tuple(row)].append(i)

    else:
        vr = np.trunc(vr * resolution) / resolution
        print("Number of unique curvature values:", len(np.unique(vr)))
        print("\nUnique curvature values:")
        print(np.unique(vr))

        for i, value in enumerate(vr):
            groups[value].append(i)

    equivalence_classes = list(groups.values())

    return equivalence_classes


def curvature(laplacianMatrix, fullVector=False, prefer_gpu=True):

    all_eigenvalues, nonzero_eigenvalues, lambda_all, V, backend = compute_laplacian_eigendecomposition( laplacianMatrix, prefer_gpu=prefer_gpu)

    lambda_fit, dim, Ti = analyze_eigenvalue_distribution(nonzero_eigenvalues)

    K1diag = compute_heat_kernel_diagonal(lambda_all,V,Ti,backend)

    equivalence_classes = compute_equivalence_classes_from_K1diag(K1diag,Ti,dim,laplacianMatrix,fullVector=fullVector)

    return equivalence_classes


def get_orbits_from_file(original_filename):
    orbits_filename = str(original_filename) + "_orbits.txt"

    orbits = []

    with open(orbits_filename) as f:
        num_of_orbits = int(next(f).split()[0])

        for i in range(num_of_orbits):
            orbit = ast.literal_eval(next(f).strip())
            orbits.append(orbit)

    return orbits


def turn_membership_array_into_clustering_array(array, num_of_nodes):
    clustering = np.zeros(num_of_nodes, dtype=int)

    for i in range(len(array)):
        for vertex in array[i]:
            clustering[vertex] = i

    return clustering


filename = "graphs/ag2-8/ag2-08__01"
orbits_filename = "orbits/ag2-8/ag2-08__01"

with open(filename) as f:
    num_of_nodes = int(next(f).split()[0])
    G = nx.parse_edgelist(f, nodetype=int)

G.add_nodes_from(range(num_of_nodes))

L = nx.laplacian_matrix(G, nodelist=range(num_of_nodes)).toarray()

orbits = get_orbits_from_file(str(orbits_filename))

orbits_clustering = turn_membership_array_into_clustering_array(orbits,num_of_nodes)

equivalence_classes = curvature(L, fullVector=False, prefer_gpu=True)

equivalence_classes_clustering = turn_membership_array_into_clustering_array(equivalence_classes,num_of_nodes)


print("NMI: ",normalized_mutual_info_score(orbits_clustering, equivalence_classes_clustering))

print("\nTrue orbits:")
print("Number of orbits:", len(orbits))
for i, orbit in enumerate(orbits):
    print(f"Orbit {i}: {sorted(orbit)}")

print("\nCurvature equivalence classes:")
print("Number of equivalence classes:", len(equivalence_classes))
for i, eq_class in enumerate(equivalence_classes):
    print(f"Class {i}: {sorted(eq_class)}")