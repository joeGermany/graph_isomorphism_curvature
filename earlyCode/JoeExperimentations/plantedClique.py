import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score

from src.k_wl.kwl import run_kwl
from src.module.heat_kernel_equivalence_compute import HeatKernelEquivalenceComputer

from src.utils.graph_utils import *
from src.probing.probing import *


def planted_clique_regular_graph(
    n_G=100,
    d_G=23,
    clique_size=20,
    seed=None,
):
    """
    Construct a regular graph with a planted clique.

    Construction:
      - G is a d_G-regular graph on n_G vertices.
      - K is a clique on clique_size vertices.
      - Each clique vertex is connected to 5 distinct vertices of G.
      - The vertices of G are partitioned into groups of 5.
      - Thus:
            degree(clique vertex) = (clique_size - 1) + 5
            degree(G vertex)      = d_G + 1

    For the default parameters:
        G: 100 vertices, degree 23
        K: 20 vertices
        final graph: 120 vertices, degree 24
    """

    if n_G % clique_size != 0:
        raise ValueError(
            "n_G must be divisible by clique_size so that "
            "the G vertices can be partitioned evenly."
        )

    attachments_per_clique_vertex = n_G // clique_size

    # Check that the construction is actually regular.
    clique_degree = (clique_size - 1) + attachments_per_clique_vertex
    G_degree = d_G + 1

    if clique_degree != G_degree:
        raise ValueError(
            f"Construction is not regular: "
            f"clique vertices have degree {clique_degree}, "
            f"while G vertices have degree {G_degree}."
        )

    # ---------------------------------------------------------
    # 1. Generate the base regular graph G
    # ---------------------------------------------------------
    G = nx.random_regular_graph(
        d=d_G,
        n=n_G,
        seed=seed,
    )

    # Relabel G vertices to 0, ..., n_G-1
    G = nx.convert_node_labels_to_integers(G)

    # ---------------------------------------------------------
    # 2. Create the clique
    # ---------------------------------------------------------
    clique_vertices = list(range(n_G, n_G + clique_size))

    K = nx.complete_graph(clique_vertices)

    # ---------------------------------------------------------
    # 3. Combine G and K
    # ---------------------------------------------------------
    H = nx.compose(G, K)

    # ---------------------------------------------------------
    # 4. Partition G into groups and attach each group
    #    to one clique vertex
    # ---------------------------------------------------------
    G_vertices = list(range(n_G))

    for i, clique_vertex in enumerate(clique_vertices):

        start = i * attachments_per_clique_vertex
        end = (i + 1) * attachments_per_clique_vertex

        group = G_vertices[start:end]

        for v in group:
            H.add_edge(clique_vertex, v)

    return H, G, clique_vertices


# =============================================================
# Example
# =============================================================

H, G, clique = planted_clique_regular_graph(
    n_G=100,
    d_G=23,
    clique_size=20,
    seed=42,
)

print("Number of vertices:", H.number_of_nodes())
print("Number of edges:", H.number_of_edges())

degrees = dict(H.degree())
print("Degrees:", set(degrees.values()))

print("Clique vertices:", clique)
print("Clique size:", len(clique))

# Verify that the planted clique really is complete
is_clique = all(
    H.has_edge(u, v)
    for i, u in enumerate(clique)
    for v in clique[i + 1:]
)

print("Is planted clique complete?", is_clique)

# Verify regularity
print("Is graph regular?", len(set(dict(H.degree()).values())) == 1)

# =============================================================
# Optional: visualize
# =============================================================

pos = nx.spring_layout(H, seed=42)

plt.figure(figsize=(12, 10))

# Base graph vertices
nx.draw_networkx_nodes(
    H,
    pos,
    nodelist=list(range(100)),
    node_size=30,
)

# Planted clique vertices
nx.draw_networkx_nodes(
    H,
    pos,
    nodelist=clique,
    node_size=100,
)

nx.draw_networkx_edges(
    H,
    pos,
    alpha=0.15,
    width=0.5,
)

plt.axis("off")
plt.show()


graph_settings = [
    {
        "name": "G",
        "transformations": [],
        "operators": ["laplacian"],
        "bfs_sort": True,
        "signature_mode": "full_matrix",
        "time_step": 1,
        "end_time_multiplier": 7,
        "fit_order": 6,
        "fit_start": 1,
        "fit_end": 6,
        "resolutions": [1e8],
        "dirac_decimals": 6,
        "dirac_mass": 1.0,
        "precision_dps": None
    },
    # {
    #     "name": "L(G)",
    #     "transformations": [("line", 1)],
    #     "operators": ["laplacian"],
    #     "bfs_sort": True,
    #     "signature_mode": "full_matrix",
    #     "time_step": 1,
    #     "end_time_multiplier": 7,
    #     "fit_order": 6,
    #     "fit_start": 1,
    #     "fit_end": 6,
    #     "resolutions": [1e8],
    #     "dirac_decimals": 6,
    #     "dirac_mass": 1.0,
    #     "precision_dps": None
    # }
]

equivalence_computer = HeatKernelEquivalenceComputer(
    graph_settings=graph_settings,
    comparison_mode="fixed_resolution",
    bfs_depth=100,
    weird_norm_threshold=1e-5,
    zero_tolerance=1e-12,
    use_ids=True,
    use_stable_ids = False,
)

graph = H
original_n=graph.number_of_nodes()

all_classes,signatures=equivalence_computer.compute_vertex_signatures(graph)

# print(signatures)

classes=restrict_to_original(all_classes,original_n)
true_orbits=get_true_orbits_restricted(graph,original_n)

class_labels=partition_to_labels(classes,original_n)
orbit_labels=partition_to_labels(true_orbits,original_n)

print(class_labels)
print(orbit_labels)

def signature_distance(a, b):
    """
    Distance between two nested tuple signatures.
    The signatures themselves are never modified.
    """

    # Exactly equal
    if a == b:
        return 0

    # Both tuples: recursively compare their contents
    if isinstance(a, tuple) and isinstance(b, tuple):

        # If lengths differ, unmatched pieces contribute distance
        n = max(len(a), len(b))

        # Pair the entries in order
        distance = abs(len(a) - len(b))

        for x, y in zip(a, b):
            distance += signature_distance(x, y)

        return distance

    # One is a tuple and the other isn't
    if isinstance(a, tuple) or isinstance(b, tuple):
        return 1

    # Different leaves
    if a != b:
        return 1

    return 0


import numpy as np

def flatten_signature(x):
    """Flatten a nested tuple/list into a 1D numerical vector."""
    if isinstance(x, (tuple, list)):
        result = []
        for item in x:
            result.extend(flatten_signature(item))
        return result
    else:
        return [x]


# Flatten every signature
flat_signatures = {
    v: np.array(flatten_signature(sig), dtype=float)
    for v, sig in signatures.items()
}

for i in range(len(flat_signatures)):
    flat_signatures[i] = flat_signatures[i] / 1e1

# # Check lengths
# for v, sig in flat_signatures.items():
#     print(v, len(sig))


def euclidean_distance(v, w):
    return np.linalg.norm(
        flat_signatures[v] - flat_signatures[w]
    )


from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

vertices = sorted(signatures)

X = np.array([
    flat_signatures[v]
    for v in vertices
])

D = squareform(pdist(X, metric="euclidean"))

Z = linkage(X, method="average", metric="euclidean")

labels = fcluster(Z, t=10, criterion="distance")

for label in sorted(set(labels)):
    cluster = [
        v for v, c in zip(vertices, labels)
        if c == label
    ]
    print(label, cluster)