import networkx as nx
import random

import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score

from src.k_wl.kwl import run_kwl
from src.module.heat_kernel_equivalence_compute import HeatKernelEquivalenceComputer

from src.utils.graph_utils import *
from src.probing.probing import *


def planted_clique_regular_graph(k, r, seed=None):
    """
    Construct a regular graph by:

      1. Generating a random d-regular graph G
      2. Adding a K_k clique
      3. Partitioning V(G) into k groups of size r
      4. Connecting clique vertex i to every vertex in group i

    Parameters
    ----------
    k : int
        Size of planted clique.
    r : int
        Number of G-vertices attached to each clique vertex.

    Returns
    -------
    H : final regular graph
    G : original regular graph
    clique : list of clique vertices
    """

    n = k * r
    d = k + r - 2

    # Need d < n
    if d >= n:
        raise ValueError(
            f"Invalid parameters: d={d} must be < n={n}"
        )

    # Need nd even for a d-regular graph
    if (n * d) % 2 != 0:
        raise ValueError(
            f"No d-regular graph possible: n*d={n*d} is odd."
        )

    # Generate original regular graph
    G = nx.random_regular_graph(
        d=d,
        n=n,
        seed=seed
    )

    G = nx.convert_node_labels_to_integers(G)

    # Clique vertices
    clique = list(range(n, n + k))

    # Add clique
    H = G.copy()
    H.add_nodes_from(clique)

    for i in range(k):
        for j in range(i + 1, k):
            H.add_edge(clique[i], clique[j])

    # Attach each clique vertex to r distinct G vertices
    G_vertices = list(range(n))

    # Shuffle so attachment is random
    rng = random.Random(seed)
    rng.shuffle(G_vertices)

    for i, c in enumerate(clique):
        group = G_vertices[i*r:(i+1)*r]

        for v in group:
            H.add_edge(c, v)

    # Verify regularity
    degrees = set(dict(H.degree()).values())

    if len(degrees) != 1:
        raise RuntimeError(
            f"Construction failed: degrees = {degrees}"
        )

    return H, G, clique


examples = [
    (10, 5),
    (10, 8),
    (15, 4),
    (15, 6),
    (20, 5),
    (20, 8),
    (25, 4),
    (25, 6),
    (30, 5),
]

for i, (k, r) in enumerate(examples):

    H, G, clique = planted_clique_regular_graph(
        k=k,
        r=r,
        seed=i
    )

    print(
        f"k={k:2d}, r={r:2d}: "
        f"|H|={H.number_of_nodes():3d}, "
        f"degree={next(iter(set(dict(H.degree()).values())))}, "
        f"clique={clique[0]}..{clique[-1]}"
    )


graphs = []

for seed in range(20):
    H, G, clique = planted_clique_regular_graph(
        k=20,
        r=5,
        seed=seed
    )

    graphs.append((H, G, clique))

    print('cliqe:', clique)

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