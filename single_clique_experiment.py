from pathlib import Path
from itertools import combinations
from collections import defaultdict

from src.module.heat_kernel_equivalence_compute import HeatKernelEquivalenceComputer
from src.utils.graph_utils import *


GRAPH_FILE = Path("graphs/nauty_graphs/dimacs/pp/pp-9-2")

VERTICES = [0, 67, 91, 97]


def fold_vertices(graph, group):
    probed = graph.copy()
    group = set(group)

    # Collect all neighbors outside the folding group
    external_neighbors = set()

    for u in group:
        for v in probed.neighbors(u):
            if v not in group:
                external_neighbors.add(v)

    # Create one new vertex representing the folded group
    new_vertex = max(probed.nodes()) + 1

    # Remove the vertices being folded
    probed.remove_nodes_from(group)

    # Add the new folded vertex
    probed.add_node(new_vertex)

    # Connect the folded vertex to all external neighbors
    for v in external_neighbors:
        probed.add_edge(new_vertex, v)

    # No self-loops
    if probed.has_edge(new_vertex, new_vertex):
        probed.remove_edge(new_vertex, new_vertex)

    return probed, new_vertex


def main():
    G = read_dimacs_or_simple_graph(GRAPH_FILE)

    setting = {
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
        "precision_dps": None,
    }

    computer = HeatKernelEquivalenceComputer(
        graph_settings=[setting],
        comparison_mode="fixed_resolution",
        bfs_depth=100,
        use_ids=False,
        use_stable_ids=False,
    )

    triplet_signatures = {}

    for triplet in combinations(VERTICES, 3):
        print(f"\nFolding {triplet}")

        probed, folded_vertex = fold_vertices(G, triplet)

        print("Original n:", G.number_of_nodes())
        print("Folded n:", probed.number_of_nodes())
        print("Folded vertex:", folded_vertex)
        print("Folded vertex degree:", probed.degree(folded_vertex))

        _, sigs = computer.compute_vertex_signatures(probed)

        # Signature of the newly created folded vertex
        triplet_signatures[triplet] = sigs[folded_vertex]

    # Group triplets by equality of folded-vertex signatures
    classes = defaultdict(list)

    for triplet, sig in triplet_signatures.items():
        classes[sig].append(triplet)

    print("\n================================")
    print("SIGNATURE GROUPS")
    print("================================")

    for group in classes.values():
        print(group)

    print("\nNumber of triplets:", len(triplet_signatures))
    print("Number of distinct signatures:", len(classes))


if __name__ == "__main__":
    main()