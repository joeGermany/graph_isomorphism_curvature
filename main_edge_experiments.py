from collections import Counter,defaultdict
from pathlib import Path
import networkx as nx
import pynauty
from sklearn.metrics import normalized_mutual_info_score
import random
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.k_wl.kwl import run_kwl
from src.module.heat_kernel_equivalence_compute import HeatKernelEquivalenceComputer
from src.utils.graph_utils import *

import numpy as np

def main():
    graph_file = Path("graphs/TedSpence/26-10-3-4/Graph3")

    # signature_mode:
    #   "coefficients" -> fitted coefficients; set the parameters fit_order/fit_start/fit_end
    #   "diagonal"     -> heat-diagonal values
    #   "full_matrix"  -> ordered full heat rows
    #

    # include_edges: include edge signatures

    # comparison_mode:
    #
    # "fixed_resolution":
    #     Compare signatures after truncating them using `resolutions`.
    #     - For "coefficients", `resolutions` must contain one value for each
    #       selected coefficient, so its length must be at least
    #       fit_end - fit_start.
    #     - For "diagonal" and "full_matrix", only resolutions[0] is used.
    #
    # "weird_norm":
    #     Compare signatures using `weird_norm_threshold`.
    #     The `resolutions` values are not used.

    # end_time_multiplier controls the upper end of the sampled time interval:
    #   1  -> end_time = 1 / lambda_min

    subdivision_k = 0

    # Use compact shared signature IDs instead of storing full signatures.
    use_ids = False

    node_settings = {
        "signature_mode": "full_matrix",
        "time_step": 0.1,
        "end_time_multiplier": 100,
        "fit_order": 6,
        "fit_start": 1,
        "fit_end": 6,
        "resolutions": [1e10],
    }

    edge_settings = {
        "signature_mode": "full_matrix",
        "time_step": 0.1,
        "end_time_multiplier": 100,
        "fit_order": 6,
        "fit_start": 1,
        "fit_end": 4,
        "resolutions": [1e10],
    }

    equivalence_computer = HeatKernelEquivalenceComputer(
        node_settings=node_settings,
        edge_settings=edge_settings,
        comparison_mode="fixed_resolution",
        include_edges=False,
        bfs_depth=100,
        weird_norm_threshold=1e-5,
        zero_tolerance=1e-12,
    )

    original_graph = read_dimacs_or_simple_graph(graph_file)
    # original_n = original_graph.number_of_nodes()
    graph = subdivide_every_edge(original_graph, subdivision_k)

    # Getting the other Edge Laplacian
    nodes = list(graph.nodes())
    edges = list(graph.edges())
    incidence = nx.incidence_matrix(graph, nodelist=nodes, edgelist=edges, oriented=True).toarray()
    laplacian_edges = np.asarray(incidence.T @ incidence, dtype=np.float64)
    adjacency_edges = np.abs(laplacian_edges - np.diag(np.diag(laplacian_edges)))
    graph_edges = nx.from_numpy_array(adjacency_edges)

    original_n = graph.number_of_nodes()
    original_n_edges = graph_edges.number_of_nodes()

    # for node, degree in graph.degree():
    #     print(f"Node {node}: {degree}")

    # for i in range(original_n):
    #     for j in range(i):
    #         print(f"Common between {j} and {i}: {nx.common_neighbors(graph, i, j)}")

    # nx.write_gexf(graph, "25-12-5-6_3_edge_graph.gexf")

    signatures = equivalence_computer.compute_vertex_signatures(graph)
    all_classes = equivalence_computer.compute_equivalence_classes(signatures)
    classes = restrict_to_original(all_classes,original_n)
    true_orbits = get_true_orbits_restricted(graph,original_n)
    class_labels = partition_to_labels(classes,original_n)
    orbit_labels = partition_to_labels(true_orbits,original_n)

    print("Nodes Classes: ", class_labels)
    print("Nodes Orbits: ", orbit_labels)

    signatures_edges = equivalence_computer.compute_vertex_signatures(graph_edges)
    all_classes_edges = equivalence_computer.compute_equivalence_classes(signatures_edges)
    classes_edges = restrict_to_original(all_classes_edges, original_n_edges)
    true_orbits_edges = get_true_orbits_restricted(graph_edges, original_n_edges)
    class_labels_edges = partition_to_labels(classes_edges, original_n_edges)
    orbit_labels_edges = partition_to_labels(true_orbits_edges, original_n_edges)
    max_label_from_nodes = int(np.max(np.array(class_labels))) + 1
    print(max_label_from_nodes)
    for i in range(len(class_labels_edges)):
        class_labels_edges[i] += max_label_from_nodes

    print("Edges Classes: ", class_labels_edges)
    print("Edges Orbits: ", orbit_labels_edges)

    final_signatures = {}
    for vertex in range(original_n):
        label = [[class_labels[vertex]], []] #, []] # since strongly regular -> will have diameter 2
        for target, distance in nx.single_source_shortest_path_length(graph, vertex).items():
            # label[distance].append()
            for edge_index in range(len(edges)):
                if (edges[edge_index][0] == vertex and edges[edge_index][1] == target) or (edges[edge_index][0] == target and edges[edge_index][1] == vertex):
                    label[distance].append(class_labels_edges[edge_index])

        label[1] = sorted(label[1])
        print(label)
        # label[2] = sorted(label[2])
        final_signatures[vertex] = tuple(tuple(item) for item in label)

    final_classes = defaultdict(list)

    for node, sig in final_signatures.items():
        final_classes[sig].append(node)

    final_class_labels = partition_to_labels(final_classes, original_n)

    print(final_class_labels)
    print(orbit_labels)

    nmi = normalized_mutual_info_score(orbit_labels, final_class_labels)
    probability = compute_overall_same_orbit_probability(graph, classes, original_n)

    print(f"NMI: {nmi:.4f}")
    print(f"Same-orbit probability: {probability:.4f}")

    # k-WL uses an external executable named `run_wl`.
    # Before running the experiment, compile it once with:
    # 
    #   cd src/k_wl
    # 
    # Linux / Windows with g++:
    #   g++ -std=c++17 -O3 main.cpp weisfeiler_lehman.cpp -o run_wl
    #
    # macOS:
    #   clang++ -std=c++17 -O3 main.cpp weisfeiler_lehman.cpp -o run_wl

    # UNCOMMENT FROM HERE:
    # k_wl_parameter = 2

    # unit_kwl_classes = run_kwl(original_graph, k_wl_parameter, output_dir="kwl_output/unit_coloring")
    # heat_kwl_classes = run_kwl(original_graph, k_wl_parameter, initial_partition=classes, output_dir="kwl_output/heat_colored")

    # unit_kwl_labels = partition_to_labels(unit_kwl_classes, original_n)
    # heat_kwl_labels = partition_to_labels(heat_kwl_classes, original_n)

    # unit_kwl_nmi = normalized_mutual_info_score(orbit_labels, unit_kwl_labels)
    # heat_kwl_nmi = normalized_mutual_info_score(orbit_labels, heat_kwl_labels)

    # true_orbit_set = {tuple(sorted(orbit)) for orbit in true_orbits}

    # unit_matches = sum(tuple(sorted(cls)) in true_orbit_set for cls in unit_kwl_classes)
    # heat_matches = sum(tuple(sorted(cls)) in true_orbit_set for cls in heat_kwl_classes)

    # print(f"{k_wl_parameter}-WL unit_coloring: NMI = {unit_kwl_nmi:.4f}, classes = {len(unit_kwl_classes)}, matching classes = {unit_matches}")
    # print(f"{k_wl_parameter}-WL heat_colored: NMI = {heat_kwl_nmi:.4f}, classes = {len(heat_kwl_classes)}, matching classes = {heat_matches}")

if __name__ == "__main__":
    main()
