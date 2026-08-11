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

def main():
    graph_file = Path("graphs/TedSpence/25-12-5-6/Graph2")

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
        "end_time_multiplier": 5,
        "fit_order": 6,
        "fit_start": 1,
        "fit_end": 6,
        "resolutions": [1e10],
    }

    edge_settings = {
        "signature_mode": "full_matrix",
        "time_step": 0.1,
        "end_time_multiplier": 5,
        "fit_order": 6,
        "fit_start": 1,
        "fit_end": 4,
        "resolutions": [1e10],
    }

    equivalence_computer = HeatKernelEquivalenceComputer(
        node_settings=node_settings,
        edge_settings=edge_settings,
        comparison_mode="fixed_resolution",
        include_edges=True,
        bfs_depth=100,
        weird_norm_threshold=1e-5,
        zero_tolerance=1e-12,
    )

    original_graph = read_dimacs_or_simple_graph(graph_file)
    original_n = original_graph.number_of_nodes()
    graph = subdivide_every_edge(original_graph, subdivision_k)

    if use_ids:
        H, mapping = randomly_permute_graph(graph)

        G_vertex_ids, H_vertex_ids = equivalence_computer.compute_shared_vertex_ids(graph, H)

        G_all_classes = equivalence_computer.compute_equivalence_classes(G_vertex_ids)
        H_all_classes = equivalence_computer.compute_equivalence_classes(H_vertex_ids)

        ids_match = all(
            G_vertex_ids[vertex] == H_vertex_ids[mapping[vertex]]
            for vertex in graph.nodes()
        )

        GREEN = "\033[92m"
        RED = "\033[91m"
        RESET = "\033[0m"

        # print(f"Shared IDs preserved under permutation: {GREEN if ids_match else RED}{ids_match}{RESET}")

        all_classes = G_all_classes

    else:
        signatures = equivalence_computer.compute_vertex_signatures(graph)
        all_classes = equivalence_computer.compute_equivalence_classes(signatures)

    classes = restrict_to_original(all_classes,original_n)
    true_orbits = get_true_orbits_restricted(graph,original_n)

    class_labels = partition_to_labels(classes,original_n)
    orbit_labels = partition_to_labels(true_orbits,original_n)

    print(class_labels)
    print(orbit_labels)

    nmi = normalized_mutual_info_score(orbit_labels,class_labels)
    probability = compute_overall_same_orbit_probability(graph,classes,original_n)

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
