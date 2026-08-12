from collections import Counter,defaultdict
from pathlib import Path
import networkx as nx
import pynauty
from sklearn.metrics import normalized_mutual_info_score
import random
from tqdm.auto import tqdm
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.k_wl.kwl import run_kwl
from src.module.heat_kernel_equivalence_compute import HeatKernelEquivalenceComputer
from src.utils.graph_utils import *

import numpy as np

def main():
    graph_type = "36-14-4-6"
    with open(f"{graph_type}_output.txt", "w") as file:
        for instance in range(1, 300):
            print(f"Graph{instance}")
            graph_file = Path(f"graphs/TedSpence/{graph_type}/Graph{instance}")

            subdivision_k = 0

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
                include_edges=True,
                bfs_depth=100,
                weird_norm_threshold=1e-5,
                zero_tolerance=1e-12,
            )

            original_graph = read_dimacs_or_simple_graph(graph_file)
            # original_n = original_graph.number_of_nodes()
            graph = subdivide_every_edge(original_graph, subdivision_k)

            # Getting the other Edge Laplacian
            # nodes = list(graph.nodes())
            # edges = list(graph.edges())
            # incidence = nx.incidence_matrix(graph, nodelist=nodes, edgelist=edges, oriented=True).toarray()
            # laplacian_edges = np.asarray(incidence.T @ incidence, dtype=np.float64)
            # adjacency_edges = np.abs(laplacian_edges - np.diag(np.diag(laplacian_edges)))
            # graph_edges = nx.from_numpy_array(adjacency_edges)

            original_n = graph.number_of_nodes()
            # original_n_edges = graph_edges.number_of_nodes()

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

            print("G1-Classes: ", class_labels)
            print("G1-Orbits: ", orbit_labels)

            adjacency_2 = np.zeros((original_n, original_n))
            for i in range(original_n):
                for target, distance in nx.single_source_shortest_path_length(graph, i, 2).items():
                    if distance == 2 and target != i:
                        adjacency_2[i][target] = 1
                        adjacency_2[target][i] = 1

            graph_2 = nx.from_numpy_array(adjacency_2)

            signatures_2 = equivalence_computer.compute_vertex_signatures(graph_2)
            all_classes_2 = equivalence_computer.compute_equivalence_classes(signatures_2)
            classes_2 = restrict_to_original(all_classes_2,original_n)
            true_orbits_2 = get_true_orbits_restricted(graph_2,original_n)
            class_labels_2 = partition_to_labels(classes_2,original_n)
            orbit_labels_2 = partition_to_labels(true_orbits_2,original_n)

            print("G2-Classes: ", class_labels_2)
            print("G2-Orbits: ", orbit_labels_2)

            new_root_signatures = {}
            for i in range(len(signatures)):
                new_root_signatures[i] = signatures[i] + signatures_2[i]
            classes_all = defaultdict(list)
            for vertex, signature in new_root_signatures.items():
                classes_all[signature].append(vertex)
            class_labels_all = partition_to_labels(classes_all,original_n)

            print("G1&G2-Classes: ", class_labels_all)
            print("G1&G2-Orbits: ", orbit_labels)

            nmi = normalized_mutual_info_score(class_labels_all, orbit_labels)
            # probability = compute_overall_same_orbit_probability(graph, classes, original_n)

            print(f"NMI: {nmi:.4f}")
            # print(f"Same-orbit probability: {probability:.4f}")

            # k-WL
            k_wl_parameter = 1

            # unit_kwl_classes = run_kwl(original_graph, k_wl_parameter, output_dir="kwl_output/unit_coloring")
            heat_kwl_classes = run_kwl(original_graph, k_wl_parameter, initial_partition=classes_all, output_dir="kwl_output/heat_colored")

            # unit_kwl_labels = partition_to_labels(unit_kwl_classes, original_n)
            heat_kwl_labels = partition_to_labels(heat_kwl_classes, original_n)

            # unit_kwl_nmi = normalized_mutual_info_score(orbit_labels, unit_kwl_labels)
            heat_kwl_nmi = normalized_mutual_info_score(orbit_labels, heat_kwl_labels)

            true_orbit_set = {tuple(sorted(orbit)) for orbit in true_orbits}

            # unit_matches = sum(tuple(sorted(cls)) in true_orbit_set for cls in unit_kwl_classes)
            heat_matches = sum(tuple(sorted(cls)) in true_orbit_set for cls in heat_kwl_classes)

            # print(f"{k_wl_parameter}-WL unit_coloring: NMI = {unit_kwl_nmi:.4f}, classes = {len(unit_kwl_classes)}, matching classes = {unit_matches}")
            print(f"{k_wl_parameter}-WL heat_colored: NMI = {heat_kwl_nmi:.4f}, classes = {len(heat_kwl_classes)}, matching classes = {heat_matches}")

            file.write(f"Graph{instance}: NMI: {nmi:.4f}, NMI with 1-WL: {heat_kwl_nmi:.4f}\n")

            # signatures_edges = equivalence_computer.compute_vertex_signatures(graph_edges)
            # all_classes_edges = equivalence_computer.compute_equivalence_classes(signatures_edges)
            # classes_edges = restrict_to_original(all_classes_edges, original_n_edges)
            # true_orbits_edges = get_true_orbits_restricted(graph_edges, original_n_edges)
            # class_labels_edges = partition_to_labels(classes_edges, original_n_edges)
            # orbit_labels_edges = partition_to_labels(true_orbits_edges, original_n_edges)
            # max_label_from_nodes = int(np.max(np.array(class_labels))) + 1
            # # print(max_label_from_nodes)
            # for i in range(len(class_labels_edges)):
            #     class_labels_edges[i] += max_label_from_nodes

            # print("Edges Classes: ", class_labels_edges)
            # print("Edges Orbits: ", orbit_labels_edges)

            # signatures_with_bfs = equivalence_computer._signatures_with_bfs(graph, signatures, signatures_edges)
            # final_classes = equivalence_computer.compute_equivalence_classes(signatures_with_bfs)


            # final_signatures = {}
            # for vertex in tqdm(range(original_n)):
            #     label = [[class_labels[vertex]], []] #, []] # since strongly regular -> will have diameter 2
            #     for target, distance in nx.single_source_shortest_path_length(graph, vertex).items():
            #         # label[distance].append()    
            #         for edge_index in range(len(edges)):
            #             if distance == 1:
            #                 if 
            #             elif distance == 2:
            #                 if edges[edge_index][0] == target or edges[edge_index][1] == target:
            #                     label[distance].append(class_labels_edges[edge_index])

                # label[1] = sorted(label[1])
                # # print(label)
                # # label[2] = sorted(label[2])
                # final_signatures[vertex] = tuple(tuple(item) for item in label)

            # final_classes = defaultdict(list)

            # for node, sig in final_signatures.items():
            #     final_classes[sig].append(node)

            # final_class_labels = partition_to_labels(final_classes, original_n)

            # print("Final classes: ", final_class_labels)
            # print("Orbits: ", orbit_labels)

            # nmi = normalized_mutual_info_score(orbit_labels, final_class_labels)
            # # probability = compute_overall_same_orbit_probability(graph, final_classes, original_n)

            # print(f"NMI of Graph{instance}: {nmi:.4f}")

            # if int(nmi) != 1:
            #     print("OH DAMN")
            #     return

            # print(f"Same-orbit probability: {probability:.4f}")

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


if __name__ == "__main__":
    main()
