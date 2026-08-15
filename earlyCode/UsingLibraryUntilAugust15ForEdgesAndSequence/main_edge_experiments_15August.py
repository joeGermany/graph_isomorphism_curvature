from collections import Counter,defaultdict
from pathlib import Path
import networkx as nx
import pynauty
from sklearn.metrics import normalized_mutual_info_score
import random
from tqdm.auto import tqdm
import time
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.k_wl.kwl import run_kwl
from src.module.heat_kernel_equivalence_compute import HeatKernelEquivalenceComputer
from src.utils.graph_utils import *

import numpy as np
import scipy.sparse as sp

def main():
    graph_type = "35-16-6-8"
    with open(f"{graph_type}_output.txt", "w") as file:
        for instance in range(1, 3855): #[5306, 7185, 7360, 10133, 10266, 10408, 10411, 10413, 10414, 10415, 10879, 10904, 11064, 13011, 15310, 17286, 21091, 21092, 22646, 22656, 23842, 24208, 24247, 25142, 25385, 27748, 27755, 27778, 28522, 28524, 28525, 28594, 28629, 28860, 28861, 28867, 28882, 28883, 28884, 28885, 28886]:#[1,3,5,12,13,15,17,21]: #[5306, 7185, 7360]:#range(1, 32548): #[31, 142, 522, 1046, 1769, 1964, 2002, 2019, 2025, 2091, 2306, 2389, 2591, 2617, 2898, 3590, 3775]:#range(1, 3855): #[15, 16, 29, 31, 34, 38, 41, 46, 47, 48, 50, 52, 73, 78]: #range(1,79): #[48, 52, 57, 60, 65, 66, 67, 70, 103, 120, 124, 134, 136, 144, 145, 146, 147, 148, 153, 155, 157, 159, 160, 161, 163, 164, 167, 169, 170, 172, 173, 174, 176, 177, 178, 179]: #range(178, 179):
            print(f"Graph{instance}")
            graph_file = Path(f"graphs/{graph_type}/Graph{instance}")

            subdivision_k = 0

            node_settings = {
                "signature_mode": "full_matrix",
                "time_step": 0.1,
                "end_time_multiplier": 15,
                "fit_order": 6,
                "fit_start": 1,
                "fit_end": 6,
                "resolutions": [1e11],
            }

            edge_settings = {
                "signature_mode": "full_matrix",
                "time_step": 1,
                "end_time_multiplier": 15,
                "fit_order": 6,
                "fit_start": 1,
                "fit_end": 4,
                "resolutions": [1e11],
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

            # true_orbits = get_true_orbits_restricted(graph,original_n)
            # orbit_labels = partition_to_labels(true_orbits,original_n)
            # print(orbit_labels)

            # Getting the other Edge Laplacian
            # nodes = list(graph.nodes())
            # edges = list(graph.edges())
            # incidence = nx.incidence_matrix(graph, nodelist=nodes, edgelist=edges, oriented=True).toarray()
            # laplacian_edges = np.asarray(incidence.T @ incidence, dtype=np.float64)
            # adjacency_edges = np.abs(laplacian_edges - np.diag(np.diag(laplacian_edges)))
            # graph_edges = nx.from_numpy_array(adjacency_edges)

            original_n = graph.number_of_nodes()
            # original_n = graph_edges.number_of_nodes()
            # original_n_edges = graph_edges.number_of_nodes()

            # for node, degree in graph.degree():
            #     print(f"Node {node}: {degree}")

            # for i in range(original_n):
            #     for j in range(i):
            #         print(f"Common between {j} and {i}: {nx.common_neighbors(graph, i, j)}")

            # nx.write_gexf(graph, "25-12-5-6_3_edge_graph.gexf")

            signatures = equivalence_computer.compute_vertex_signatures(graph)
            # signatures = equivalence_computer.compute_vertex_signatures(graph_edges)
            all_classes = equivalence_computer.compute_equivalence_classes(signatures)
            classes = restrict_to_original(all_classes,original_n)
            true_orbits = get_true_orbits_restricted(graph,original_n)
            # true_orbits = get_true_orbits_restricted(graph_edges,original_n)
            class_labels = partition_to_labels(classes,original_n)
            orbit_labels = partition_to_labels(true_orbits,original_n)

            print("G1-Classes: ", class_labels)
            print("G1-Orbits: ", orbit_labels)

            # nmi = normalized_mutual_info_score(class_labels, orbit_labels)
            # print(f"NMI: {nmi:.4f}")

            # file.write(f"Graph{instance}: NMI (with L(L(G))): {nmi:.4f}\n")


            # To get dirac operator
            # B1 = nx.incidence_matrix(G, oriented=True)
            # zero_nodes = sp.csr_matrix((B1.shape[0], B1.shape[0]))
            # zero_edges = sp.csr_matrix((B1.shape[1], B1.shape[1]))
            # dirac_mat = sp.bmat([[zero_nodes, B1], [B1.T, zero_edges]])
            # dirac_operator = dirac_mat.toarray()


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
            # # probability = compute_overall_same_orbit_probability(graph, classes, original_n)

            print(f"NMI: {nmi:.4f}")
            # # print(f"Same-orbit probability: {probability:.4f}")

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

            file.write(f"Graph{instance}: NMI: {nmi:.4f}, NMI with 1-WL: {heat_kwl_nmi:.4f}, #Orbits of G: {np.max(orbit_labels)+1}, Orbits: {orbit_labels}\n")


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
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print("Total running time: ", end_time - start_time)