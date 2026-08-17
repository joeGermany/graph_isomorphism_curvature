from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score
import time
import networkx as nx
import math

# from src.k_wl.kwl import run_kwl
from src.module.heat_kernel_equivalence_compute_with_dirac import HeatKernelEquivalenceComputer
from src.utils.graph_utils_noPynauty import read_dimacs_or_simple_graph, subdivide_every_edge, restrict_to_original, partition_to_labels

def latin_graph(encoded):
    encoded = encoded.strip()

    # n^2 entries -> Latin square of order n
    n = math.isqrt(len(encoded))

    if n * n != len(encoded):
        raise ValueError(
            f"Invalid encoding: length {len(encoded)} is not a perfect square."
        )

    # Decode Latin square
    L = [
        list(encoded[i*n:(i+1)*n])
        for i in range(n)
    ]

    G = nx.Graph()

    # Each cell is a vertex
    vertices = [(i, j) for i in range(n) for j in range(n)]
    G.add_nodes_from(vertices)

    # Same row
    for i in range(n):
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                G.add_edge((i, j1), (i, j2))

    # Same column
    for j in range(n):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                G.add_edge((i1, j), (i2, j))

    # Same symbol
    for symbol in set(encoded):
        cells = [
            (i, j)
            for i in range(n)
            for j in range(n)
            if L[i][j] == symbol
        ]

        for a in range(len(cells)):
            for b in range(a + 1, len(cells)):
                G.add_edge(cells[a], cells[b])

    return G

def main():
    graph_type = "order8"

    with open(f"graphs/{graph_type}_isotopy_nontrivial.txt", "r") as f:
        for instance, line in enumerate(f, 1):
            with open(f"{graph_type}_output_round1_withoutL(L(G))_batch{(instance-1) // 5000}.txt", "a") as file:

                print(f"Graph{instance}")
                line = line.strip()
                if not line:
                    continue

                original_graph = latin_graph(line)
                original_graph = nx.convert_node_labels_to_integers(original_graph)

                subdivision_k=0

                graph_settings = [
                    {
                        "name": "G",
                        "signature_mode": "full_matrix",
                        "time_step": 0.1,
                        "end_time_multiplier": 100,
                        "fit_order": 6,
                        "fit_start": 1,
                        "fit_end": 6,
                        "resolutions": [1e10],
                    },
                    {
                        "name": "L(G)",
                        "signature_mode": "full_matrix",
                        "time_step": 0.1,
                        "end_time_multiplier": 40,
                        "fit_order": 6,
                        "fit_start": 1,
                        "fit_end": 4,
                        "resolutions": [1e10],
                    },
                    # {
                    #     "name": "L(L(G))",
                    #     "signature_mode": "full_matrix",
                    #     "time_step": 1,
                    #     "end_time_multiplier": 15,
                    #     "fit_order": 6,
                    #     "fit_start": 1,
                    #     "fit_end": 4,
                    #     "resolutions": [1e10],
                    # },
                ]

                equivalence_computer=HeatKernelEquivalenceComputer(
                    graph_settings=graph_settings,
                    comparison_mode="fixed_resolution",
                    bfs_depth=100,
                    weird_norm_threshold=1e-5,
                    zero_tolerance=1e-12,
                    use_ids=True,
                )

                original_n=original_graph.number_of_nodes()

                graph=subdivide_every_edge(original_graph,subdivision_k)

                all_classes,signatures=equivalence_computer.compute_vertex_signatures(graph)

                classes=restrict_to_original(all_classes,original_n)
                class_labels=partition_to_labels(classes,original_n)

                # print("Orbits: ", orbit_labels)
                print("Classes: ", class_labels)

                # nmi=normalized_mutual_info_score(class_labels,orbit_labels)
                # probability=compute_overall_same_orbit_probability(graph,classes,original_n)

                # print(f"NMI: {nmi:.4f}")
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

                # k_wl_parameter=1

                # # unit_kwl_classes=run_kwl(original_graph,k_wl_parameter,output_dir="kwl_output/unit_coloring",)
                # heat_kwl_classes=run_kwl(original_graph,k_wl_parameter,initial_partition=classes,output_dir="kwl_output/heat_colored",)

                # # unit_kwl_labels=partition_to_labels(unit_kwl_classes,original_n)
                # heat_kwl_labels=partition_to_labels(heat_kwl_classes,original_n)

                # print("Classes with 1-WL: ", heat_kwl_labels)

                # unit_kwl_nmi=normalized_mutual_info_score(orbit_labels,unit_kwl_labels)
                # heat_kwl_nmi=normalized_mutual_info_score(orbit_labels,heat_kwl_labels)

                # true_orbit_set={tuple(sorted(orbit)) for orbit in true_orbits}

                # unit_matches=sum(tuple(sorted(cls)) in true_orbit_set for cls in unit_kwl_classes)
                # heat_matches=sum(tuple(sorted(cls)) in true_orbit_set for cls in heat_kwl_classes)

                # print(f"{k_wl_parameter}-WL unit_coloring: NMI = {unit_kwl_nmi:.4f}, classes = {len(unit_kwl_classes)}, matching classes = {unit_matches}")
                # print(f"{k_wl_parameter}-WL heat_colored: NMI = {heat_kwl_nmi:.4f}, classes = {len(heat_kwl_classes)}, matching classes = {heat_matches}")

                # file.write(f"Graph{instance}: NMI: {nmi:.4f}, NMI with 1-WL: {heat_kwl_nmi:.4f}, #Orbits of G: {np.max(orbit_labels)+1}, Orbits: {orbit_labels}\n")

                file.write(f"Graph{instance}: Classes: {class_labels}\n") #, Classes with 1-WL: {heat_kwl_labels}\n")

if __name__=="__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print("Total running time: ", end_time - start_time)