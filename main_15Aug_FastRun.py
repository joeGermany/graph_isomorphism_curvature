from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score
import time

from src.k_wl.kwl import run_kwl
from src.module.heat_kernel_equivalence_compute_with_dirac import HeatKernelEquivalenceComputer
from src.utils.graph_utils import *


def main():
    graph_type = "order7"
    with open(f"latin_graphs_isotopy_nontrivial_{graph_type}_output_round1_withoutL(L(G)).txt", "w") as file:
        for instance in range(1, 150):
            # [17]:
            # range(1, 6761):
            # [15, 16, 29, 31, 34, 38, 41, 46, 47, 48, 50, 52, 73, 78]:
            # [48, 52, 57, 60, 65, 66, 67, 70, 103, 120, 124, 134, 136, 144, 145, 146, 147, 148, 153, 155, 157, 159, 160, 161, 163, 164, 167, 169, 170, 172, 173, 174, 176, 177, 178, 179]:
            # [625, 647, 707, 1062, 1174, 1346, 1593, 1618, 1624, 1735, 1756, 1847, 2432, 2666, 3263, 3291, 3292]:
            # [1,3,5,12,13,15,17,21]:
            # [5306, 7185, 7360, 10133, 10266, 10408, 10411, 10413, 10414, 10415, 10879, 10904, 11064, 13011, 15310, 17286, 21091, 21092, 22646, 22656, 23842, 24208, 24247, 25142, 25385, 27748, 27755, 27778, 28522, 28524, 28525, 28594, 28629, 28860, 28861, 28867, 28882, 28883, 28884, 28885, 28886]:
            print(f"Graph{instance}")
            graph_file=Path(f"graphs/latin_graphs_isotopy_nontrivial/{graph_type}/Graph{instance}")

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
                    "end_time_multiplier": 20,
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

            original_graph=read_dimacs_or_simple_graph(graph_file)
            original_n=original_graph.number_of_nodes()

            graph=subdivide_every_edge(original_graph,subdivision_k)

            all_classes,signatures=equivalence_computer.compute_vertex_signatures(graph)

            classes=restrict_to_original(all_classes,original_n)
            true_orbits=get_true_orbits_restricted(graph,original_n)

            class_labels=partition_to_labels(classes,original_n)
            orbit_labels=partition_to_labels(true_orbits,original_n)

            print("Orbits: ", orbit_labels)
            print("Classes: ", class_labels)

            nmi=normalized_mutual_info_score(class_labels,orbit_labels)
            probability=compute_overall_same_orbit_probability(graph,classes,original_n)

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

            k_wl_parameter=1

            # unit_kwl_classes=run_kwl(original_graph,k_wl_parameter,output_dir="kwl_output/unit_coloring",)
            heat_kwl_classes=run_kwl(original_graph,k_wl_parameter,initial_partition=classes,output_dir="kwl_output/heat_colored",)

            # unit_kwl_labels=partition_to_labels(unit_kwl_classes,original_n)
            heat_kwl_labels=partition_to_labels(heat_kwl_classes,original_n)

            # unit_kwl_nmi=normalized_mutual_info_score(orbit_labels,unit_kwl_labels)
            heat_kwl_nmi=normalized_mutual_info_score(orbit_labels,heat_kwl_labels)

            true_orbit_set={tuple(sorted(orbit)) for orbit in true_orbits}

            # unit_matches=sum(tuple(sorted(cls)) in true_orbit_set for cls in unit_kwl_classes)
            heat_matches=sum(tuple(sorted(cls)) in true_orbit_set for cls in heat_kwl_classes)

            # print(f"{k_wl_parameter}-WL unit_coloring: NMI = {unit_kwl_nmi:.4f}, classes = {len(unit_kwl_classes)}, matching classes = {unit_matches}")
            print(f"{k_wl_parameter}-WL heat_colored: NMI = {heat_kwl_nmi:.4f}, classes = {len(heat_kwl_classes)}, matching classes = {heat_matches}")

            file.write(f"Graph{instance}: NMI: {nmi:.4f}, NMI with 1-WL: {heat_kwl_nmi:.4f}, #Orbits of G: {np.max(orbit_labels)+1}\n")

            file.flush()

if __name__=="__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print("Total running time: ", end_time - start_time)