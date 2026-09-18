from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score

from src.k_wl.kwl import run_kwl
from src.module.heat_kernel_equivalence_compute import HeatKernelEquivalenceComputer

from src.utils.graph_utils import *
from src.probing.probing import *


def main():
    graph_file=Path("graphs/nauty_graphs/pp/pp-9-2")

    # signature_mode:
    #   "coefficients" -> fitted coefficients; set fit_order/fit_start/fit_end
    #   "diagonal"     -> heat-diagonal values
    #   "full_matrix"  -> ordered full heat rows
    #
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
    #
    # end_time_multiplier controls the upper end of the sampled time interval:
    #   1 -> end_time = 1 / lambda_min
    #
    # transformations:
    #   []                              -> G
    #   [("line",1)]                    -> L(G)
    #   [("distance",2)]                -> G2
    #   [("line",1),("distance",2)]     -> L(G2)
    #   [("line",2),("distance",2)]     -> L(L(G2))
    #   Transformations are written as the final expression is read.
    # operators:
    #   "laplacian"    -> heat-kernel signatures
    #   "dirac"        -> Dirac signatures
    #   "topological"  -> structural/topological measures
    #   "correlation"  -> Laplacian pseudoinverse correlation signatures
    #   []             -> skip signature computation for this graph

    # precision_dps:
    #   None -> use standard NumPy float64 precision
    #   N    -> use mpmath arbitrary precision with N decimal digits

    subdivision_k=0

    graph_settings = [
        {
            "name": "G",
            "transformations": [],
            "operators": ["dirac_correlation"],
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
        #     "operators": ["dirac_correlation"],
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

    original_graph=read_dimacs_or_simple_graph(graph_file)
    original_n=original_graph.number_of_nodes()

    graph=subdivide_every_edge(original_graph,subdivision_k)
    H, mapping = randomly_permute_graph(graph)

    # Optional to pass H
    all_classes,signatures=equivalence_computer.compute_vertex_signatures(graph)

    classes=restrict_to_original(all_classes,original_n)
    true_orbits=get_true_orbits_restricted(graph,original_n)

    class_labels=partition_to_labels(classes,original_n)
    orbit_labels=partition_to_labels(true_orbits,original_n)

    print("classes: ", class_labels)
    print("orbits: ", orbit_labels)

    nmi=normalized_mutual_info_score(class_labels,orbit_labels)
    probability=compute_overall_same_orbit_probability(graph,classes,original_n)

    print(f"NMI: {nmi:.4f}")
    print(f"Same-orbit probability: {probability:.4f}")

    # -----------------------
    # Probing
    # -----------------------

    # probing_classes = probe_partition(graph, equivalence_computer, classes, clique_addition, p=2, c=2, M=1)
    # probing_class_labels = partition_to_labels(probing_classes,original_n)
    # probing_nmi = normalized_mutual_info_score(probing_class_labels, orbit_labels)
    # probing_probability = compute_overall_same_orbit_probability(graph, probing_classes, original_n)
    # print("Probing classes: ", probing_class_labels)
    # print(f"NMI After Probing: {probing_nmi:.4f}")
    # print(f"Same-orbit probability After Probing: {probing_probability:.4f}")



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

    unit_kwl_classes=run_kwl(original_graph,k_wl_parameter,output_dir="kwl_output/unit_coloring",)

    heat_kwl_classes=run_kwl(original_graph,k_wl_parameter,initial_partition=classes,output_dir="kwl_output/heat_colored",)

    unit_kwl_labels=partition_to_labels(unit_kwl_classes,original_n)
    heat_kwl_labels=partition_to_labels(heat_kwl_classes,original_n)

    unit_kwl_nmi=normalized_mutual_info_score(orbit_labels,unit_kwl_labels)
    heat_kwl_nmi=normalized_mutual_info_score(orbit_labels,heat_kwl_labels)

    true_orbit_set={tuple(sorted(orbit)) for orbit in true_orbits}

    unit_matches=sum(tuple(sorted(cls)) in true_orbit_set for cls in unit_kwl_classes)

    heat_matches=sum(tuple(sorted(cls)) in true_orbit_set for cls in heat_kwl_classes)

    print(
        f"{k_wl_parameter}-WL unit_coloring: "
        f"NMI = {unit_kwl_nmi:.4f}, "
        f"classes = {len(unit_kwl_classes)}, "
        f"matching classes = {unit_matches}"
    )

    print(
        f"{k_wl_parameter}-WL heat_colored: "
        f"NMI = {heat_kwl_nmi:.4f}, "
        f"classes = {len(heat_kwl_classes)}, "
        f"matching classes = {heat_matches}"
    )


if __name__=="__main__":
    main()