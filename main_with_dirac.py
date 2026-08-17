from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score

from src.k_wl.kwl import run_kwl
from src.module.heat_kernel_equivalence_compute_with_dirac import HeatKernelEquivalenceComputer
from src.utils.graph_utils import *


def main():
    graph_file=Path("graphs/35-16-6-8/Graph625")

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
    # graph_settings determines which line-graph levels are included:
    #   1 setting  -> G
    #   2 settings -> G, L(G)
    #   3 settings -> G, L(G), L(L(G))
    #   etc.
    #
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
            "time_step": 1,
            "end_time_multiplier": 15,
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
    # H, mapping = randomly_permute_graph(graph)

    true_orbits=get_true_orbits_restricted(graph,original_n)
    orbit_labels=partition_to_labels(true_orbits,original_n)

    print(orbit_labels)
    # Optional to pass H
    all_classes,signatures=equivalence_computer.compute_vertex_signatures_with_dirac(graph)

    classes=restrict_to_original(all_classes,original_n)

    class_labels=partition_to_labels(classes,original_n)

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