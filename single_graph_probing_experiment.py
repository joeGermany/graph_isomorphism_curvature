from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score

from src.module.heat_kernel_equivalence_compute import HeatKernelEquivalenceComputer
from src.utils.graph_utils import *
from src.probing.probing import *


GRAPH_FILE = Path("graphs/nauty_graphs/dimacs/pp/pp-9-2")
OUTPUT_FILE = Path("results/pp-9-2_probing.txt")

PROBES = [
    # ("edge_addition", edge_addition, {"M": 1}),
    # ("edge_deletion", edge_deletion, {"M": 1}),
    ("folding", folding, {"p": 2, "M": 1}),
    ("edge_swap", edge_swap, {"M": 1}),
    ("clique_addition", clique_addition, {"p": 2, "c": 2, "M": 1}),
]


def main():
    G = read_dimacs_or_simple_graph(GRAPH_FILE)
    n = G.number_of_nodes()
    orbit_labels = partition_to_labels(get_true_orbits_restricted(G, n), n)
    classes = {0: list(G.nodes())}

    setting = {
        "name": "G-probe",
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
        "precision_dps": None,
    }

    computer = HeatKernelEquivalenceComputer(
        graph_settings=[setting],
        comparison_mode="fixed_resolution",
        bfs_depth=100,
        use_ids=True,
        use_stable_ids=True,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w") as f:
        def write(s):
            print(s, end="")
            f.write(s)
            f.flush()

        for name, probe, kwargs in PROBES:
            classes = probe_partition(G, computer, classes, probe, **kwargs)

            nmi = normalized_mutual_info_score(
                orbit_labels, partition_to_labels(classes, n)
            )
            p = compute_overall_same_orbit_probability(G, classes, n)

            write(
                f"{name:<18} NMI={nmi:.6f}   "
                f"P={p:.6f}   classes={len(classes)}\n"
            )


if __name__ == "__main__":
    main()