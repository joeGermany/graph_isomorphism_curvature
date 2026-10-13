from pathlib import Path

from sklearn.metrics import normalized_mutual_info_score

from src.k_wl.kwl import run_kwl
from src.utils.graph_utils import *


GRAPH_FILE = Path("graphs/nauty_graphs/dimacs/pp/pp-9-2")
OUTPUT_FILE = Path("results/kwl_pp-9-2.txt")

MAX_K = 10
SPECIAL_VERTEX = 0


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    graph = read_dimacs_or_simple_graph(GRAPH_FILE)
    n = graph.number_of_nodes()

    true_orbits = get_true_orbits_restricted(graph, n)
    orbit_labels = partition_to_labels(true_orbits, n)

    initial_partition = [
        [SPECIAL_VERTEX],
        [v for v in graph.nodes() if v != SPECIAL_VERTEX],
    ]

    with OUTPUT_FILE.open("w") as f:

        def write(text):
            print(text, end="")
            f.write(text)
            f.flush()

        write(
            "=" * 80 + "\n"
            f"Graph: {GRAPH_FILE.name}\n"
            f"Initial coloring: {{{SPECIAL_VERTEX}}} | V\\{{{SPECIAL_VERTEX}}}\n"
            + "=" * 80 + "\n"
        )

        for k in range(1, MAX_K + 1):

            classes = run_kwl(
                graph,
                k,
                initial_partition=initial_partition,
                output_dir=f"kwl_output/{GRAPH_FILE.name}/vertex_{SPECIAL_VERTEX}/k{k}",
            )

            classes = {i: cls for i, cls in enumerate(classes)}
            labels = partition_to_labels(classes, n)

            nmi = normalized_mutual_info_score(orbit_labels, labels)

            probability = compute_overall_same_orbit_probability(
                graph,
                classes,
                n,
            )
            print(classes)

            write(
                f"{k}-WL   "
                f"NMI={nmi:.6f}   "
                f"P={probability:.6f}   "
                f"classes={len(classes)}\n"
            )

            if nmi >= 1 - 1e-12:
                write(f"\nSolved at {k}-WL\n")
                write("\nSTATUS: SOLVED\n")
                break

        else:
            write(f"\nSTATUS: NOT SOLVED up to {MAX_K}-WL\n")


if __name__ == "__main__":
    main()