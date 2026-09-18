from pathlib import Path
import re

from sklearn.metrics import normalized_mutual_info_score

from src.k_wl.kwl import run_kwl
from src.utils.graph_utils import *


GRAPH_DIR = Path("graphs/nauty_graphs/dimacs/grid")
OUTPUT_FILE = Path("results/kwl_results.txt")

MAX_K = 10


def natural_key(path):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", path.name)]


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    graph_files = sorted(
        (p for p in GRAPH_DIR.iterdir() if p.is_file()),
        key=natural_key,
    )

    with OUTPUT_FILE.open("w") as f:

        def write(text):
            print(text, end="")
            f.write(text)
            f.flush()

        for graph_file in graph_files:

            write(
                "\n" + "=" * 80 + "\n"
                f"Graph: {graph_file.name}\n"
                + "=" * 80 + "\n"
            )

            graph = read_dimacs_or_simple_graph(graph_file)
            n = graph.number_of_nodes()

            true_orbits = get_true_orbits_restricted(graph, n)
            orbit_labels = partition_to_labels(true_orbits, n)

            solved = False

            for k in range(1, MAX_K + 1):

                classes = run_kwl(
                    graph,
                    k,
                    output_dir=f"kwl_output/{graph_file.name}/k{k}",
                )

                classes = {i: cls for i, cls in enumerate(classes)}

                labels = partition_to_labels(classes, n)

                nmi = normalized_mutual_info_score(
                    orbit_labels,
                    labels,
                )

                probability = compute_overall_same_orbit_probability(
                    graph,
                    classes,
                    n,
                )

                write(
                    f"{k}-WL   "
                    f"NMI={nmi:.6f}   "
                    f"P={probability:.6f}   "
                    f"classes={len(classes)}\n"
                )

                if nmi >= 1 - 1e-12:
                    solved = True
                    write(f"\nSolved at {k}-WL\n")
                    break

            if solved:
                write("\nSTATUS: SOLVED\n")
            else:
                write(f"\nSTATUS: NOT SOLVED up to {MAX_K}-WL\n")


if __name__ == "__main__":
    main()