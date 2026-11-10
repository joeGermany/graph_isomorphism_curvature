from pathlib import Path
from collections import defaultdict
import re

from sklearn.metrics import normalized_mutual_info_score
from src.k_wl import k_wl
from src.utils.graph_utils import *

GRAPH_DIR = Path("graphs/nauty_graphs/k")
OUTPUT_FILE = Path("results/kwl_k.txt")
MAX_K = 10
MAX_TUPLES = 50_000_000


#TO RUN K-WL:
# python -m pip install pybind11
# bash src/k_wl/build_k_wl.sh

def natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]
def get_max_tuples():
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
        group = next(
            line.split("::", 1)[1]
            for line in lines
            if line.startswith("0::")
        )

        path = Path("/sys/fs/cgroup") / group.lstrip("/")
        limit = (path / "memory.max").read_text().strip()

        if limit == "max":
            return MAX_TUPLES

        used = int((path / "memory.current").read_text())
        available = max(0, int(limit) - used)

        # Heuristic: use at most 40% of remaining memory,
        # assuming approximately 256 bytes per tuple.
        return int(available * 0.8 / 256)

    except (OSError, ValueError, StopIteration):
        return MAX_TUPLES

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    graph_files = sorted((p for p in GRAPH_DIR.iterdir() if p.is_file()), key=natural_key)

    with OUTPUT_FILE.open("w") as f:
        def write(text):
            print(text, end="")
            f.write(text)
            f.flush()

        for graph_file in graph_files:
            write(f"\n{'=' * 80}\nGraph: {graph_file.name}\n{'=' * 80}\n")

            graph = read_dimacs_or_simple_graph(graph_file)
            n = graph.number_of_nodes()

            try:
                true_orbits = load_traces_orbits(graph_file, n)
                print("loaded traces orbits")
            except (FileNotFoundError, ValueError, KeyError) as e:
                print(f"Could not load Traces orbits for {graph_file.name}: {e}")
                print("Computing orbits instead...", flush=True)
                true_orbits = get_true_orbits_restricted(graph, n)

            orbit_labels = partition_to_labels(true_orbits, n)
            solved = False

            for k in range(1, MAX_K + 1):
                tuple_limit = get_max_tuples()
                print("tuple limit: ", tuple_limit)
                if n ** k > tuple_limit:
                    write(
                        f"{k}-WL SKIPPED: {n ** k:,} tuples "
                        f"exceed estimated safe limit {tuple_limit:,}\n"
                    )
                    break

                colors = k_wl(graph, k, max_tuples=tuple_limit)

                groups = defaultdict(list)
                for v, color in colors.items():
                    groups[color].append(v)

                classes = dict(enumerate(groups.values()))
                labels = partition_to_labels(classes, n)

                nmi = normalized_mutual_info_score(orbit_labels, labels)
                probability = compute_overall_same_orbit_probability(graph, classes, n)

                write(f"{k}-WL   NMI={nmi:.6f}   P={probability:.6f}   classes={len(classes)}\n")

                if nmi >= 1 - 1e-12:
                    solved = True
                    write(f"\nSolved at {k}-WL\n")
                    break

            write("\nSTATUS: SOLVED\n" if solved else "\nSTATUS: NOT SOLVED\n")


if __name__ == "__main__":
    main()