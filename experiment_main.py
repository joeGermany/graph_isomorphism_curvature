from pathlib import Path
import re
import traceback
from sklearn.metrics import normalized_mutual_info_score

from src.module.heat_kernel_equivalence_compute import HeatKernelEquivalenceComputer
from src.utils.graph_utils import *
from src.probing.probing import *


GRAPH_DIR = Path("graphs/nauty_graphs/dimacs/grid")
OUTPUT_FILE = Path("results/results_grid.txt")

MAX_LINE_ORDER = 2
SUBDIVISION_K = 0
OPERATORS = ["laplacian", "correlation", "dirac"]

BASE_SETTING = {
    "bfs_sort": True,
    "signature_mode": "full_matrix",
    "time_step": 1,
    "end_time_multiplier": 7,
    "fit_order": 6,
    "fit_start": 1,
    "fit_end": 6,
    "resolutions": [1e6],
    "dirac_decimals": 6,
    "precision_dps": None,
}


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(
        (p for p in GRAPH_DIR.iterdir() if p.is_file()),
        key=lambda p: [int(x) if x.isdigit() else x.lower()
                       for x in re.split(r"(\d+)", p.name)]
    )

    with OUTPUT_FILE.open("w") as f:

        def write(s):
            print(s, end="")
            f.write(s)
            f.flush()

        for graph_file in files:
            try:
                G0 = read_dimacs_or_simple_graph(graph_file)
                n = G0.number_of_nodes()
                G = subdivide_every_edge(G0, SUBDIVISION_K)
                orbit_labels = partition_to_labels(get_true_orbits_restricted(G, n), n)

                classes = None
                solved = aborted = False

                write(
                    "\n" + "=" * 80 +
                    f"\nGraph: {graph_file.name}\n"
                    f"Original number of vertices: {n}\n" +
                    "=" * 80 + "\n"
                )

                def refine(new):
                    nonlocal classes
                    if classes is None:
                        classes = new
                        return

                    labels = {v: i for i, cls in new.items() for v in cls}
                    out = []
                    for cls in classes.values():
                        groups = {}
                        for v in cls:
                            groups.setdefault(labels[v], []).append(v)
                        out.extend(groups.values())
                    classes = dict(enumerate(out))

                def evaluate(stage):
                    nonlocal solved, aborted

                    nmi = normalized_mutual_info_score(
                        orbit_labels, partition_to_labels(classes, n)
                    )
                    p = compute_overall_same_orbit_probability(G, classes, n)

                    write(
                        f"{stage:<42} NMI={nmi:.6f}   "
                        f"P={p:.6f}   classes={len(classes)}\n"
                    )

                    if p <= 1e-12:
                        aborted = True
                        write(f"\nABORTED at {stage}: P=0\n")
                    elif nmi >= 1 - 1e-12:
                        solved = True
                        write(f"\nSolved at {stage}\n")

                for order in range(MAX_LINE_ORDER + 1):
                    name = (
                        "G" if order == 0 else
                        "L(G)" if order == 1 else
                        f"L^{order}(G)"
                    )
                    line_t = [] if order == 0 else [("line", order)]

                    # Actual G / L(G) / L²(G)
                    base_graph, membership = construct_transformed_graph(G, line_t)
                    if nx.is_connected(base_graph):
                        diameter = nx.diameter(base_graph)
                    else:
                        diameter = max(
                            nx.diameter(base_graph.subgraph(c))
                            for c in nx.connected_components(base_graph)
                        )

                    write(f"\n--- {name}, diameter={diameter} ---\n")

                    # Base graph, then distance graphs up to its diameter
                    stages = [(name, line_t)] + [
                        (f"{name}_{d}", [("distance", d)] + line_t)
                        for d in range(2, diameter + 1)
                    ]

                    # Laplacian -> Correlation -> Dirac
                    for stage_name, transforms in stages:
                        for op in OPERATORS:
                            s = {
                                **BASE_SETTING,
                                "name": f"{stage_name}-{op}",
                                "transformations": transforms,
                                "operators": [op],
                            }

                            computer = HeatKernelEquivalenceComputer(
                                graph_settings=[s],
                                comparison_mode="fixed_resolution",
                                bfs_depth=1000,
                                use_ids=True,
                                use_stable_ids=False,
                            )

                            new, _ = computer.compute_vertex_signatures(G)
                            refine(restrict_to_original(new, n))
                            evaluate(s["name"])

                            if solved or aborted:
                                break

                        if solved or aborted:
                            break

                    if solved or aborted:
                        break

                    # Edge-addition probe directly on G / L(G) / L²(G)
                    probe_setting = {
                        **BASE_SETTING,
                        "name": f"{name}-probe",
                        "transformations": [],
                        "operators": ["laplacian"],
                    }

                    probe_computer = HeatKernelEquivalenceComputer(
                        graph_settings=[probe_setting],
                        comparison_mode="fixed_resolution",
                        bfs_depth=100,
                        use_ids=True,
                        use_stable_ids=True,
                    )

                    profiles = edge_addition(
                        base_graph, probe_computer,
                        list(base_graph.nodes()), M=1
                    )

                    # Map probe information back to original G vertices
                    probe_signatures = {
                        v: tuple(sorted(
                            (profiles[u] for u in membership[v]),
                            key=repr
                        ))
                        for v in G.nodes()
                    }

                    groups = {}
                    for v, sig in probe_signatures.items():
                        groups.setdefault(sig, []).append(v)

                    refine(dict(enumerate(groups.values())))
                    evaluate(f"{name}-probe-edge_addition")

                    if solved or aborted:
                        break

                if solved:
                    write("\nSTATUS: SOLVED\n")
                elif aborted:
                    write("\nSTATUS: ABORTED — P=0\n")
                else:
                    write(f"\nSTATUS: NOT SOLVED up to L^{MAX_LINE_ORDER}(G)\n")

            except MemoryError as e:
                write(
                    f"\nMEMORY ERROR on {graph_file.name}: {e}\n"
                    f"{traceback.format_exc()}\n"
                    "STATUS: STOPPED — MEMORY ERROR\n"
                )
                continue


if __name__ == "__main__":
    main()