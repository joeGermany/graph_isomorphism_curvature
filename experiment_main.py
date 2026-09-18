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


START_RESOLUTION = 1e8
MIN_RESOLUTION = 1

BASE_SETTING = {
    "bfs_sort": True,
    "signature_mode": "full_matrix",
    "time_step": 1,
    "end_time_multiplier": 7,
    "fit_order": 6,
    "fit_start": 1,
    "fit_end": 6,
    "resolutions": [START_RESOLUTION],
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
                orbit_labels = partition_to_labels(
                    get_true_orbits_restricted(G, n), n
                )

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
                    nonlocal solved

                    nmi = normalized_mutual_info_score(
                        orbit_labels, partition_to_labels(classes, n)
                    )
                    p = compute_overall_same_orbit_probability(
                        G, classes, n
                    )

                    write(
                        f"{stage:<42} NMI={nmi:.6f}   "
                        f"P={p:.6f}   classes={len(classes)}\n"
                    )

                    if p > 1e-12 and nmi >= 1 - 1e-12:
                        solved = True
                        write(f"\nSolved at {stage}\n")

                    return p

                for order in range(MAX_LINE_ORDER + 1):
                    name = (
                        "G" if order == 0 else
                        "L(G)" if order == 1 else
                        f"L^{order}(G)"
                    )
                    line_t = [] if order == 0 else [("line", order)]

                    base_graph, membership = construct_transformed_graph(
                        G, line_t
                    )

                    if nx.is_connected(base_graph):
                        diameter = nx.diameter(base_graph)
                    else:
                        diameter = max(
                            nx.diameter(base_graph.subgraph(c))
                            for c in nx.connected_components(base_graph)
                        )

                    write(f"\n--- {name}, diameter={diameter} ---\n")

                    stages = [(name, line_t)] + [
                        (f"{name}_{d}", [("distance", d)] + line_t)
                        for d in range(2, diameter + 1)
                    ]

                    # Laplacian -> Correlation -> Dirac
                    for stage_name, transforms in stages:
                        for op in OPERATORS:

                            # Save partition before this stage
                            previous_classes = (
                                None if classes is None else
                                {i: list(cls) for i, cls in classes.items()}
                            )

                            resolution = START_RESOLUTION

                            while True:
                                # Restore partition before retrying
                                classes = (
                                    None if previous_classes is None else
                                    {i: list(cls)
                                     for i, cls in previous_classes.items()}
                                )

                                s = {
                                    **BASE_SETTING,
                                    "name": f"{stage_name}-{op}",
                                    "transformations": transforms,
                                    "operators": [op],
                                    "resolutions": [resolution],
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
                                p = evaluate(s["name"])

                                if solved or p > 1e-12:
                                    break

                                if resolution <= MIN_RESOLUTION:
                                    aborted = True
                                    write(
                                        f"\nABORTED at {s['name']}: "
                                        f"P=0 even at r={resolution:.0e}\n"
                                    )
                                    break

                                resolution /= 10
                                write(
                                    f"P=0 at {s['name']}; "
                                    f"retrying with r={resolution:.0e}\n"
                                )

                            if solved or aborted:
                                break

                        if solved or aborted:
                            break

                    if solved or aborted:
                        break

                    # Edge-addition probe directly on G / L(G) / L²(G)
                    previous_classes = (
                        None if classes is None else
                        {i: list(cls) for i, cls in classes.items()}
                    )

                    resolution = START_RESOLUTION

                    while True:
                        classes = (
                            None if previous_classes is None else
                            {i: list(cls)
                             for i, cls in previous_classes.items()}
                        )

                        probe_setting = {
                            **BASE_SETTING,
                            "name": f"{name}-probe",
                            "transformations": [],
                            "operators": ["laplacian"],
                            "resolutions": [resolution],
                        }

                        probe_computer = HeatKernelEquivalenceComputer(
                            graph_settings=[probe_setting],
                            comparison_mode="fixed_resolution",
                            bfs_depth=100,
                            use_ids=True,
                            use_stable_ids=True,
                        )

                        profiles = edge_addition(
                            base_graph,
                            probe_computer,
                            list(base_graph.nodes()),
                            M=1
                        )

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
                        stage = f"{name}-probe-edge_addition"
                        p = evaluate(stage)

                        if solved or p > 1e-12:
                            break

                        if resolution <= MIN_RESOLUTION:
                            aborted = True
                            write(
                                f"\nABORTED at {stage}: "
                                f"P=0 even at r={resolution:.0e}\n"
                            )
                            break

                        resolution /= 10
                        write(
                            f"P=0 at {stage}; "
                            f"retrying with r={resolution:.0e}\n"
                        )

                    if solved or aborted:
                        break

                if solved:
                    write("\nSTATUS: SOLVED\n")
                elif aborted:
                    write("\nSTATUS: ABORTED — P=0 AT MINIMUM RESOLUTION\n")
                else:
                    write(
                        f"\nSTATUS: NOT SOLVED up to "
                        f"L^{MAX_LINE_ORDER}(G)\n"
                    )

            except MemoryError as e:
                write(
                    f"\nMEMORY ERROR on {graph_file.name}: {e}\n"
                    f"{traceback.format_exc()}\n"
                    "STATUS: SKIPPED — MEMORY ERROR\n"
                )
                continue


if __name__ == "__main__":
    main()