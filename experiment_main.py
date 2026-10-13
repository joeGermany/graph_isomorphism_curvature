from pathlib import Path
import re
import traceback
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score

from src.module.heat_kernel_equivalence_compute import HeatKernelEquivalenceComputer
from src.utils.graph_utils import *
from src.probing.probing import *


# GRAPH_FILE = Path("graphs/nauty_graphs/pp/pp-9-2")
# OUTPUT_FILE = Path("results/pp-9-2-L(L(G))-run.txt")

GRAPH_DIR = Path("graphs/nauty_graphs/dimacs/mz-aug")
OUTPUT_DIR = Path("results/mz-aug")

LINE_ORDER_START = 0
MAX_LINE_ORDER = 2
SUBDIVISION_K = 0

OPERATORS = ["laplacian", "correlation","dirac"]

START_RESOLUTION = 1e8
MIN_RESOLUTION = 1
EPS = 1e-12


PROBES = [
    ("neighborhood_deletion", neighborhood_deletion, {"p": 3, "max_distance": 3}),
    # ("single_clique", single_clique_addition, {"c": 40}),
    # ("folding", folding, {"p": 2, "M": 1}),
    # ("node_deletion", node_deletion, {}),
    # ("edge_addition", edge_addition, {"M": 1}),
    # ("edge_deletion", edge_deletion, {"M": 1}),
    # ("edge_swap", edge_swap, {"M": 1}),
    # ("clique_addition", clique_addition, {"p": 4, "c": 10, "M": 1}),

]


BASE_SETTING = {
    "bfs_sort": True,
    "signature_mode": "full_matrix",
    "time_step": 1,
    "end_time_multiplier": 7,
    "fit_order": 6,
    "fit_start": 1,
    "fit_end": 6,
    "precision_dps": None,
}


def copy_classes(classes):
    return None if classes is None else {i: list(cls) for i, cls in classes.items()}


def process_graph(graph_file):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{graph_file.name}_output.txt"

    with output_file.open("w") as f:

        def write(s):
            print(s, end="")
            f.write(s)
            f.flush()

        try:

            G0 = read_dimacs_or_simple_graph(graph_file)
            n = G0.number_of_nodes()
            G = subdivide_every_edge(G0, SUBDIVISION_K)

            orbit_labels = partition_to_labels(get_true_orbits_restricted(G, n), n)

            classes = None
            solved = False
            aborted = False

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
                refined = []

                for cls in classes.values():
                    groups = {}
                    for v in cls:
                        groups.setdefault(labels[v], []).append(v)
                    refined.extend(groups.values())

                classes = dict(enumerate(refined))



            def evaluate(stage):
                nonlocal solved

                nmi = normalized_mutual_info_score(orbit_labels, partition_to_labels(classes, n))
                p = compute_overall_same_orbit_probability(G, classes, n)

                write(f"{stage:<45} NMI={nmi:.6f}   P={p:.6f}   classes={len(classes)}\n")

                if p > EPS and nmi >= 1 - EPS:
                    solved = True
                    write(f"\nSolved at {stage}\n")

                return p


            # Retry one experiment at decreasing resolution

            def run_with_resolution(stage, compute_partition):
                nonlocal classes, aborted

                before = copy_classes(classes)
                resolution = START_RESOLUTION

                while True:
                    classes = copy_classes(before)

                    new_partition = compute_partition(resolution)
                    refine(new_partition)

                    p = evaluate(stage)

                    if solved or p > EPS:
                        return

                    if resolution <= MIN_RESOLUTION:
                        aborted = True
                        write(f"\nABORTED at {stage}: P=0 even at r={resolution:.0e}\n")
                        return

                    resolution /= 10
                    write(f"P=0 at {stage}; retrying with r={resolution:.0e}\n")


            # G, L(G), L²(G), ...

            for order in range(LINE_ORDER_START, MAX_LINE_ORDER + 1):

                name = "G" if order == 0 else "L(G)" if order == 1 else f"L^{order}(G)"
                line_t = [] if order == 0 else [("line", order)]

                line_graph, _ = construct_transformed_graph(G, line_t)

                if nx.is_connected(line_graph):
                    diameter = nx.diameter(line_graph)
                else:
                    diameter = max(nx.diameter(line_graph.subgraph(c)) for c in nx.connected_components(line_graph))

                write(f"\n================ {name}, diameter={diameter} ================\n")

                # G, G_2, G_3, ...
                # L(G), L(G)_2, L(G)_3, ...
                stages = [(name, line_t)]
                stages += [(f"{name}_{d}", [("distance", d)] + line_t) for d in range(2, diameter + 1)]

                for stage_name, transforms in stages:

                    write(f"\n--- {stage_name} ---\n")

                    stage_graph, membership = construct_transformed_graph(G, transforms)



                    for op in OPERATORS:

                        def spectral_partition(resolution, op=op):
                            setting = {
                                **BASE_SETTING,
                                "name": f"{stage_name}-{op}",
                                "transformations": transforms,
                                "operators": [op],
                                "resolutions": [resolution],
                            }

                            computer = HeatKernelEquivalenceComputer(
                                graph_settings=[setting],
                                comparison_mode="fixed_resolution",
                                bfs_depth=1000,
                                use_ids=True,
                                use_stable_ids=False,
                            )

                            new, _ = computer.compute_vertex_signatures(G)
                            return restrict_to_original(new, n)

                        run_with_resolution(f"{stage_name}-{op}", spectral_partition)

                        if solved or aborted:
                            break

                    if solved or aborted:
                        break


                    # 2. Probing on THIS SAME graph

                    # if stage_name == "G":
                    #     continue
                    for probe_name, probe, probe_args in PROBES:

                        def probe_partition(resolution, probe=probe, probe_args=probe_args):

                            setting = {
                                **BASE_SETTING,
                                "name": f"{stage_name}-probe-{probe_name}",
                                "transformations": [],
                                "operators": ["laplacian", "correlation"],
                                "resolutions": [resolution],
                            }

                            computer = HeatKernelEquivalenceComputer(
                                graph_settings=[setting],
                                comparison_mode="fixed_resolution",
                                bfs_depth=100,
                                use_ids=True,
                                use_stable_ids=True,
                            )

                            # Probe the ACTUAL current graph: G, G_2, L(G), L(G)_2, ...
                            profiles = probe(
                                stage_graph,
                                computer,
                                list(stage_graph.nodes()),
                                **probe_args
                            )

                            # Map transformed-graph profiles back to
                            # the original vertices.
                            signatures = {
                                v: tuple(sorted((profiles[u] for u in membership[v]), key=repr))
                                for v in G.nodes()
                            }

                            groups = {}
                            for v, sig in signatures.items():
                                groups.setdefault(sig, []).append(v)

                            return dict(enumerate(groups.values()))

                        run_with_resolution(
                            f"{stage_name}-probe-{probe_name}",
                            probe_partition
                        )

                        if solved or aborted:
                            break

                    if solved or aborted:
                        break

                if solved or aborted:
                    break

            # ----------------------------------------------------------
            # Final status
            # ----------------------------------------------------------

            if solved:
                write("\nSTATUS: SOLVED\n")
            elif aborted:
                write("\nSTATUS: ABORTED — P=0 AT MINIMUM RESOLUTION\n")
            else:
                write(f"\nSTATUS: NOT SOLVED up to L^{MAX_LINE_ORDER}(G)\n")

        except MemoryError as e:
            write(
                f"\nMEMORY ERROR on {graph_file.name}: {e}\n"
                f"{traceback.format_exc()}\n"
                "STATUS: SKIPPED — MEMORY ERROR\n"
            )


def main():
    files = sorted(
        (p for p in GRAPH_DIR.rglob("*") if p.is_file()),
        key=lambda p: [int(x) if x.isdigit() else x.lower()
                       for x in re.split(r"(\d+)", str(p.relative_to(GRAPH_DIR)))]
    )

    print(f"Found {len(files)} graphs in {GRAPH_DIR}")

    for i, graph_file in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing {graph_file.name}")
        try:
            process_graph(graph_file)
        except Exception:
            print(f"ERROR processing {graph_file.name}")
            traceback.print_exc()



if __name__ == "__main__":
    main()
