from pathlib import Path
import re
import traceback
import numpy as np
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score

from src.module.heat_kernel_equivalence_compute import HeatKernelEquivalenceComputer
from src.utils.graph_utils import *
from src.probing.probing import *

GRAPH_DIR = Path("graphs/nauty_graphs/cfi")
OUTPUT_DIR = Path("results/cfi")

LINE_ORDER_START = 0
MAX_LINE_ORDER = 2
SUBDIVISION_K = 0

OPERATORS = ["laplacian","correlation","dirac"]

START_RESOLUTION = 1e8
MIN_RESOLUTION = 1
EPS = 1e-12

PROBE_ORBIT_REP = False

PROBES = [

    ("folding", folding, {"p": 2, "M": 1}),
    ("node_deletion", node_deletion, {"M": 1}),
    ("single_clique", single_clique_addition, {"c": 40}),
    ("neighborhood_deletion", neighborhood_deletion, {"p": 3, "max_distance": 2, "distance_mode": "equal"}),
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
            try:
                true_orbits = load_traces_orbits(graph_file, n)
            except (FileNotFoundError, ValueError, KeyError) as e:
                print(f"Could not load Traces orbits for {graph_file.name}: {e}")
                print("Computing orbits instead...", flush=True)
                true_orbits = get_true_orbits_restricted(G, n)
            traces_generators = None  # Load lazily only if G-neighborhood deletion is reached
            orbit_labels = partition_to_labels(true_orbits, n)
            orbit_reps = {i: next(iter(cls)) for i, cls in enumerate(true_orbits)}
            vertex_orbit = {v: i for i, cls in enumerate(true_orbits) for v in cls}

            classes = None
            solved = False
            aborted = False

            write("\n" + "=" * 80 + f"\nGraph: {graph_file.name}\nOriginal number of vertices: {n}\n" + "=" * 80 + "\n")

            computer = HeatKernelEquivalenceComputer(graph_settings=[
                {**BASE_SETTING, "name": "initial", "transformations": [], "operators": ["laplacian"],
                 "resolutions": [START_RESOLUTION]}], comparison_mode="fixed_resolution", bfs_depth=1000, use_ids=True,
                                                     use_stable_ids=False)

            def configure(name, transformations, operators, resolution, bfs_depth=1000, stable_ids=False):
                computer.graph_settings = [
                    {**BASE_SETTING, "name": name, "transformations": transformations, "operators": operators,
                     "resolutions": np.asarray([resolution], dtype=np.float64)}]
                computer.bfs_depth = bfs_depth
                computer.use_ids = True
                computer.use_stable_ids = stable_ids

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
                p = compute_overall_same_orbit_probability(G, classes, n, true_orbits=true_orbits)
                # print(f"\n=== {stage} ===")
                #
                # print("\nTrue orbits:")
                # for i, orbit in enumerate(true_orbits):
                #     print(f"Orbit {i}: {sorted(orbit)}")
                #
                # print("\nComputed classes:")
                # for i, cls in enumerate(classes.values()):
                #     print(f"Class {i}: {sorted(cls)}")
                #
                # print()
                write(f"{stage:<45} NMI={nmi:.6f}   P={p:.6f}   classes={len(classes)}\n")

                if p > EPS and nmi >= 1 - EPS:
                    solved = True
                    write(f"\nSolved at {stage}\n")

                return p

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

            for order in range(LINE_ORDER_START, MAX_LINE_ORDER + 1):

                name = "G" if order == 0 else "L(G)" if order == 1 else f"L^{order}(G)"
                line_t = [] if order == 0 else [("line", order)]

                line_graph, _ = construct_transformed_graph(G, line_t)

                if nx.is_connected(line_graph):
                    diameter = nx.diameter(line_graph)
                else:
                    diameter = max(nx.diameter(line_graph.subgraph(c)) for c in nx.connected_components(line_graph))

                write(f"\n================ {name}, diameter={diameter} ================\n")

                stages = [(name, line_t)]
                stages += [(f"{name}_{d}", [("distance", d)] + line_t) for d in range(2, diameter + 1)]

                for stage_name, transforms in stages:

                    write(f"\n--- {stage_name} ---\n")

                    stage_graph, membership = construct_transformed_graph(G, transforms)

                    for op in OPERATORS:

                        def spectral_partition(resolution, op=op):
                            configure(f"{stage_name}-{op}", transforms, [op], resolution, bfs_depth=1000,
                                      stable_ids=False)
                            new, _ = computer.compute_vertex_signatures(G)
                            return restrict_to_original(new, n)

                        run_with_resolution(f"{stage_name}-{op}", spectral_partition)

                        if solved or aborted:
                            break

                    if solved or aborted:
                        break

                    for probe_name, probe, probe_args in PROBES:
                        # Saved Traces generators are for the original, unsubdivided G only.
                        # if probe_name == "neighborhood_deletion" and (stage_name != "G" or SUBDIVISION_K != 0):
                        #     continue

                        def compute_probe(resolution, probe=probe, probe_args=probe_args, probe_name=probe_name):
                            nonlocal traces_generators
                            configure(f"{stage_name}-probe-{probe_name}", [], ["correlation"], resolution,
                                      bfs_depth=100, stable_ids=True)

                            kwargs = dict(probe_args)
                            if probe_name == "neighborhood_deletion":
                                if traces_generators is None:
                                    traces_generators = load_traces_generators(graph_file, n)
                                    write(f"Loaded {len(traces_generators)} Traces generators for {graph_file.name}\n")
                                kwargs["generators"] = traces_generators

                            if PROBE_ORBIT_REP:
                                unresolved = {
                                    vertex_orbit[v]
                                    for cls in classes.values()
                                    if len({vertex_orbit[v] for v in cls}) > 1
                                    for v in cls
                                }

                                reps = [orbit_reps[i] for i in sorted(unresolved)]
                                probe_roots = list({u for rep in reps for u in membership[rep]})

                                write(f"Probing {len(reps)}/{len(true_orbits)} orbit representatives\n")

                                profiles = probe(stage_graph, computer, probe_roots, **kwargs) if reps else {}

                                rep_signatures = {
                                    i: tuple(sorted((profiles[u] for u in membership[orbit_reps[i]]), key=repr))
                                    for i in unresolved
                                }

                                groups = {}
                                for class_id, cls in classes.items():
                                    for v in cls:
                                        i = vertex_orbit[v]
                                        key = (class_id, rep_signatures[i]) if i in unresolved else (class_id, None)
                                        groups.setdefault(key, []).append(v)

                            else:
                                profiles = probe(stage_graph, computer, list(stage_graph.nodes()), **kwargs)

                                signatures = {
                                    v: tuple(sorted((profiles[u] for u in membership[v]), key=repr))
                                    for v in range(n)
                                }

                                groups = {}
                                for v, sig in signatures.items():
                                    groups.setdefault(sig, []).append(v)

                            return dict(enumerate(groups.values()))

                        run_with_resolution(f"{stage_name}-probe-{probe_name}", compute_probe)

                        if solved or aborted:
                            break

                    if solved or aborted:
                        break

                if solved or aborted:
                    break

            if solved:
                write("\nSTATUS: SOLVED\n")
            elif aborted:
                write("\nSTATUS: ABORTED — P=0 AT MINIMUM RESOLUTION\n")
            else:
                write(f"\nSTATUS: NOT SOLVED up to L^{MAX_LINE_ORDER}(G)\n")

        except MemoryError as e:
            write(
                f"\nMEMORY ERROR on {graph_file.name}: {e}\n{traceback.format_exc()}\nSTATUS: SKIPPED — MEMORY ERROR\n")


def main():
    files = sorted((p for p in GRAPH_DIR.rglob("*") if p.is_file()),
                   key=lambda p: [int(x) if x.isdigit() else x.lower() for x in
                                  re.split(r"(\d+)", str(p.relative_to(GRAPH_DIR)))])
    # files = [
    #     p for p in files
    #     if (m := re.match(r"cfi-(\d+)", p.name))
    #        and int(m.group(1)) >= 116
    # ]

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