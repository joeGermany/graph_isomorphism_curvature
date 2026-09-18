import networkx as nx
from sklearn.metrics import normalized_mutual_info_score

from src.module.heat_kernel_equivalence_compute import HeatKernelEquivalenceComputer
from src.utils.graph_utils import *


CYCLE_N = 200
WHEEL_N = 100
CLIQUE_SIZE = 3
PRECISION_DPS = 20
RESOLUTION = 1e15


computer = HeatKernelEquivalenceComputer(
    graph_settings=[{
        "name": "G",
        "transformations": [],
        "operators": ["laplacian"],
        "signature_mode": "full_matrix",
        "time_step": 1,
        "end_time_multiplier": 7,
        "fit_order": 6,
        "fit_start": 1,
        "fit_end": 6,
        "resolutions": [RESOLUTION],
        "dirac_decimals": 6,
        "precision_dps": PRECISION_DPS,
    },
        {
            "name": "L(G)",
            "transformations": [("line", 1)],
            "operators": ["laplacian"],
            "bfs_sort": True,
            "signature_mode": "full_matrix",
            "time_step": 1,
            "end_time_multiplier": 7,
            "fit_order": 6,
            "fit_start": 1,
            "fit_end": 4,
            "resolutions": [RESOLUTION],
            "dirac_decimals": 6,
            "precision_dps": PRECISION_DPS
        },
    ],
    distance_graph_order=1,
    comparison_mode="fixed_resolution",
    bfs_depth=None,
    weird_norm_threshold=1e-5,
    zero_tolerance=1e-12,
    use_ids=True,
)


def run(graph, name):
    n = graph.number_of_nodes()
    classes, _ = computer.compute_vertex_signatures(graph)
    true_orbits = get_true_orbits_restricted(graph, n)

    nmi = normalized_mutual_info_score(
        partition_to_labels(classes, n),
        partition_to_labels(true_orbits, n)
    )
    same_orbit_probability = compute_overall_same_orbit_probability(
        graph,
        classes,
        n
    )

    print(f"\n{name}")
    print(f"NMI: {nmi}")
    print(f"Same-orbit probability: {same_orbit_probability}")
    print(f"Classes: {len(classes)}")
    print(f"True orbits: {len(true_orbits)}")



# # Cycle + clique
# graph = nx.cycle_graph(CYCLE_N)
# clique = list(range(CYCLE_N, CYCLE_N + CLIQUE_SIZE))
# graph.add_edges_from((u, v) for i, u in enumerate(clique) for v in clique[i + 1:])
# graph.add_edges_from((0, v) for v in clique)
# run(graph, "Cycle + clique")


# Wheel + clique
graph = nx.wheel_graph(WHEEL_N)
clique = list(range(WHEEL_N, WHEEL_N + CLIQUE_SIZE))
graph.add_edges_from((u, v) for i, u in enumerate(clique) for v in clique[i + 1:])
graph.add_edges_from((1, v) for v in clique)  # vertex 1 is on outer cycle
run(graph, "Wheel + clique")