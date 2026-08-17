from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score
import time

from src.k_wl.kwl import run_kwl
from src.module.heat_kernel_equivalence_compute_with_dirac import HeatKernelEquivalenceComputer
from src.utils.graph_utils import *

def srg_parameters(G):
    v = G.number_of_nodes()

    # Check regularity
    degrees = [d for _, d in G.degree()]
    if len(set(degrees)) != 1:
        return None

    k = degrees[0]

    # Find lambda and mu
    lambda_values = set()
    mu_values = set()

    nodes = list(G.nodes())

    for i, u in enumerate(nodes):
        for w in nodes[i+1:]:
            common = len(list(nx.common_neighbors(G, u, w)))

            if G.has_edge(u, w):
                lambda_values.add(common)
            else:
                mu_values.add(common)

    if len(lambda_values) != 1 or len(mu_values) != 1:
        return None

    lam = lambda_values.pop()
    mu = mu_values.pop()

    return (v, k, lam, mu)

def read_sts_file(filename, sts_size):
    systems = []

    num_triples = sts_size * (sts_size - 1) // 6
    expected_length = 3 * num_triples

    with open(filename) as f:
        for line in f:
            s = line.strip()

            assert len(s) == expected_length, (
                f"Expected {expected_length} characters for STS-{sts_size}, "
                f"got {len(s)}"
            )

            triples = [
                tuple(ord(c) - ord('a') for c in s[i:i+3])
                for i in range(0, len(s), 3)
            ]

            systems.append(triples)

    return systems

def sts_to_graph(triples, n):
    # G = nx.Graph()
    # G.add_nodes_from(range(n))

    # for a, b, c in triples:
    #     G.add_edge(a, b)
    #     G.add_edge(a, c)
    #     G.add_edge(b, c)

    # return G

    BIG = nx.Graph()

    for i, triplet in enumerate(triples):
        BIG.add_node(i, triplet=triplet)

    num_blocks = len(triples)
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            # Convert to sets to easily find common elements
            set_i = set(triples[i])
            set_j = set(triples[j])
            
            # If they share exactly 1 element, connect them
            if len(set_i.intersection(set_j)) == 1:
                BIG.add_edge(i, j)

    return BIG

def main():
    # graph_type = "order7"
    # with open(f"latin_graphs_isotopy_nontrivial_{graph_type}_output_round1_withoutL(L(G)).txt", "w") as file:
    sts_size = 15

    num_triples = sts_size * (sts_size - 1) // 6
    expected_length = 3 * num_triples

    with open(f"sts-{sts_size}_output_round1_withoutL(L(G)).txt", "w") as file:
        with open(f"graphs/sts/sts-{sts_size}") as f: #_1M_sample_decompressed
            for instance, line in enumerate(f, start=1):
                if instance == 1001:
                    return
                
                print(f"Graph{instance}")

                s = line.strip()

                assert len(s) == expected_length, (
                    f"Expected {expected_length} characters for STS-{sts_size}, "
                    f"got {len(s)}"
                )

                triples = [
                    tuple(ord(c) - ord('a') for c in s[i:i+3])
                    for i in range(0, len(s), 3)
                ]

                original_graph = sts_to_graph(triples, sts_size)

                print(srg_parameters(original_graph))

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
                        "time_step": 0.1,
                        "end_time_multiplier": 20,
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

                original_n=original_graph.number_of_nodes()

                graph=subdivide_every_edge(original_graph,subdivision_k)

                all_classes,signatures=equivalence_computer.compute_vertex_signatures(graph)

                classes=restrict_to_original(all_classes,original_n)
                true_orbits=get_true_orbits_restricted(graph,original_n)

                class_labels=partition_to_labels(classes,original_n)
                orbit_labels=partition_to_labels(true_orbits,original_n)

                print("Orbits: ", orbit_labels)
                print("Classes: ", class_labels)

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

                # unit_kwl_classes=run_kwl(original_graph,k_wl_parameter,output_dir="kwl_output/unit_coloring",)
                heat_kwl_classes=run_kwl(original_graph,k_wl_parameter,initial_partition=classes,output_dir="kwl_output/heat_colored",)

                # unit_kwl_labels=partition_to_labels(unit_kwl_classes,original_n)
                heat_kwl_labels=partition_to_labels(heat_kwl_classes,original_n)

                # unit_kwl_nmi=normalized_mutual_info_score(orbit_labels,unit_kwl_labels)
                heat_kwl_nmi=normalized_mutual_info_score(orbit_labels,heat_kwl_labels)

                true_orbit_set={tuple(sorted(orbit)) for orbit in true_orbits}

                # unit_matches=sum(tuple(sorted(cls)) in true_orbit_set for cls in unit_kwl_classes)
                heat_matches=sum(tuple(sorted(cls)) in true_orbit_set for cls in heat_kwl_classes)

                # print(f"{k_wl_parameter}-WL unit_coloring: NMI = {unit_kwl_nmi:.4f}, classes = {len(unit_kwl_classes)}, matching classes = {unit_matches}")
                print(f"{k_wl_parameter}-WL heat_colored: NMI = {heat_kwl_nmi:.4f}, classes = {len(heat_kwl_classes)}, matching classes = {heat_matches}")

                file.write(f"Graph{instance}: NMI: {nmi:.4f}, NMI with 1-WL: {heat_kwl_nmi:.4f}, #Orbits of G: {np.max(orbit_labels)+1}, Orbits: {orbit_labels}\n")

if __name__=="__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print("Total running time: ", end_time - start_time)