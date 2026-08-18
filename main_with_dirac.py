from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score
from tqdm.auto import tqdm

from src.k_wl.kwl import run_kwl
from src.module.heat_kernel_equivalence_compute_with_dirac import HeatKernelEquivalenceComputer
from src.utils.graph_utils import *


def main():
    # graph_file=Path("graphs/35-16-6-8/Graph625")
    graph_file=Path("graphs/TedSpence/36-14-4-6/Graph48")

    subdivision_k=0

    graph_settings = [
        {
            "name": "G",
            "signature_mode": "full_matrix",
            "time_step": 1,
            "end_time_multiplier": 7,
            "fit_order": 6,
            "fit_start": 1,
            "fit_end": 6,
            "resolutions": [1e10],
        },
        {
            "name": "L(G)",
            "signature_mode": "full_matrix",
            "time_step": 1,
            "end_time_multiplier": 50,
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
        use_ids=False,
    )

    graph_original=read_dimacs_or_simple_graph(graph_file)

    # To get G2:
    # adjacency_2 = np.zeros((graph_original.number_of_nodes(), graph_original.number_of_nodes()))
    # for i in range(graph_original.number_of_nodes()):
    #     for target, distance in nx.single_source_shortest_path_length(graph_original, i, 2).items():
    #         if distance == 2 and target != i:
    #             adjacency_2[i][target] = 1
    #             adjacency_2[target][i] = 1

    # graph_original = nx.from_numpy_array(adjacency_2)


    nodes = list(graph_original.nodes())
    edges = list(graph_original.edges())

    edge_incidences_original = defaultdict(list)
    for root in range(len(nodes)):
        for i, (u,v) in enumerate(edges):
            if u == root or v == root:
                edge_incidences_original[root].append(i)

    print(edge_incidences_original)


    incidence = nx.incidence_matrix(graph_original, nodelist=nodes, edgelist=edges, oriented=True).toarray()
    laplacian_edges = np.asarray(incidence.T @ incidence, dtype=np.float64)
    adjacency_edges = np.abs(laplacian_edges - np.diag(np.diag(laplacian_edges)))
    graph = nx.from_numpy_array(adjacency_edges)

    original_n=graph.number_of_nodes()

    true_orbits=get_true_orbits_restricted(graph_original,graph_original.number_of_nodes())
    orbit_labels=partition_to_labels(true_orbits,graph_original.number_of_nodes())

    # all_classes,signatures=equivalence_computer.compute_vertex_signatures(graph)
    # all_classes, signatures = equivalence_computer.compute_vertex_signatures_with_dirac(graph)

    # classes=restrict_to_original(all_classes,original_n)
    # class_labels=partition_to_labels(classes,original_n)

    # print(orbit_labels)
    # print(class_labels)

    # nmi=normalized_mutual_info_score(class_labels,orbit_labels)
    # # probability=compute_overall_same_orbit_probability(graph,classes,original_n)

    # print(f"NMI: {nmi:.4f}")

    def complex_signature(values, decimals=6):
        return tuple(sorted((round(abs(z.real), decimals), round(abs(z.imag), decimals)) for z in values))
        # return tuple(sorted(round(abs(z), decimals) for z in values))

    settings = equivalence_computer.graph_settings[0]
    dirac_spectral_data = equivalence_computer._dirac_spectral_data(graph, settings)
    matrix = equivalence_computer._dirac_full_matrix_values(graph, dirac_spectral_data, settings)

    nodes = dirac_spectral_data["nodes"]
    edges = dirac_spectral_data["edges"]
    eigenvalues = dirac_spectral_data["eigenvalues"]
    eigenvectors = dirac_spectral_data["eigenvectors"]
    times = dirac_spectral_data["times"]

    n = len(nodes)
    e = len(edges)

    vertex_vertex_matrix = matrix[:n, :n]
    vertex_edge_matrix = matrix[:n, n:]
    edge_edge_matrix = matrix[n:, n:]

    raw_sigs = defaultdict(list)
    raw_sigs_edges = defaultdict(list)

    for time_index in tqdm(range(len(times))):
        for root in range(n):
            raw_sigs[root].append(complex_signature(vertex_vertex_matrix[root, :, time_index]))
            raw_sigs[root].append(complex_signature(vertex_edge_matrix[root, :, time_index]))

    for time_index in tqdm(range(len(times))):
        for root in range(e):
            raw_sigs_edges[root].append(complex_signature(vertex_edge_matrix[:, root, time_index]))
            raw_sigs_edges[root].append(complex_signature(edge_edge_matrix[:, root, time_index]))


    all_signatures = set()

    for root in tqdm(range(n)):
        all_signatures.update(raw_sigs[root])

    for root in tqdm(range(e)):
        all_signatures.update(raw_sigs_edges[root])

    signature_to_id = {
        sig: i
        for i, sig in enumerate(sorted(all_signatures))
    }

    sigs = defaultdict(list)
    sigs_edges = defaultdict(list)

    for root in tqdm(range(n)):
        sigs[root] = [signature_to_id[sig] for sig in raw_sigs[root]]

    for root in tqdm(range(e)):
        sigs_edges[root] = [signature_to_id[sig] for sig in raw_sigs_edges[root]]

    edge_incidences = defaultdict(list)
    for root in tqdm(range(n)):
        for i, (u,v) in enumerate(edges):
            if u == root or v == root:
                edge_incidences[root].append(i)

    for root in tqdm(range(n)):
        edge_sigs_to_append = []
        for edge in edge_incidences[root]:
            edge_sigs_to_append.append(tuple(sigs_edges[edge]))

        sigs[root].append(tuple(sorted(edge_sigs_to_append)))
        sigs[root] = tuple(sigs[root])

    sigs_compiled = {}
    for root in tqdm(range(n)):
        by_distance = defaultdict(list)
        distances = nx.single_source_shortest_path_length(graph, root)

        for target, distance in distances.items():
            by_distance[distance].append(sigs[target])

        # sigs_compiled[root] = [sorted(by_distance[d]) for d in sorted(by_distance)]
        sigs_compiled[root] = tuple(tuple(sorted(by_distance[d])) for d in sorted(by_distance))

    
    equivalence_classes = defaultdict(list)

    for root in tqdm(range(n)):
        equivalence_classes[tuple(sigs_compiled[root])].append(root)

    # print(equivalence_classes)
    for a,b in equivalence_classes.items():
        print(b)

    class_labels=partition_to_labels(equivalence_classes, original_n)
    print("Edge classes: ", class_labels)

    final_node_sigs = {}
    for node in tqdm(range(graph_original.number_of_nodes())):
        neighbors_labels = []
        for neighboring_edge in edge_incidences_original[node]:
            neighbors_labels.append(class_labels[neighboring_edge])

        final_node_sigs[node] = tuple(sorted(neighbors_labels))

    # print(final_node_sigs)

    equivalence_classes_re = defaultdict(list)

    for node in tqdm(range(graph_original.number_of_nodes())):
        equivalence_classes_re[tuple(final_node_sigs[node])].append(node)

    class_labels_re = partition_to_labels(equivalence_classes_re, graph_original.number_of_nodes())
    print("classes: ", class_labels_re)
    print("orbits: ", orbit_labels)

    nmi = normalized_mutual_info_score(orbit_labels, class_labels_re)
    print(f"NMI: {nmi}")

    k_wl_parameter=1

    heat_kwl_classes=run_kwl(graph_original,k_wl_parameter,initial_partition=equivalence_classes_re,output_dir="kwl_output/heat_colored",)
    heat_kwl_labels=partition_to_labels(heat_kwl_classes,graph_original.number_of_nodes())
    heat_kwl_nmi=normalized_mutual_info_score(orbit_labels,heat_kwl_labels)
    true_orbit_set={tuple(sorted(orbit)) for orbit in true_orbits}
    heat_matches=sum(tuple(sorted(cls)) in true_orbit_set for cls in heat_kwl_classes)

    print(f"{k_wl_parameter}-WL heat_colored: NMI = {heat_kwl_nmi:.4f}, classes = {len(heat_kwl_classes)}, matching classes = {heat_matches}")



    # item_to_index = {
    #     item: index
    #     for index, item in enumerate(nodes)
    # }

    # distances = [
    #     {
    #         item_to_index[target]: distance
    #         for target, distance in nx.single_source_shortest_path_length(graph, root).items()
    #     }
    #     for root in nodes
    # ]

    # values_nodes = np.zeros((n, len(times), n), dtype=np.complex128)
    # values_edges = np.zeros((n, len(times), e), dtype=np.complex128)

    # for time_index, time_value in tqdm(enumerate(times), total=len(times), desc=f"{settings['name']} Dirac signatures", unit="time", dynamic_ncols=True):
    #     vertex_vertex_kernel = vertex_vertex_matrix[:, :, time_index]
    #     vertex_edge_kernel = vertex_edge_matrix[:, :, time_index]

    #     for root in range(n):
    #         ordered_values = sorted(
    #             (
    #                 distances[root].get(target, n + 1),
    #                 vertex_vertex_kernel[root, target],
    #             )
    #             for target in range(n)
    #         )

    #         values_nodes[root, time_index] = [
    #             entry[-1]
    #             for entry in ordered_values
    #         ]

    #     # for root in range(e):
    #     #     ordered_values_edges = sorted(

    #     #     )

    # values_nodes = values_nodes.reshape(n, -1)



    # for level, settings in enumerate(equivalence_computer.graph_settings):
    #     if level != 0:
    #         return
        
    #     dirac_spectral_data = equivalence_computer._dirac_spectral_data(graph, settings)
    #     # dirac = equivalence_computer._dirac_full_matrix_values(graph, dirac_spectral_data, settings)
    #     # print(dirac.shape)

    #     nodes = dirac_spectral_data["nodes"]
    #     edges = dirac_spectral_data["edges"]
    #     eigenvalues = dirac_spectral_data["eigenvalues"]
    #     eigenvectors = dirac_spectral_data["eigenvectors"]
    #     times = dirac_spectral_data["times"]

    #     n = len(nodes)
    #     e = len(edges)


    #     vertices = list(graph.nodes())
    #     vertex_signatures = [{v: [] for v in nodes}]
    #     memberships = [{v: {v} for v in vs} for vs in vertices]
    #     for v in nodes:
    #         vertex_signatures[v].append(
    #             tuple(sorted(values[item] for item in memberships[v]))
    #         )

    #     item_to_index = {
    #         item: index
    #         for index, item in enumerate(nodes)
    #     }

    #     distances = [
    #         {
    #             item_to_index[target]: distance
    #             for target, distance in nx.single_source_shortest_path_length(graph, root).items()
    #         }
    #         for root in nodes
    #     ]

        # distances_edges = [
        #     {
        #         node: min(
        #             nx.shortest_path_length(graph, u, node),
        #             nx.shortest_path_length(graph, v, node)
        #         )
        #         for node in graph.nodes
        #     }
        #     for u, v in edges
        # ]

        # values_nodes = np.zeros((n, len(times), n), dtype=np.complex128)
        # values_edges = np.zeros((n, len(times), e), dtype=np.complex128)

        # for time_index, time_value in tqdm(enumerate(times), total=len(times), desc=f"{settings['name']} Dirac signatures", unit="time", dynamic_ncols=True):
        #     # e^{itD}
        #     dirac_kernel = (eigenvectors * np.exp(1j * eigenvalues * time_value)) @ eigenvectors.T

        #     vertex_vertex_kernel = dirac_kernel[:n, :n]
        #     vertex_edge_kernel = dirac_kernel[:n, n:]

        #     for root in range(n):
        #         ordered_values = sorted(
        #             (
        #                 distances[root].get(target, n + 1),
        #                 vertex_vertex_kernel[root, target],
        #             )
        #             for target in range(n)
        #         )

        #         values_nodes[root, time_index] = [
        #             entry[-1]
        #             for entry in ordered_values
        #         ]

        #     # for root in range(e):
        #     #     ordered_values_edges = sorted(

        #     #     )

        # values_nodes = values_nodes.reshape(n, -1)

   


        # item_to_index = {
        #     item: index
        #     for index, item in enumerate(nodes)
        # }

        # distances = [
        #     {
        #         item_to_index[target]: distance
        #         for target, distance in nx.single_source_shortest_path_length(
        #             graph, root
        #         ).items()
        #     }
        #     for root in nodes
        # ]

        # distance_matrix = dict(
        #     nx.all_pairs_shortest_path_length(graph)
        # )

        # distances_edges = [
        #     {
        #         edge_index: min(
        #             distance_matrix[root][u],
        #             distance_matrix[root][v]
        #         )
        #         for edge_index, (u, v) in enumerate(edges)
        #     }
        #     for root in nodes
        # ]

        # values_nodes = np.zeros(
        #     (n, len(times), n),
        #     dtype=np.complex128
        # )

        # values_edges = np.zeros(
        #     (n, len(times), e),
        #     dtype=np.complex128
        # )

        # for time_index, time_value in tqdm(
        #     enumerate(times),
        #     total=len(times),
        #     desc=f"{settings['name']} Dirac signatures",
        #     unit="time",
        #     dynamic_ncols=True
        # ):

        #     # e^{itD}
        #     dirac_kernel = (
        #         eigenvectors
        #         * np.exp(1j * eigenvalues * time_value)
        #     ) @ eigenvectors.T

        #     vertex_vertex_kernel = dirac_kernel[:n, :n]
        #     vertex_edge_kernel = dirac_kernel[:n, n:]

        #     for root in range(n):

        #         ordered_values = sorted(
        #             (
        #                 distances[root].get(target, n + 1),
        #                 vertex_vertex_kernel[root, target]
        #             )
        #             for target in range(n)
        #         )

        #         values_nodes[root, time_index] = [
        #             value
        #             for distance, value in ordered_values
        #         ]

        #         ordered_values_edges = sorted(
        #             (
        #                 distances_edges[root][edge_index],
        #                 vertex_edge_kernel[root, edge_index]
        #             )
        #             for edge_index in range(e)
        #         )

        #         values_edges[root, time_index] = [
        #             value
        #             for distance, value in ordered_values_edges
        #         ]

        # values_nodes = values_nodes.reshape(n, -1)
        # values_edges = values_edges.reshape(n, -1)

        # print(values_nodes.shape, values_edges.shape)

        # sig_nodes = {
        #     item: tuple(values_nodes[index].ravel())
        #     for index, item in enumerate(nodes)
        # }

        # sig_edges = {
        #     item: tuple(values_edges[index].ravel())
        #     for index, item in enumerate(nodes)
        # }

        # signatures = {
        #     item: sig_nodes[item] + sig_edges[item]
        #     for item in sig_edges
        # }

        # signatures = equivalence_computer._dirac_truncate_signatures(signatures, settings)

        # memberships = {v: {v} for v in nodes}
        # vertex_signatures = {v: [] for v in nodes}

        # for v in nodes:
        #     vertex_signatures[v].append(
        #         tuple(sorted(signatures[item] for item in memberships[v]))
        #     )

        # print(vertex_signatures)

        # sigs = {v: tuple(sig) for v, sig in vertex_signatures.items()}

        # sigs = equivalence_computer._signatures_with_bfs(graph, sigs)

        # classes = equivalence_computer._group_by_equality(sigs)

        # class_labels=partition_to_labels(classes,original_n)

        # print(orbit_labels)
        # print(class_labels)

        # nmi=normalized_mutual_info_score(class_labels,orbit_labels)
        # print(f"NMI: {nmi:.4f}")


    return

    classes=restrict_to_original(all_classes,original_n)
    class_labels=partition_to_labels(classes,original_n)

    print(orbit_labels)
    print(class_labels)

    nmi=normalized_mutual_info_score(class_labels,orbit_labels)
    # probability=compute_overall_same_orbit_probability(graph,classes,original_n)

    print(f"NMI: {nmi:.4f}")
    # print(f"Same-orbit probability: {probability:.4f}")

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

    unit_kwl_classes=run_kwl(original_graph,k_wl_parameter,output_dir="kwl_output/unit_coloring",)

    heat_kwl_classes=run_kwl(original_graph,k_wl_parameter,initial_partition=classes,output_dir="kwl_output/heat_colored",)

    unit_kwl_labels=partition_to_labels(unit_kwl_classes,original_n)
    heat_kwl_labels=partition_to_labels(heat_kwl_classes,original_n)

    unit_kwl_nmi=normalized_mutual_info_score(orbit_labels,unit_kwl_labels)
    heat_kwl_nmi=normalized_mutual_info_score(orbit_labels,heat_kwl_labels)

    true_orbit_set={tuple(sorted(orbit)) for orbit in true_orbits}

    unit_matches=sum(tuple(sorted(cls)) in true_orbit_set for cls in unit_kwl_classes)

    heat_matches=sum(tuple(sorted(cls)) in true_orbit_set for cls in heat_kwl_classes)

    print(
        f"{k_wl_parameter}-WL unit_coloring: "
        f"NMI = {unit_kwl_nmi:.4f}, "
        f"classes = {len(unit_kwl_classes)}, "
        f"matching classes = {unit_matches}"
    )

    print(
        f"{k_wl_parameter}-WL heat_colored: "
        f"NMI = {heat_kwl_nmi:.4f}, "
        f"classes = {len(heat_kwl_classes)}, "
        f"matching classes = {heat_matches}"
    )


if __name__=="__main__":
    main()