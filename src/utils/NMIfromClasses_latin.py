import re
from sklearn.metrics import normalized_mutual_info_score
import networkx as nx
import math

from src.k_wl.kwl import run_kwl
from src.utils.graph_utils import *

graph_type = "order8"

def latin_graph(encoded):
    encoded = encoded.strip()

    # n^2 entries -> Latin square of order n
    n = math.isqrt(len(encoded))

    if n * n != len(encoded):
        raise ValueError(
            f"Invalid encoding: length {len(encoded)} is not a perfect square."
        )

    # Decode Latin square
    L = [
        list(encoded[i*n:(i+1)*n])
        for i in range(n)
    ]

    G = nx.Graph()

    # Each cell is a vertex
    vertices = [(i, j) for i in range(n) for j in range(n)]
    G.add_nodes_from(vertices)

    # Same row
    for i in range(n):
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                G.add_edge((i, j1), (i, j2))

    # Same column
    for j in range(n):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                G.add_edge((i1, j), (i2, j))

    # Same symbol
    for symbol in set(encoded):
        cells = [
            (i, j)
            for i in range(n)
            for j in range(n)
            if L[i][j] == symbol
        ]

        for a in range(len(cells)):
            for b in range(a + 1, len(cells)):
                G.add_edge(cells[a], cells[b])

    return G

with open(f"{graph_type}_output_round1_withoutL(L(G))_batch0-NMIs.txt", "w") as file:
    with open(f"{graph_type}_output_round1_withoutL(L(G))_batch0.txt", "r") as f, open(f"graphs/{graph_type}_isotopy_nontrivial.txt", "r") as f_graph:
        for (line, line_graph) in zip(f, f_graph):
            match = re.match(r"Graph(\d+): Classes: \[(.*?)\]", line)

            if match:
                graph_num = int(match.group(1))
                class_labels = [int(x.strip()) for x in match.group(2).split(",")]

                line_graph = line_graph.strip()
                if not line_graph:
                    continue

                graph = latin_graph(line_graph)
                graph = nx.convert_node_labels_to_integers(graph)
                original_n = graph.number_of_nodes()

                print((graph_num, class_labels))

                true_orbits=get_true_orbits_restricted(graph,original_n)
                orbit_labels=partition_to_labels(true_orbits,original_n)

                nmi=normalized_mutual_info_score(class_labels,orbit_labels)
                print(f"NMI: {nmi:.4f}")

                k_wl_parameter=1

                classes = defaultdict(list)
                for i in range(len(class_labels)):
                    classes[tuple([class_labels[i]])].append(i)

                # unit_kwl_classes=run_kwl(original_graph,k_wl_parameter,output_dir="kwl_output/unit_coloring",)
                heat_kwl_classes=run_kwl(graph,k_wl_parameter,initial_partition=classes,output_dir="kwl_output/heat_colored",)

                # unit_kwl_labels=partition_to_labels(unit_kwl_classes,original_n)
                heat_kwl_labels=partition_to_labels(heat_kwl_classes,original_n)

                # unit_kwl_nmi=normalized_mutual_info_score(orbit_labels,unit_kwl_labels)
                heat_kwl_nmi=normalized_mutual_info_score(orbit_labels,heat_kwl_labels)

                true_orbit_set={tuple(sorted(orbit)) for orbit in true_orbits}

                # unit_matches=sum(tuple(sorted(cls)) in true_orbit_set for cls in unit_kwl_classes)
                heat_matches=sum(tuple(sorted(cls)) in true_orbit_set for cls in heat_kwl_classes)

                # print(f"{k_wl_parameter}-WL unit_coloring: NMI = {unit_kwl_nmi:.4f}, classes = {len(unit_kwl_classes)}, matching classes = {unit_matches}")
                print(f"{k_wl_parameter}-WL heat_colored: NMI = {heat_kwl_nmi:.4f}, classes = {len(heat_kwl_classes)}, matching classes = {heat_matches}")

                file.write(f"Graph{graph_num}: NMI: {nmi:.4f}, NMI with 1-WL: {heat_kwl_nmi:.4f}, #Orbits of G: {np.max(orbit_labels)+1}, Orbits: {orbit_labels}\n")
