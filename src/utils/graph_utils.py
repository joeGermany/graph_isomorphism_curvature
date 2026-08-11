from collections import Counter,defaultdict
from pathlib import Path
import networkx as nx
import pynauty
import random

def randomly_permute_graph(G,seed=None):
    rng=random.Random(seed)

    vertices=list(G.nodes())
    permuted_vertices=vertices.copy()
    rng.shuffle(permuted_vertices)
    # print(permuted_vertices)
    mapping=dict(zip(vertices,permuted_vertices))
    H=nx.relabel_nodes(G,mapping,copy=True)

    return H,mapping

def get_orbits_with_pynauty(G):
    n = G.number_of_nodes()

    adjacency_dict = {
        v:sorted(int(u) for u in G.neighbors(v))
        for v in range(n)
    }

    g = pynauty.Graph(number_of_vertices=n,directed=False,adjacency_dict=adjacency_dict)
    generators,group_size,_,orbit_labels,num_orbits = pynauty.autgrp(g)

    groups = defaultdict(list)

    for v,orbit_label in enumerate(orbit_labels):
        groups[int(orbit_label)].append(v)

    orbits = [sorted(cls) for cls in groups.values()]
    orbits.sort(key=lambda cls:cls[0] if cls else -1)

    return orbits

def get_true_orbits_restricted(G,original_n):

    restricted_orbits = []

    for orbit in get_orbits_with_pynauty(G):
        original_part = sorted(int(vertex) for vertex in orbit if int(vertex) < original_n)

        if original_part:
            restricted_orbits.append(original_part)

    restricted_orbits.sort(key=lambda orbit:(orbit[0],len(orbit),orbit))

    return restricted_orbits

def compute_overall_same_orbit_probability(G,G_classes,original_n):
    """
    Compute the overall probability that random individualization selects
    corresponding vertices from the same true orbit.
    """
    eligible_signatures = sorted(
        [signature for signature,cell in G_classes.items() if len(cell) > 1],
        key=repr,
    )

    if not eligible_signatures:
        return 0.0

    true_orbits = get_true_orbits_restricted(G,original_n)

    vertex_to_orbit = {
        int(vertex):orbit_id
        for orbit_id,orbit in enumerate(true_orbits)
        for vertex in orbit
    }

    class_probabilities = {}

    for signature in eligible_signatures:
        cell = sorted(int(vertex) for vertex in G_classes[signature])
        cell_size = len(cell)

        orbit_counts = Counter(vertex_to_orbit[vertex] for vertex in cell)
        orbit_sizes = sorted(orbit_counts.values(),reverse=True)

        probability = sum(size ** 2 for size in orbit_sizes) / cell_size ** 2

        class_probabilities[signature] = {
            "cell":cell,
            "cell_size":cell_size,
            "orbit_sizes":orbit_sizes,
            "probability":probability,
        }

    overall_probability = sum(
        information["probability"]
        for information in class_probabilities.values()
    ) / len(eligible_signatures)

    return overall_probability

def subdivide_every_edge(G,k):
    n = G.number_of_nodes()
    out = nx.Graph()
    out.add_nodes_from(range(n))

    edges = sorted({
        (min(int(u),int(v)),max(int(u),int(v)))
        for u,v in G.edges()
        if int(u) != int(v)
    })

    if k == 0:
        out.add_edges_from(edges)
        return out

    next_new = n

    for u,v in edges:
        previous_vertex = u

        for _ in range(k):
            new_vertex = next_new
            next_new += 1

            out.add_node(new_vertex)
            out.add_edge(previous_vertex,new_vertex)
            previous_vertex = new_vertex

        out.add_edge(previous_vertex,v)

    return out

def restrict_to_original(equivalence_classes,original_n):
    restricted = defaultdict(list)

    for signature,vertex_class in equivalence_classes.items():
        original_class = sorted(vertex for vertex in vertex_class if vertex < original_n)

        if original_class:
            restricted[signature] = original_class

    return restricted

def partition_to_labels(partition,original_n):
    labels = [-1] * original_n
    classes = partition.values() if isinstance(partition,dict) else partition

    for class_id,vertex_class in enumerate(classes):
        for vertex in vertex_class:
            labels[int(vertex)] = class_id

    return labels

def read_dimacs_or_simple_graph(filename):
    lines = [
        line.strip()
        for line in Path(filename).read_text(encoding="utf-8").splitlines()
        if line.strip()
        and not line.strip().startswith(("c","#"))
    ]

    first_line = lines[0].split()
    is_dimacs = first_line[0] == "p"

    if is_dimacs:
        number_of_vertices = int(first_line[-2])
        edge_lines = lines[1:]
        edges = [
            (int(parts[1]) - 1,int(parts[2]) - 1)
            for line in edge_lines
            if (parts := line.split())[0] == "e"
        ]
    else:
        number_of_vertices = int(first_line[0])
        edge_lines = lines[1:]
        edges = [
            (int(parts[0]),int(parts[1]))
            for line in edge_lines
            if len(parts := line.split()) >= 2
        ]

    graph = nx.Graph()
    graph.add_nodes_from(range(number_of_vertices))
    graph.add_edges_from((u,v) for u,v in edges if u != v)

    return graph