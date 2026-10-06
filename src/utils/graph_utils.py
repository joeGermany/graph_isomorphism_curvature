from collections import Counter,defaultdict
from pathlib import Path
import networkx as nx
import pynauty
import random
import numpy as np
from src.utils.memory_capacity import guard_remaining_memory

def construct_distance_graph(G,distance):
    nodes=list(G.nodes())
    distance_graph=nx.Graph()
    distance_graph.add_nodes_from(nodes)

    for root in nodes:
        for target,d in nx.single_source_shortest_path_length(G,root,cutoff=distance,).items():
            if d==distance and root!=target:
                distance_graph.add_edge(root,target)

    return distance_graph

def construct_next_line_graph(current_graph,current_membership):
    nodes=list(current_graph.nodes())
    edges=list(current_graph.edges())

    incidence=nx.incidence_matrix(
        current_graph,
        nodelist=nodes,
        edgelist=edges,
        oriented=True,
    ).toarray()

    BTB=np.asarray(incidence.T@incidence,dtype=np.float64)
    adjacency=np.abs(BTB-np.diag(np.diag(BTB)))
    next_graph=nx.from_numpy_array(adjacency)
    next_membership={
        vertex:set()
        for vertex in current_membership
    }
    for new_node,(u,v) in enumerate(edges):
        for original_vertex,items in current_membership.items():
            if u in items or v in items:
                next_membership[original_vertex].add(new_node)

    return next_graph,next_membership


def construct_transformed_graph(G,transformations):
    current_graph=G
    membership={v:{v} for v in G.nodes()}

    for operation,value in reversed(transformations):
        guard_remaining_memory("line graph construction",reserve_percent=15)
        if operation=="distance":
            current_graph=construct_distance_graph(current_graph,value,)
        elif operation=="line":
            for _ in range(value):
                current_graph,membership=construct_next_line_graph(current_graph,membership,)
        else:
            raise ValueError(f"Unknown graph transformation: {operation}")

    return current_graph,membership

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

def equivalence_classes_split_orbits(G_classes,true_orbits):

    vertex_to_class={
        int(vertex):class_id
        for class_id,cell in enumerate(G_classes.values())
        for vertex in cell
    }

    for orbit in true_orbits:
        classes={
            vertex_to_class[int(vertex)]
            for vertex in orbit
        }

        if len(classes)>1:
            return True

    return False

def compute_overall_same_orbit_probability(G,G_classes,original_n):
    """
    Compute the overall probability that random individualization selects
    corresponding vertices from the same true orbit.
    """
    true_orbits=get_true_orbits_restricted(G,original_n)

    if equivalence_classes_split_orbits(G_classes,true_orbits):
        return 0.0

    if all(len(orbit)==1 for orbit in true_orbits) and all(
        len(cell)==1 for cell in G_classes.values()
    ):
        return 1.0

    eligible_classes=[cell for cell in G_classes.values() if len(cell)>1]

    vertex_to_orbit={
        int(vertex):orbit_id
        for orbit_id,orbit in enumerate(true_orbits)
        for vertex in orbit
    }

    probabilities=[]

    for cell in eligible_classes:
        orbit_counts=Counter(vertex_to_orbit[int(vertex)] for vertex in cell)
        probabilities.append(sum(size**2 for size in orbit_counts.values())/len(cell)**2)

    return sum(probabilities)/len(probabilities)

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