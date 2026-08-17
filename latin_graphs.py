import networkx as nx
import numpy as np
import os

def make_cyclic_latin_square(n):
    """Generates a standard n x n valid Latin Square grid using a cyclic shift."""
    return np.array([(np.arange(n) + i) % n for i in range(n)])

def build_latin_square_graph(n):
    M = make_cyclic_latin_square(n)
    G = nx.Graph()

    for r in range(n):
        for c in range(n):
            G.add_node((r, c), symbol=M[r, c])

    nodes = list(G.nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            r1, c1 = nodes[i]
            r2, c2 = nodes[j]

            if r1 == r2 or c1 == c2 or M[r1, c1] == M[r2, c2]:
                G.add_edge(nodes[i], nodes[j])

    return G

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

latin_square_graphs = {}

for size_n in range(2, 16):
    graph = build_latin_square_graph(size_n)
    graph = nx.convert_node_labels_to_integers(graph)

    v_count = graph.number_of_nodes()
    latin_square_graphs[v_count] = graph

    params = srg_parameters(graph)
    print(params)

    # Export to dimacs
    output_path = "graphs/latin_graphs" + f"/Graph{size_n}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"c {output_path}\n")

        f.write(
            f"p edge {graph.number_of_nodes()} "
            f"{graph.number_of_edges()}\n"
        )

        for u, v in graph.edges():
            f.write(f"e {u + 1} {v + 1}\n")
