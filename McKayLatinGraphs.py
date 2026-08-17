import networkx as nx
import math
import os

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

graphs = []

with open("order7_isotopy_nontrivial.txt", "r") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        G = latin_graph(line)
        G = nx.convert_node_labels_to_integers(G)
        graphs.append(G)

for i, G in enumerate(graphs):
    print(
        f"Graph {i}: "
        f"|V|={G.number_of_nodes()}, "
        f"|E|={G.number_of_edges()}"
    )

    print(srg_parameters(G))

    # # Export to dimacs
    # output_path = f"graphs/latin_graphs_isotopy_nontrivial/order8/Graph{i+1}"
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # with open(output_path, "w") as f:
    #     f.write(f"c {output_path}\n")

    #     f.write(
    #         f"p edge {G.number_of_nodes()} "
    #         f"{G.number_of_edges()}\n"
    #     )

    #     for u, v in G.edges():
    #         f.write(f"e {u + 1} {v + 1}\n")
