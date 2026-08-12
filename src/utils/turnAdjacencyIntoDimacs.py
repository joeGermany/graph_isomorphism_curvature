import numpy as np
import networkx as nx

number = "35-18-9-9"
common = f"graphs/TedSpence/{number}"

with open(f"{common}/{number}_adjacencies", "r") as f:
    content = f.read()

# Blank lines separate graphs
blocks = content.strip().split("\n\n")

for i, block in enumerate(blocks, start=1):
    rows = [
        ''.join(line.split())
        for line in block.splitlines()
        if line.strip()
    ]

    A = np.array(
        [[int(x) for x in row] for row in rows],
        dtype=int
    )

    # Checks
    assert A.shape[0] == A.shape[1], \
        f"Graph {i}: matrix is not square"

    assert np.array_equal(A, A.T), \
        f"Graph {i}: adjacency matrix is not symmetric"

    assert np.all((A == 0) | (A == 1)), \
        f"Graph {i}: matrix contains entries other than 0 and 1"

    G = nx.from_numpy_array(A)

    filename = f"{common}/Graph{i}"

    with open(filename, "w") as f:
        f.write(f"c {common}_Graph{i}\n")
        f.write(
            f"p edge {G.number_of_nodes()} "
            f"{G.number_of_edges()}\n"
        )

        # NetworkX uses vertices 0,...,n-1
        # DIMACS uses vertices 1,...,n
        for u, v in G.edges():
            f.write(f"e {u + 1} {v + 1}\n")