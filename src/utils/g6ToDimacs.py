import os
import networkx as nx

filename = "graphs/35-16-6-8.g6"

with open(filename, "rb") as f:
    for i, line in enumerate(f):
        line = line.strip()

        if not line:
            continue

        G = nx.from_graph6_bytes(line)

        output_path = f"graphs/35-16-6-8/Graph{i+1}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write(f"c {output_path}\n")

            f.write(
                f"p edge {G.number_of_nodes()} "
                f"{G.number_of_edges()}\n"
            )

            for u, v in G.edges():
                f.write(f"e {u + 1} {v + 1}\n")