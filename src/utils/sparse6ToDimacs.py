import os
import networkx as nx
from collections import defaultdict

INPUT_DIR = "graphs/graphs-s6-20260814141330"
OUTPUT_DIR = "graphs/encyclopedia"

os.makedirs(OUTPUT_DIR, exist_ok=True)

import networkx as nx

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

params_dicts = {}

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".s6"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)

    G = nx.read_sparse6(input_path)
    params = srg_parameters(G)

    if params not in params_dicts:
        params_dicts[params] = 1
    else:
        params_dicts[params] += 1

    output_path = OUTPUT_DIR + f"/{params[0]}-{params[1]}-{params[2]}-{params[3]}/Graph{params_dicts[params]}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"c {output_path}\n")

        f.write(
            f"p edge {G.number_of_nodes()} "
            f"{G.number_of_edges()}\n"
        )

        for u, v in G.edges():
            f.write(f"e {u + 1} {v + 1}\n")