import re

threshold = 1e-8
graphs = []

with open("results/36-15-6-6_output.txt", "r") as f:
    for line in f:
        match = re.search(
            r"Graph(\d+):.*NMI with 1-WL:\s*([0-9.eE+-]+)",
            line
        )

        if match:
            graph_num = int(match.group(1))
            nmi = float(match.group(2))

            if abs(nmi - 1.0) > threshold:
                graphs.append(graph_num)

print(graphs)