input_file = "sts-9"
output_file = "sts-9_0indexed"

edges = []
max_label = -1

with open(input_file, "r") as f:
    for line in f:
        u, v = map(int, line.split())

        u -= 1
        v -= 1

        edges.append((u, v))

        if u > max_label:
            max_label = u
        if v > max_label:
            max_label = v

with open(output_file, "w") as f:
    f.write(f"{max_label+1} {len(edges)}\n")

    for u, v in edges:
        f.write(f"{u} {v}\n")