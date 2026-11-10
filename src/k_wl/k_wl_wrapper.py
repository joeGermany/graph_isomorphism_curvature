"""NetworkX graph -> compiled k-WL -> one stable color per vertex."""

from collections.abc import Mapping

from . import _k_wl as _k_wl_cpp


def k_wl(graph, k, initial_colors=None, *, max_tuples=2_000_000):
    """Return {vertex: stable_color} from ordinary 1-WL or standard k-WL.

    Args:
        graph: NetworkX undirected simple graph.
        k: WL dimension (>= 1).
        initial_colors: Optional complete vertex-coloring, either
            - mapping {vertex: color_label}, or
            - sequence of labels aligned with list(graph.nodes()).
            Labels must be hashable; equal labels denote the same color.
        max_tuples: Maximum number of ordered k-tuples to process.

    Only diagonal tuple colors are returned; the full k-WL refinement still
    runs internally in C++.
    """
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Expected an undirected simple NetworkX graph")

    nodes = list(graph.nodes())
    n = len(nodes)
    if not n:
        return {}

    index = {node: i for i, node in enumerate(nodes)}
    adjacency = [[0] * n for _ in range(n)]
    for u, v in graph.edges():
        i, j = index[u], index[v]
        adjacency[i][j] = 1
        adjacency[j][i] = 1

    if initial_colors is None:
        color_ids = []  # Uncolored graph: C++ uses the same initial label for all vertices.
    else:
        if isinstance(initial_colors, Mapping):
            if set(initial_colors) != set(nodes):
                raise ValueError("initial_colors must specify exactly every graph vertex")
            labels = [initial_colors[v] for v in nodes]
        else:
            if isinstance(initial_colors, (str, bytes)):
                raise TypeError("initial_colors must be a mapping or sequence of color labels")
            labels = list(initial_colors)
            if len(labels) != n:
                raise ValueError(f"Expected {n} initial colors, received {len(labels)}")

        # Normalize color labels to compact ints. Their *equality* matters,
        # not their original values (e.g., strings or large signatures).
        label_ids = {}
        color_ids = []
        for label in labels:
            if label not in label_ids:
                label_ids[label] = len(label_ids)
            color_ids.append(label_ids[label])

    vertex_colors = _k_wl_cpp.run(
        adjacency, k, vertex_colors=color_ids, max_tuples=max_tuples
    )
    return dict(zip(nodes, vertex_colors))
