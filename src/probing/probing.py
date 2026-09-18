from itertools import combinations, permutations, product
import networkx as nx
from tqdm.auto import tqdm

def _signature(computer, graph, vertex):
    old_use_ids = computer.use_ids
    old_use_stable_ids = computer.use_stable_ids
    old_show_progress = computer.show_progress

    computer.use_ids = True
    computer.use_stable_ids = True
    computer.show_progress = False

    try:
        _, signatures = computer.compute_vertex_signatures(graph)
        return signatures[vertex]
    finally:
        computer.use_ids = old_use_ids
        computer.use_stable_ids = old_use_stable_ids
        computer.show_progress = old_show_progress

def _profile(signatures):
    """A probing profile is just the sorted multiset of obtained signatures."""
    return tuple(sorted(signatures, key=repr))



def edge_addition(graph, computer, cls, M=1):
    profiles = {u: [] for u in cls}
    nodes = list(graph.nodes())

    non_edges = [
        (a, b) for a, b in combinations(nodes, 2)
        if not graph.has_edge(a, b)
    ]

    for u in tqdm(cls, desc="Edge addition probing", unit="vertex", dynamic_ncols=True):
        u_non_edges = [e for e in non_edges if u in e]

        for first_edge in u_non_edges:
            remaining = [e for e in non_edges if e != first_edge]

            for extra_edges in combinations(remaining, M - 1):
                probed = graph.copy()
                probed.add_edges_from((first_edge,) + extra_edges)

                profiles[u].append(_signature(computer, probed, u))

    return {u: _profile(sigs) for u, sigs in profiles.items()}



def edge_deletion(graph, computer, cls, M=1):
    profiles = {u: [] for u in cls}
    edges = list(graph.edges())
    components = nx.number_connected_components(graph)

    for u in tqdm(cls, desc="Edge deletion probing", unit="vertex", dynamic_ncols=True):
        u_edges = [e for e in edges if u in e]

        for first_edge in u_edges:
            remaining = [e for e in edges if e != first_edge]

            for extra_edges in combinations(remaining, M - 1):
                deleted = (first_edge,) + extra_edges
                probed = graph.copy()
                probed.remove_edges_from(deleted)

                if nx.number_connected_components(probed) > components:
                    continue

                profiles[u].append(_signature(computer, probed, u))

    return {u: _profile(sigs) for u, sigs in profiles.items()}


def edge_swap(graph, computer, cls, M=1):
    profiles = {u: [] for u in cls}
    edges = list(graph.edges())
    n_edges = len(edges)

    for u in tqdm(cls, desc="Edge swap probing", unit="vertex", dynamic_ncols=True):
        def choose(start_i, start_j, start_type, depth, removed, added, touches_u):
            if depth == M:
                if not touches_u:
                    return

                probed = graph.copy()
                probed.remove_edges_from(map(tuple, removed))
                probed.add_edges_from(map(tuple, added))

                if nx.is_connected(probed):
                    profiles[u].append(_signature(computer, probed, u))
                return

            for i in range(start_i, n_edges - 1):
                j_begin = start_j if i == start_i else i + 1

                for j in range(j_begin, n_edges):
                    a, b = edges[i]
                    c, d = edges[j]

                    if len({a, b, c, d}) < 4:
                        continue

                    new_removed = {frozenset((a, b)), frozenset((c, d))}
                    type_begin = start_type if i == start_i and j == start_j else 0

                    for swap_type in range(type_begin, 2):
                        if swap_type == 0:
                            new_added = {frozenset((a, c)), frozenset((b, d))}
                        else:
                            new_added = {frozenset((a, d)), frozenset((b, c))}

                        if any(graph.has_edge(*edge) for edge in new_added):
                            continue
                        if removed & new_removed or added & new_added:
                            continue
                        if removed & new_added or added & new_removed:
                            continue

                        involves_u = u in {a, b, c, d}

                        if swap_type == 0:
                            next_i, next_j, next_type = i, j, 1
                        elif j + 1 < n_edges:
                            next_i, next_j, next_type = i, j + 1, 0
                        else:
                            next_i, next_j, next_type = i + 1, i + 2, 0

                        choose(4, next_j, next_type, depth + 1, removed | new_removed, added | new_added, touches_u or involves_u)

        choose(0, 1, 0, 0, set(), set(), False)

    return {u: _profile(signatures) for u, signatures in profiles.items()}

def _new_nodes(graph, count):
    start = max((v for v in graph.nodes() if isinstance(v, int)), default=-1) + 1
    return list(range(start, start + count))


def _add_clique(graph, vertices, size):
    clique = _new_nodes(graph, size)
    graph.add_nodes_from(clique)
    graph.add_edges_from(combinations(clique, 2))

    for u in vertices:
        graph.add_edges_from((u, v) for v in clique)


def _add_unique_clique_and_path(graph, u, size):
    # Private clique
    clique = _new_nodes(graph, size)
    graph.add_nodes_from(clique)
    graph.add_edges_from(combinations(clique, 2))
    graph.add_edges_from((u, v) for v in clique)

    # Private path of the same size
    path = _new_nodes(graph, size)
    graph.add_nodes_from(path)
    graph.add_edges_from(zip(path, path[1:]))
    graph.add_edge(u, path[0])


def _add_clique_probe(graph, group, size):
    # Shared clique
    _add_clique(graph, group, size)
    size += 1

    # Unique clique + path for every vertex in the group
    for vertex in group:
        _add_unique_clique_and_path(graph, vertex, size)
        size += 1

    return size

def _disjoint_group_sets_no_yield(nodes, p, count):
    order = {v: i for i, v in enumerate(nodes)}

    def generate(available, remaining_count, last_key=None):
        if remaining_count == 0:
            return [()]

        results = []

        for group in combinations(available, p):
            key = tuple(order[v] for v in group)

            if last_key is not None and key <= last_key:
                continue

            remaining = [v for v in available if v not in group]
            rests = generate(remaining, remaining_count - 1, key)

            for rest in rests:
                results.append((group,) + rest)

        return results

    return generate(list(nodes), count)
def _disjoint_group_sets(nodes, p, count):
    """
    Generate all ways of choosing 'count' disjoint groups of size p.
    At this stage, order inside a group and order between groups do not matter.
    """
    order = {v: i for i, v in enumerate(nodes)}

    def generate(available, remaining_count, last_key=None):
        if remaining_count == 0:
            yield ()
            return

        for group in combinations(available, p):
            key = tuple(order[v] for v in group)

            # Do not generate the same collection of groups
            # again with the groups in another order.
            if last_key is not None and key <= last_key:
                continue

            remaining = [v for v in available if v not in group]

            for rest in generate(remaining, remaining_count - 1, key):
                yield (group,) + rest

    yield from generate(list(nodes), count)


def clique_addition(graph, computer, cls, p, c, M=1):
    profiles = {u: [] for u in cls}
    nodes = list(graph.nodes())

    if p * M > len(nodes):
        raise ValueError("Not enough vertices for M disjoint groups of size p.")

    for u in tqdm(cls, desc="Clique addition probing", unit="vertex", dynamic_ncols=True):
        others = [v for v in nodes if v != u]

        # Root group:
        # u is ALWAYS fixed in the first position.
        for chosen in permutations(others, p - 1):
            root_group = (u,) + chosen
            remaining = [v for v in others if v not in chosen]

            # Choose the other M-1 disjoint groups.
            for extra_sets in _disjoint_group_sets(remaining, p, M - 1):

                for ordered_sets in permutations(extra_sets):

                    for ordered_groups in product(
                            *(permutations(group) for group in ordered_sets)
                    ):
                        probed = graph.copy()
                        size = c

                        size = _add_clique_probe(probed, root_group, size)

                        for group in ordered_groups:
                            size = _add_clique_probe(probed, group, size)

                        profiles[u].append(_signature(computer, probed, u))

    return {u: _profile(signatures) for u, signatures in profiles.items()}


def _fold_groups(graph, groups):
    mapping = {}

    for group, representative in groups:
        for v in group:
            mapping[v] = representative

    folded = nx.Graph()
    folded.add_nodes_from(mapping.get(v, v) for v in graph.nodes())

    for a, b in graph.edges():
        a_new, b_new = mapping.get(a, a), mapping.get(b, b)

        if a_new != b_new:
            folded.add_edge(a_new, b_new)

    return folded


def folding(graph, computer, cls, p, M=1):
    profiles = {u: [] for u in cls}
    nodes = list(graph.nodes())

    if p < 2:
        raise ValueError("p must be at least 2.")

    if p * M > len(nodes):
        raise ValueError("Not enough vertices for M disjoint groups of size p.")

    for u in cls:
        others = [v for v in nodes if v != u]

        # Choose the p-1 other vertices that will be folded with u.
        for chosen in combinations(others, p - 1):
            root_group = (u,) + chosen
            remaining = [v for v in others if v not in chosen]

            # Choose the other M-1 disjoint folding groups.
            for extra_groups in _disjoint_group_sets(remaining, p, M - 1):
                groups = [(root_group, u)]

                for group in extra_groups:
                    groups.append((group, group[0]))

                probed = _fold_groups(graph, groups)
                profiles[u].append(_signature(computer, probed, u))

    return {u: _profile(signatures) for u, signatures in profiles.items()}

def probe_partition(graph, computer, classes, probe, **kwargs):
    refined = []

    for cls in classes.values():
        if len(cls) == 1:
            refined.append(cls)
            continue

        profiles = probe(graph, computer, cls, **kwargs)

        groups = {}
        for u in cls:
            groups.setdefault(profiles[u], []).append(u)

        refined.extend(groups.values())

    return {i: cls for i, cls in enumerate(refined)}