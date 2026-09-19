from collections import defaultdict
from itertools import combinations
import networkx as nx
import numpy as np
from scipy.stats import linregress
import mpmath as mp
from tqdm.auto import tqdm
from src.utils.graph_utils import construct_transformed_graph
from scipy.sparse.csgraph import shortest_path

from src.module.vertex_equivalence_interface import VertexEquivalenceInterface
from src.utils.memory_capacity import guard_memory, guard_remaining_memory
import hashlib
FLOAT64_BYTES = 8
COMPLEX128_BYTES = 16
class HeatKernelEquivalenceComputer(VertexEquivalenceInterface):
    def __init__(
            self,
            graph_settings,
            comparison_mode="fixed_resolution",
            bfs_depth=6,
            weird_norm_threshold=1e-5,
            zero_tolerance=1e-12,
            use_ids=False,
            use_stable_ids = False,
            show_progress=True,
    ):
        self.graph_settings = [dict(settings) for settings in graph_settings]

        for settings in self.graph_settings:
            settings["resolutions"] = np.atleast_1d(
                np.asarray(settings["resolutions"], dtype=np.float64)
            )

        self.comparison_mode = comparison_mode
        self.bfs_depth = bfs_depth
        self.weird_norm_threshold = float(weird_norm_threshold)
        self.zero_tolerance = float(zero_tolerance)
        self.use_ids = use_ids
        self.use_stable_ids = use_stable_ids
        self.show_progress = show_progress

    def _use_mpmath(self, settings):
        return settings.get("precision_dps") is not None

    def _stable_id(self, signature):
        h = hashlib.blake2b(digest_size=32)

        if isinstance(signature, np.ndarray):
            array = np.ascontiguousarray(signature)

            # Make +0.0 and -0.0 identical
            array[array == 0] = 0.0

            h.update(array.dtype.str.encode())
            h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            h.update(memoryview(array).cast("B"))

        else:
            h.update(repr(signature).encode("utf-8"))

        return int.from_bytes(h.digest(), "big")

    def _eigendecomposition(self, matrix, settings):
        precision_dps = settings.get("precision_dps")

        if precision_dps is None:
            return np.linalg.eigh(matrix)

        with mp.workdps(precision_dps):
            matrix_mp = mp.matrix([
                [mp.mpf(value) for value in row]
                for row in matrix
            ])

            eigenvalues, eigenvectors = mp.eigsy(matrix_mp)

        return eigenvalues, eigenvectors

    def _positive_eigenvalues(self, eigenvalues, settings):
        if not self._use_mpmath(settings):
            values = np.real(eigenvalues)
            values = values[values > self.zero_tolerance]
            return np.unique(np.sort(np.round(values, 12)))

        tol = mp.mpf(str(self.zero_tolerance))
        values = sorted(eigenvalues[i] for i in range(len(eigenvalues)) if eigenvalues[i] > tol)

        unique_values = []
        for value in values:
            if not unique_values or abs(value - unique_values[-1]) > tol:
                unique_values.append(value)

        return unique_values


    def _time_values(self, eigenvalues, settings):
        positive_eigenvalues = self._positive_eigenvalues(eigenvalues, settings)

        lambda_min = positive_eigenvalues[0]
        lambda_max = positive_eigenvalues[-1]

        start_time = 1 / lambda_max
        end_time = settings["end_time_multiplier"] / lambda_min

        return np.arange(start_time, end_time, settings["time_step"])


    def _spectral_data(self, G, settings, operator):
        nodes = list(G.nodes())
        data = {"items": nodes}
        edges = list(G.edges())
        n = len(nodes)
        m = len(edges)
        dimension = n if operator == "laplacian" else n + m

        guard_memory(
            6 * FLOAT64_BYTES * dimension ** 2,
            f"{operator} eigendecomposition ({dimension} x {dimension})"
        )
        if operator == "laplacian":
            matrix = nx.laplacian_matrix(G, nodelist=nodes).toarray()

        elif operator == "dirac":
            B = nx.incidence_matrix(G, nodelist=nodes, edgelist=edges, oriented=True).toarray()
            matrix = np.block([
                [np.zeros((n, n)), B],
                [B.T, np.zeros((m, m))]
            ])

            data["edges"] = edges

        eigenvalues, eigenvectors = self._eigendecomposition(matrix,settings)

        data.update({
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "times": self._time_values(eigenvalues, settings),
        })

        return data

    def _dimension(self, eigenvalues, settings):
        values = self._positive_eigenvalues(eigenvalues, settings)
        low_points = 2
        if len(values) < low_points:
            raise ValueError("Coefficient fitting requires at least two unique positive eigenvalues.")

        x = np.log(values)
        y = np.log(np.arange(1, len(values) + 1) / len(values))
        scores = [linregress(x[:number_of_points], y[:number_of_points]).rvalue ** 2 for number_of_points in range(low_points, len(values) + 1)]
        best_number_of_points = low_points + int(np.argmax(scores))
        fit = linregress(x[:best_number_of_points], y[:best_number_of_points])

        return 2 * fit.slope

    def _heat_diagonal_values(self, spectral_data, settings):
        eigenvalues = spectral_data["eigenvalues"]
        eigenvectors = spectral_data["eigenvectors"]
        times = spectral_data["times"]

        if not self._use_mpmath(settings):
            n = len(eigenvalues)
            t = len(times)

            required_bytes = FLOAT64_BYTES * (n ** 2 + 2 * t * n)
            guard_memory(required_bytes,f"heat diagonal computation "f"(n={n}, time_points={t})")
            exponentials = np.exp(-times[:, None] * eigenvalues[None, :])
            return (exponentials @ (eigenvectors ** 2).T).T

        with mp.workdps(settings["precision_dps"]):
            n = eigenvectors.rows
            times_mp = [mp.mpf(str(t)) for t in times]
            return [
                [mp.fsum(eigenvectors[v, k] ** 2 * mp.exp(-eigenvalues[k] * t)
                         for k in range(len(eigenvalues))) for t in times_mp]
                for v in range(n)
            ]

    def _coefficient_values(self, spectral_data, settings):
        diagonal = self._heat_diagonal_values(spectral_data, settings)
        eigenvalues = spectral_data["eigenvalues"]
        times = spectral_data["times"]
        dimension = self._dimension(eigenvalues, settings)

        if not self._use_mpmath(settings):
            fitting_matrix = np.column_stack([
                times ** i / ((4 * np.pi * times) ** (dimension / 2))
                for i in range(settings["fit_order"])
            ])
            coefficients = diagonal @ np.linalg.pinv(fitting_matrix).T
            return coefficients[:, settings["fit_start"]:settings["fit_end"]]

        with mp.workdps(settings["precision_dps"]):
            d = mp.mpf(str(dimension))
            A = mp.matrix([
                [
                    mp.mpf(str(t)) ** i /
                    ((4 * mp.pi * mp.mpf(str(t))) ** (d / 2))
                    for i in range(settings["fit_order"])
                ]
                for t in times
            ])
            result = []
            for vertex_diagonal in diagonal:
                coefficients, _ = mp.qr_solve(A, mp.matrix(vertex_diagonal))
                result.append([coefficients[i] for i in range(settings["fit_start"], settings["fit_end"])])
            return result

    # @profile
    def _full_matrix_values(self, G, spectral_data, settings):
        items = spectral_data["items"]
        eigenvalues = spectral_data["eigenvalues"]
        eigenvectors = spectral_data["eigenvectors"]
        times = spectral_data["times"]

        size = len(items)
        use_mp = self._use_mpmath(settings)
        bfs_sort = settings.get("bfs_sort", True)

        if use_mp:
            values = [[[] for _ in times] for _ in range(size)]
        else:
            values = np.zeros((size, len(times), size), dtype=np.float64)

        if bfs_sort:
            adjacency = nx.to_scipy_sparse_array(G, nodelist=items, format="csr", dtype=float)
            distance_matrix = shortest_path(adjacency, directed=False, unweighted=True)
            distance_matrix[np.isinf(distance_matrix)] = size + 1

        for time_index, time_value in tqdm(enumerate(times), total=len(times),
                                               desc=f"{settings['name']} heat signatures", unit="time",
                                               dynamic_ncols=True, disable=not self.show_progress):
            if use_mp:
                with mp.workdps(settings["precision_dps"]):
                    t = mp.mpf(str(time_value))
                    D = mp.diag([mp.exp(-eigenvalues[i] * t) for i in range(len(eigenvalues))])
                    heat_matrix = eigenvectors * D * eigenvectors.T
            else:
                heat_matrix = (eigenvectors * np.exp(-eigenvalues * time_value)) @ eigenvectors.T

            for root in range(size):
                if use_mp:
                    if bfs_sort:
                        ordered_values = sorted(
                            (distance_matrix[root, target], heat_matrix[root, target]) for target in range(size))
                        values[root][time_index] = [entry[-1] for entry in ordered_values]
                    else:
                        values[root][time_index] = sorted(heat_matrix[root, target] for target in range(size))
                else:
                    if bfs_sort:
                        order = np.lexsort((heat_matrix[root], distance_matrix[root]))
                        values[root, time_index] = heat_matrix[root, order]
                    else:
                        values[root, time_index] = np.sort(heat_matrix[root])
        if use_mp:
            return [
                [value for time_values in values[root] for value in time_values]
                for root in range(size)
            ]

        return values.reshape(size, -1)


    def _full_matrix_optimized_helper(self, G, spectral_data, settings):
        """Yield one truncated full-matrix signature dictionary per time instance."""
        items = spectral_data["items"]
        eigenvalues = spectral_data["eigenvalues"]
        eigenvectors = spectral_data["eigenvectors"]
        times = spectral_data["times"]
        size = len(items)
        resolution = settings["resolutions"][0]
        bfs_sort = settings.get("bfs_sort", True)
        matrix_bytes = FLOAT64_BYTES * size ** 2
        if bfs_sort:
            guard_memory(
                2 * matrix_bytes,
                f"shortest-path matrix ({size} x {size})"
            )
            adjacency = nx.to_scipy_sparse_array(G, nodelist=items, format="csr", dtype=float)
            distance_matrix = shortest_path(adjacency, directed=False, unweighted=True)
            distance_matrix[np.isinf(distance_matrix)] = size + 1

        for time_value in tqdm(times, total=len(times), desc=f"{settings['name']} heat signatures", unit="time",dynamic_ncols=True, disable=not self.show_progress):

            guard_memory(
                3 * matrix_bytes,
                f"heat matrix ({size} x {size})"
            )
            heat_matrix = (eigenvectors * np.exp(-eigenvalues * time_value)) @ eigenvectors.T

            for root in range(size):
                if bfs_sort:
                    order = np.lexsort((heat_matrix[root], distance_matrix[root]))
                    heat_matrix[root] = heat_matrix[root, order]
                else:
                    heat_matrix[root].sort()

            np.multiply(heat_matrix, resolution, out=heat_matrix)
            np.trunc(heat_matrix, out=heat_matrix)

            yield {item: heat_matrix[index] for index, item in enumerate(items)}

    def _full_matrix_optimized(self, graphs, settings, start_id=0):
        spectral_data = [self._spectral_data(g, settings, "laplacian") for g in graphs]
        generators = [
            self._full_matrix_optimized_helper(g, d, settings)
            for g, d in zip(graphs, spectral_data)
        ]
        time_instances = len(spectral_data[0]["times"])

        if self.use_stable_ids:
            hashers = [
                {item: hashlib.blake2b(digest_size=32) for item in d["items"]}
                for d in spectral_data
            ]

            for _ in range(time_instances):
                signatures = [next(gen) for gen in generators]

                for k, sigs in enumerate(signatures):
                    for item, sig in sigs.items():
                        arr = np.ascontiguousarray(sig.copy())
                        arr[arr == 0] = 0.0
                        hashers[k][item].update(arr.dtype.str.encode())
                        hashers[k][item].update(
                            np.asarray(arr.shape, dtype=np.int64).tobytes()
                        )
                        hashers[k][item].update(memoryview(arr).cast("B"))

            ids = [
                {
                    item: int.from_bytes(h.digest(), "big")
                    for item, h in graph_hashers.items()
                }
                for graph_hashers in hashers
            ]
            return ids, start_id

        classes = [[
            (k, item)
            for k, data in enumerate(spectral_data)
            for item in data["items"]
        ]]

        for _ in range(time_instances):
            signatures = [next(gen) for gen in generators]
            new_classes = []

            for current_class in classes:
                if len(current_class) == 1:
                    new_classes.append(current_class)
                    continue

                matrix = np.stack([signatures[k][item] for k, item in current_class])
                _, inverse = np.unique(matrix, axis=0, return_inverse=True)

                splits = defaultdict(list)
                for record, class_id in zip(current_class, inverse):
                    splits[int(class_id)].append(record)

                new_classes.extend(splits.values())

            classes = new_classes

        ids = [{} for _ in graphs]
        for current_class in classes:
            for k, item in current_class:
                ids[k][item] = start_id
            start_id += 1

        return ids, start_id

    def _full_correlation_values(self, G, settings):
        items = list(G.nodes())
        size = len(items)

        L = nx.laplacian_matrix(G, nodelist=items).toarray()
        C = np.linalg.pinv(L)
        del L

        if settings.get("bfs_sort", True):
            adjacency = nx.to_scipy_sparse_array(G, nodelist=items, format="csr", dtype=float)
            distance_matrix = shortest_path(adjacency, directed=False, unweighted=True)
            distance_matrix[np.isinf(distance_matrix)] = size + 1

            for root in range(size):
                order = np.lexsort((C[root], distance_matrix[root]))
                C[root] = C[root, order]

        else:
            C.sort(axis=1)

        return C

    def _topological_signatures(self, G):
        nodes = list(G.nodes())

        clustering = nx.clustering(G)
        degree = dict(G.degree())
        average_neighbor_degree = nx.average_neighbor_degree(G)
        betweenness = nx.betweenness_centrality(G)
        communicability = nx.communicability(G)

        return {
            v: (
                   clustering[v],
                   degree[v],
                   average_neighbor_degree[v],
                   betweenness[v],
               ) + tuple(sorted(communicability[v].values()))
            for v in nodes
        }

    def _dirac_signatures(self, G, spectral_data, settings):
        nodes = spectral_data["items"]
        edges = spectral_data["edges"]
        eigenvalues = spectral_data["eigenvalues"]
        eigenvectors = spectral_data["eigenvectors"]
        times = spectral_data["times"]

        n = len(nodes)
        m = len(edges)
        dimension = n + m
        decimals = settings.get("dirac_decimals", 6)
        use_mp = self._use_mpmath(settings)

        def complex_signature(values):
            if use_mp:
                scale = mp.mpf(10) ** decimals
                return tuple(sorted(
                    (mp.nint(abs(mp.re(z)) * scale) / scale,
                     mp.nint(abs(mp.im(z)) * scale) / scale)
                    for z in values
                ))

            pairs = np.column_stack((
                np.round(np.abs(values.real), decimals),
                np.round(np.abs(values.imag), decimals)
            ))

            pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]

            return tuple(map(tuple, pairs))

        node_signatures = {v: [] for v in nodes}
        edge_signatures = {i: [] for i in range(m)}

        kernel_bytes = COMPLEX128_BYTES * dimension ** 2
        for time_value in tqdm(times, total=len(times), desc=f"{settings['name']} Dirac signatures", unit="time", dynamic_ncols=True, disable=not self.show_progress):
            if use_mp:
                with mp.workdps(settings["precision_dps"]):
                    t = mp.mpf(str(time_value))
                    D = mp.diag([mp.exp(mp.j * eigenvalues[i] * t) for i in range(len(eigenvalues))])
                    kernel = eigenvectors * D * eigenvectors.T
            else:
                guard_memory(
                    3 * kernel_bytes,
                    f"Dirac kernel ({dimension} x {dimension})",
                )
                kernel = (eigenvectors * np.exp(1j * eigenvalues * time_value)) @ eigenvectors.T

            for i, v in enumerate(nodes):
                if use_mp:
                    vv_values = [kernel[i, j] for j in range(n)]
                    ve_values = [kernel[i, n + j] for j in range(m)]
                else:
                    vv_values = kernel[i, :n]
                    ve_values = kernel[i, n:]

                node_signatures[v].append(complex_signature(vv_values))
                node_signatures[v].append(complex_signature(ve_values))

            for i in range(m):
                if use_mp:
                    ve_values = [kernel[j, n + i] for j in range(n)]
                    ee_values = [kernel[n + i, n + j] for j in range(m)]
                else:
                    ve_values = kernel[:n, n + i]
                    ee_values = kernel[n + i, n:]

                edge_signatures[i].append(complex_signature(ve_values))
                edge_signatures[i].append(complex_signature(ee_values))

            del kernel
            # 2. Check accumulated Python signatures
            guard_remaining_memory(
                f"{settings['name']} Dirac signature storage",
                reserve_percent=10,
            )

        guard_remaining_memory(
            f"{settings['name']} Dirac incident-edge aggregation",
            reserve_percent=10,
        )
        incidences = {v: [] for v in nodes}

        for i, (u, v) in enumerate(edges):
            incidences[u].append(i)
            incidences[v].append(i)

        for v in nodes:
            incident_edge_signatures = [tuple(edge_signatures[i]) for i in incidences[v]]
            node_signatures[v].append(tuple(sorted(incident_edge_signatures)))
            node_signatures[v] = tuple(node_signatures[v])

        return node_signatures


    def _dirac_correlation_signatures(self, G, settings):
        nodes = list(G.nodes())
        edges = list(G.edges())

        n = len(nodes)
        m = len(edges)
        dimension = n + m
        decimals = settings.get("dirac_decimals", 6)
        use_mp = self._use_mpmath(settings)

        def complex_signature(values):
            if use_mp:
                scale = mp.mpf(10) ** decimals
                return tuple(sorted(
                    (mp.nint(abs(mp.re(z)) * scale) / scale,
                        mp.nint(abs(mp.im(z)) * scale) / scale)
                    for z in values
                ))

            pairs = np.column_stack((
                np.round(np.abs(values.real), decimals),
                np.round(np.abs(values.imag), decimals)
            ))

            pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]

            return tuple(map(tuple, pairs))

        node_signatures = {v: [] for v in nodes}
        edge_signatures = {i: [] for i in range(m)}

        B = nx.incidence_matrix(G, nodelist=nodes, edgelist=edges, oriented=True).toarray()
        mass = settings.get("dirac_mass", 1.0)

        matrix = np.block([
            [mass * np.eye(n), B],
            [B.T, -mass * np.eye(m)]
        ])

        kernel = np.linalg.inv(matrix)
        del matrix

        for i, v in tqdm(enumerate(nodes), total=len(nodes), desc="Dirac Correlation Nodes Signatures", unit="vertex", dynamic_ncols=True):
            vv_values = kernel[i, :n]
            ve_values = kernel[i, n:]

            node_signatures[v].append(complex_signature(vv_values))
            node_signatures[v].append(complex_signature(ve_values))

        for i in tqdm(range(m), total=len(edges), desc="Dirac Correlation Edges Signatures", unit="edge", dynamic_ncols=True):
            ve_values = kernel[:n, n + i]
            ee_values = kernel[n + i, n:]

            edge_signatures[i].append(complex_signature(ve_values))
            edge_signatures[i].append(complex_signature(ee_values))

        del kernel

        incidences = {v: [] for v in nodes}

        for i, (u, v) in enumerate(edges):
            incidences[u].append(i)
            incidences[v].append(i)

        for v in nodes:
            incident_edge_signatures = [tuple(edge_signatures[i]) for i in incidences[v]]
            node_signatures[v].append(tuple(sorted(incident_edge_signatures)))
            node_signatures[v] = tuple(node_signatures[v])

        return node_signatures


    def _compute_signatures(self, G, settings, operator):
        if operator == "correlation":
            values = self._full_correlation_values(G, settings)
            items = list(G.nodes())
            return {item: values[index] for index, item in enumerate(items)}

        if operator == "dirac_correlation":
            values = self._dirac_correlation_signatures(G, settings)
            items = list(G.nodes())
            return {item: values[index] for index, item in enumerate(items)}

        if operator == "topological":
            return self._topological_signatures(G)

        spectral_data = self._spectral_data(G, settings, operator)
        if operator == "dirac":
            return self._dirac_signatures(G, spectral_data, settings)

        signature_mode = settings["signature_mode"]

        if signature_mode == "coefficients":
            values = self._coefficient_values(spectral_data, settings)
        elif signature_mode == "diagonal":
            values = self._heat_diagonal_values(spectral_data, settings)
        else:
            values = self._full_matrix_values(G, spectral_data, settings)

        return {
            item: values[index].ravel() if isinstance(values[index], np.ndarray)
                                        else tuple(values[index])
            for index, item in enumerate(spectral_data["items"])
        }

    def _truncate_signatures(self, signatures, settings):
        signature_mode = settings["signature_mode"]
        resolutions = settings["resolutions"]
        truncated = {}

        if signature_mode == "coefficients":
            number_of_coefficients = settings["fit_end"] - settings["fit_start"]

            if len(resolutions) < number_of_coefficients:
                raise ValueError("Not enough resolution values for the selected coefficients.")

            resolutions = resolutions[:number_of_coefficients]
        else:
            resolutions = resolutions[0]

        for item, signature in signatures.items():
            if self._use_mpmath(settings):
                if signature_mode == "coefficients":
                    res = [mp.mpf(str(r)) for r in resolutions]
                    truncated[item] = tuple(
                        int(mp.nint(value * res[i])) for i, value in enumerate(signature)
                    )
                else:
                    res = mp.mpf(str(resolutions))
                    truncated[item] = tuple(int(mp.nint(value * res)) for value in signature)

            else:
                np.multiply(signature, resolutions, out=signature)
                np.trunc(signature, out=signature)
                truncated[item] = signature

        return truncated

    def _weird_norm(self, U, V):
        U = np.asarray(U,dtype=np.float64).copy()
        V = np.asarray(V,dtype=np.float64).copy()

        U[np.abs(U) < self.zero_tolerance] = 0.0
        V[np.abs(V) < self.zero_tolerance] = 0.0

        differences = np.zeros_like(U)
        both_nonzero = (U != 0.0) & (V != 0.0)
        one_nonzero = (U != 0.0) ^ (V != 0.0)

        differences[both_nonzero] = (U[both_nonzero] - V[both_nonzero]) / U[both_nonzero]
        differences[one_nonzero] = np.where(U[one_nonzero] != 0.0,U[one_nonzero],V[one_nonzero])

        return np.linalg.norm(differences)

    def _group_by_equality(self, signatures):
        first = next(iter(signatures.values()))

        if isinstance(first, np.ndarray):
            items = list(signatures)
            stack_bytes = len(items) * first.nbytes
            guard_memory(stack_bytes,f"signature stack "f"({len(items)} x {first.size})")

            matrix = np.stack([signatures[item] for item in items])

            guard_memory(3 * matrix.nbytes,f"signature unique "f"({matrix.shape[0]} x {matrix.shape[1]})")

            _, inverse = np.unique(matrix, axis=0, return_inverse=True)

            del matrix
            classes = defaultdict(list)
            for item, class_id in zip(items, inverse):
                classes[class_id].append(item)

            return classes

        classes = defaultdict(list)
        for item, signature in signatures.items():
            classes[signature].append(item)

        return classes
    def _group_by_weird_norm(self, signatures):
        similarity_graph = nx.Graph()
        similarity_graph.add_nodes_from(signatures)

        for item1, item2 in combinations(signatures, 2):
            distance = max(
                self._weird_norm(signatures[item1], signatures[item2]),
                self._weird_norm(signatures[item2], signatures[item1])
            )

            if distance < self.weird_norm_threshold:
                similarity_graph.add_edge(item1, item2)

        return {
            i: sorted(component)
            for i, component in enumerate(nx.connected_components(similarity_graph))
        }

    def assign_shared_ids(self, G_signatures, H_signatures, start_id=0):
        first = next(iter(G_signatures.values()))

        if isinstance(first, np.ndarray):
            G_items = list(G_signatures)
            H_items = list(H_signatures)

            number_of_rows = len(G_items) + len(H_items)
            stack_bytes = number_of_rows * first.nbytes

            guard_memory(stack_bytes,f"shared signature stack "f"({number_of_rows} x {first.size})")

            matrix = np.stack(
                [G_signatures[item] for item in G_items] +
                [H_signatures[item] for item in H_items]
            )

            guard_memory(3 * matrix.nbytes,f"shared signature unique "f"({matrix.shape[0]} x {matrix.shape[1]})")

            unique, inverse = np.unique(matrix, axis=0, return_inverse=True)

            G_ids = {item: start_id + int(class_id) for item, class_id in zip(G_items, inverse[:len(G_items)])}
            H_ids = {item: start_id + int(class_id) for item, class_id in zip(H_items, inverse[len(G_items):])}

            return G_ids, H_ids, start_id + len(unique)

        signature_to_id = {}
        G_ids = {}
        H_ids = {}
        next_id = start_id

        for item, signature in G_signatures.items():
            if signature not in signature_to_id:
                signature_to_id[signature] = next_id
                next_id += 1
            G_ids[item] = signature_to_id[signature]

        for item, signature in H_signatures.items():
            if signature not in signature_to_id:
                signature_to_id[signature] = next_id
                next_id += 1
            H_ids[item] = signature_to_id[signature]

        return G_ids, H_ids, next_id

    def assign_weird_norm_ids(self, G_signatures, H_signatures=None, start_id=0):
        records = [("G", item, sig) for item, sig in G_signatures.items()]

        if H_signatures is not None:
            records += [("H", item, sig) for item, sig in H_signatures.items()]

        similarity_graph = nx.Graph()
        similarity_graph.add_nodes_from(range(len(records)))

        number_of_pairs = len(records) * (len(records) - 1) // 2

        for i, j in tqdm(combinations(range(len(records)), 2), total=number_of_pairs,desc="Weird-norm comparisons", unit="pair", dynamic_ncols=True):
            signature_i = records[i][2]
            signature_j = records[j][2]

            distance = max(
                self._weird_norm(signature_i, signature_j),
                self._weird_norm(signature_j, signature_i)
            )

            if distance < self.weird_norm_threshold:
                similarity_graph.add_edge(i, j)

        G_ids = {}
        H_ids = {} if H_signatures is not None else None
        next_id = start_id

        for component in nx.connected_components(similarity_graph):
            for index in component:
                graph_name, item, _ = records[index]

                if graph_name == "G":
                    G_ids[item] = next_id
                else:
                    H_ids[item] = next_id

            next_id += 1

        return G_ids, H_ids, next_id


    def _signatures_with_bfs(self, G, vertex_signatures):
        root_signatures = {}

        nodes = G.nodes()
        for root in tqdm(nodes, total=len(nodes), desc="Computing BFS signatures", unit="node", dynamic_ncols=True,
                         disable=not self.show_progress):
            distances = nx.single_source_shortest_path_length(
                G,
                root,
                cutoff=self.bfs_depth,
            )

            layers = [[] for _ in range(self.bfs_depth + 1)]

            for vertex, depth in distances.items():
                layers[depth].append(vertex_signatures[vertex])

            root_signatures[root] = tuple(
                tuple(sorted(layer))
                for layer in layers
            )

        return root_signatures

    def compute_vertex_signatures(self, G, H=None):
        graphs = [G] if H is None else [G, H]
        vertices = [list(graph.nodes()) for graph in graphs]
        vertex_signatures = [{v: [] for v in vs} for vs in vertices]
        next_id = 0

        for settings in self.graph_settings:
            current_graphs = []
            memberships = []

            for graph in graphs:
                current_graph, membership = construct_transformed_graph(graph,settings["transformations"],)
                current_graphs.append(current_graph)
                memberships.append(membership)

            for operator in settings["operators"]:
                optimized_full_matrix = operator == "laplacian" and settings[
                    "signature_mode"] == "full_matrix" and self.use_ids and self.comparison_mode == "fixed_resolution" and not self._use_mpmath(
                    settings)

                # optimized_full_matrix = False

                if optimized_full_matrix:
                    values, next_id = self._full_matrix_optimized(current_graphs, settings, next_id)

                else:

                    signatures = [self._compute_signatures(graph, settings, operator) for graph in current_graphs]
                    if (self.comparison_mode == "fixed_resolution" and operator in ["laplacian", "topological", "correlation"]):
                        signatures = [self._truncate_signatures(sig, settings) for sig in signatures]

                    if self.comparison_mode == "weird_norm":
                        if H is not None:
                            G_ids, H_ids, next_id = self.assign_weird_norm_ids(signatures[0],signatures[1],next_id,)
                            values = [G_ids, H_ids]
                        else:
                            G_ids, _, next_id = self.assign_weird_norm_ids(signatures[0],start_id=next_id,)
                            values = [G_ids]


                    elif self.use_ids:

                        if self.use_stable_ids:
                            # IDs must be comparable across separate calls
                            values = [
                                {
                                    item: self._stable_id(signature)
                                    for item, signature in graph_signatures.items()
                                }
                                for graph_signatures in signatures
                            ]

                        elif H is not None:
                            # G and H are being processed together,
                            # so assign IDs jointly
                            G_ids, H_ids, next_id = self.assign_shared_ids(
                                signatures[0],
                                signatures[1],
                                next_id,
                            )
                            values = [G_ids, H_ids]

                        else:
                            # Normal single-graph run.
                            # IDs only need to be meaningful inside this call.
                            classes = self._group_by_equality(signatures[0])
                            ids = {}

                            for items in classes.values():
                                for item in items:
                                    ids[item] = next_id
                                next_id += 1

                            values = [ids]

                    else:
                        values = signatures

                for k in range(len(graphs)):
                    for v in vertices[k]:
                        parts = [tuple(values[k][item]) if isinstance(values[k][item], np.ndarray)
                                 else values[k][item] for item in memberships[k][v]]

                        vertex_signatures[k][v].append(tuple(sorted(parts)))



                if self.comparison_mode == "weird_norm":
                    del signatures, values

        results = []

        for k in range(len(graphs)):
            sigs = {v: tuple(signature) for v, signature in vertex_signatures[k].items()}

            if self.bfs_depth is not None:
                sigs = self._signatures_with_bfs(graphs[k],sigs,)

            classes = self._group_by_equality(sigs)
            results.append((classes, sigs))

        return results[0] if H is None else (*results[0], *results[1])