from collections import defaultdict
from itertools import combinations
import networkx as nx
import numpy as np
from scipy.stats import linregress
from tqdm.auto import tqdm
from src.utils.graph_utils import construct_next_line_graph, construct_distance_graph

from src.module.vertex_equivalence_interface import VertexEquivalenceInterface

class HeatKernelEquivalenceComputer(VertexEquivalenceInterface):
    def __init__(
            self,
            graph_settings,
            distance_graph_order=1,
            comparison_mode="fixed_resolution",
            bfs_depth=6,
            weird_norm_threshold=1e-5,
            zero_tolerance=1e-12,
            use_ids=False,
    ):
        self.graph_settings = [dict(settings) for settings in graph_settings]

        for settings in self.graph_settings:
            settings["resolutions"] = np.atleast_1d(
                np.asarray(settings["resolutions"], dtype=np.float64)
            )
        self.distance_graph_order = distance_graph_order
        self.comparison_mode = comparison_mode
        self.bfs_depth = bfs_depth
        self.weird_norm_threshold = float(weird_norm_threshold)
        self.zero_tolerance = float(zero_tolerance)
        self.use_ids = use_ids


    def _positive_eigenvalues(self, eigenvalues):
        values = np.real(eigenvalues)
        values = values[values > self.zero_tolerance]
        return np.unique(np.sort(np.round(values, 12)))

    def _laplacian_data(self, G):
        nodes = list(G.nodes())
        laplacian = nx.laplacian_matrix(G, nodelist=nodes).toarray()

        return np.asarray(laplacian, dtype=np.float64), nodes

    def _time_values(self, eigenvalues, settings):
        positive_eigenvalues = self._positive_eigenvalues(eigenvalues)

        lambda_min = positive_eigenvalues[0]
        lambda_max = positive_eigenvalues[-1]

        start_time = 1 / lambda_max
        end_time = settings["end_time_multiplier"] / lambda_min

        return np.arange(start_time, end_time, settings["time_step"])

    def _spectral_data(self, G, settings):
        laplacian, items = self._laplacian_data(G)
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        return {
            "items": items,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "times": self._time_values(eigenvalues, settings),
        }

    def _dimension(self, eigenvalues):
        values = self._positive_eigenvalues(eigenvalues)
        low_points = 2
        if len(values) < low_points:
            raise ValueError("Coefficient fitting requires at least two unique positive eigenvalues.")

        x = np.log(values)
        y = np.log(np.arange(1, len(values) + 1) / len(values))
        scores = [linregress(x[:number_of_points], y[:number_of_points]).rvalue ** 2 for number_of_points in range(low_points, len(values) + 1)]
        best_number_of_points = low_points + int(np.argmax(scores))
        fit = linregress(x[:best_number_of_points], y[:best_number_of_points])

        return 2 * fit.slope

    @staticmethod
    def _heat_diagonal_values(spectral_data):
        eigenvalues = spectral_data["eigenvalues"]
        eigenvectors = spectral_data["eigenvectors"]
        times = spectral_data["times"]
        exponentials = np.exp(-times[:, None] * eigenvalues[None, :])

        return (exponentials @ (eigenvectors ** 2).T).T

    def _coefficient_values(self, spectral_data, settings):
        diagonal = self._heat_diagonal_values(spectral_data)
        eigenvalues = spectral_data["eigenvalues"]
        times = spectral_data["times"]
        dimension = self._dimension(eigenvalues)

        fitting_matrix = np.column_stack([
            (times ** coefficient_index) / ((4 * np.pi * times) ** (dimension / 2))
            for coefficient_index in range(settings["fit_order"])
        ])

        coefficients = diagonal @ np.linalg.pinv(fitting_matrix).T

        return coefficients[:, settings["fit_start"]:settings["fit_end"]]

    def _full_matrix_values(self, G, spectral_data, settings):
        items = spectral_data["items"]
        eigenvalues = spectral_data["eigenvalues"]
        eigenvectors = spectral_data["eigenvectors"]
        times = spectral_data["times"]

        size = len(items)
        values = np.zeros((size, len(times), size), dtype=np.float64)

        item_to_index = {item: index for index, item in enumerate(items)}

        distances = [
            {
                item_to_index[target]: distance
                for target, distance in nx.single_source_shortest_path_length(G, root).items()
            }
            for root in items
        ]

        for time_index, time_value in tqdm(enumerate(times),total=len(times),desc=f"{settings['name']} heat signatures",unit="time",dynamic_ncols=True):
            heat_matrix = (eigenvectors * np.exp(-eigenvalues * time_value)) @ eigenvectors.T

            for root in range(size):
                ordered_values = sorted((distances[root].get(target, size + 1),heat_matrix[root, target],)for target in range(size))
                values[root, time_index] = [entry[-1] for entry in ordered_values]

        return values.reshape(size, -1)


    def _compute_signatures(self, G, settings):
        spectral_data = self._spectral_data(G, settings)
        signature_mode = settings["signature_mode"]

        if signature_mode == "coefficients":
            values = self._coefficient_values(spectral_data, settings)
        elif signature_mode == "diagonal":
            values = self._heat_diagonal_values(spectral_data)
        else:
            values = self._full_matrix_values(G, spectral_data, settings)

        return {
            item: tuple(values[index].ravel())
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
            values = np.asarray(signature, dtype=np.float64)
            # truncated[item] = tuple(int(value) for value in np.trunc(values * resolutions))
            truncated[item] = tuple(int(value) for value in np.round(values * resolutions))

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
        classes = defaultdict(list)
        for item, signature in tqdm(signatures.items(), total=len(signatures), desc="Combining signatures", unit="sig", dynamic_ncols=True):
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
        for root in tqdm(nodes, total=len(nodes), desc=f"Computing BFS signatures", unit="node", dynamic_ncols=True):
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

        for distance in range(1, self.distance_graph_order + 1):
            distance_graphs = graphs if distance == 1 else [
                construct_distance_graph(graph, distance) for graph in graphs
            ]

            # Signatures for this Gi only.
            current_vertex_signatures = [{v: [] for v in vs} for vs in vertices]
            current_graphs = distance_graphs.copy()
            memberships = [{v: {v} for v in vs} for vs in vertices]

            for level, settings in enumerate(self.graph_settings):
                signatures = [self._compute_signatures(graph, settings) for graph in current_graphs]

                if self.comparison_mode == "fixed_resolution":
                    signatures = [self._truncate_signatures(sig, settings) for sig in signatures]

                if self.comparison_mode == "weird_norm":
                    if H is not None:
                        G_ids, H_ids, next_id = self.assign_weird_norm_ids(signatures[0], signatures[1], next_id)
                        values = [G_ids, H_ids]
                    else:
                        G_ids, _, next_id = self.assign_weird_norm_ids(signatures[0], start_id=next_id)
                        values = [G_ids]

                elif self.use_ids:
                    if H is not None:
                        G_ids, H_ids, next_id = self.assign_shared_ids(signatures[0], signatures[1], next_id)
                        values = [G_ids, H_ids]
                    else:
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
                        current_vertex_signatures[k][v].append(
                            tuple(sorted(values[k][item] for item in memberships[k][v]))
                        )

                if self.use_ids or self.comparison_mode == "weird_norm":
                    del signatures, values

                if level < len(self.graph_settings) - 1:
                    for k in range(len(graphs)):
                        current_graphs[k], memberships[k] = construct_next_line_graph(
                            current_graphs[k], memberships[k]
                        )

            # Apply BFS independently on this Gi.
            for k in range(len(graphs)):
                gi_signatures = {
                    v: tuple(signature)
                    for v, signature in current_vertex_signatures[k].items()
                }

                if self.bfs_depth is not None:
                    gi_signatures = self._signatures_with_bfs(distance_graphs[k], gi_signatures)

                for v in vertices[k]:
                    vertex_signatures[k][v].append(gi_signatures[v])

        # Combine the results from G1, G2, ...
        results = []

        for k in range(len(graphs)):
            sigs = {v: tuple(signature) for v, signature in vertex_signatures[k].items()}
            classes = self._group_by_equality(sigs)
            results.append((classes, sigs))

        return results[0] if H is None else (*results[0], *results[1])