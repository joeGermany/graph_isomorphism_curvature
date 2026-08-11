from collections import defaultdict
from itertools import combinations
import networkx as nx
import numpy as np
from scipy.stats import linregress
from tqdm.auto import tqdm

from src.module.vertex_equivalence_interface import VertexEquivalenceInterface, EquivalenceClasses

class HeatKernelEquivalenceComputer(VertexEquivalenceInterface):
    def __init__(
            self,
            node_settings,
            edge_settings=None,
            comparison_mode="fixed_resolution",
            include_edges=False,
            bfs_depth=6,
            weird_norm_threshold=1e-5,
            zero_tolerance=1e-12,
    ):
        self.node_settings = dict(node_settings)
        self.node_settings["resolutions"] = np.atleast_1d(
            np.asarray(self.node_settings["resolutions"], dtype=np.float64))

        self.edge_settings = dict(edge_settings) if edge_settings is not None else None

        if self.edge_settings is not None:
            self.edge_settings["resolutions"] = np.atleast_1d(
                np.asarray(self.edge_settings["resolutions"], dtype=np.float64))

        self.comparison_mode = comparison_mode
        self.include_edges = include_edges
        self.bfs_depth = bfs_depth
        self.weird_norm_threshold = float(weird_norm_threshold)
        self.zero_tolerance = float(zero_tolerance)

    def _domain_settings(self, domain):
        return self.node_settings if domain == "node" else self.edge_settings

    @staticmethod
    def _edge_key(u, v):
        return (min(u, v), max(u, v))

    def _positive_eigenvalues(self, eigenvalues):
        values = np.real(eigenvalues)
        values = values[values > self.zero_tolerance]
        return np.unique(np.sort(np.round(values, 12)))

    def _laplacian_data(self, G, domain):
        nodes = list(G.nodes())

        if domain == "node":
            laplacian = nx.laplacian_matrix(G, nodelist=nodes).toarray()
            return np.asarray(laplacian, dtype=np.float64), nodes, None

        if domain == "edge":
            edges = list(G.edges())
            incidence = nx.incidence_matrix(G, nodelist=nodes, edgelist=edges, oriented=True).toarray()
            edge_items = [self._edge_key(u, v) for u, v in edges]
            return np.asarray(incidence.T @ incidence, dtype=np.float64), edge_items, edges

    def _time_values(self, eigenvalues, domain):
        settings = self._domain_settings(domain)
        positive_eigenvalues = self._positive_eigenvalues(eigenvalues)

        lambda_min = positive_eigenvalues[0]
        lambda_max = positive_eigenvalues[-1]

        start_time = 1 / lambda_max
        end_time = settings["end_time_multiplier"] / lambda_min

        return np.arange(start_time, end_time, settings["time_step"])

    def _spectral_data(self, G, domain):
        laplacian, items, edges = self._laplacian_data(G, domain)
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        return {
            "items": items,
            "edges": edges,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "times": self._time_values(eigenvalues, domain),
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

    def _coefficient_values(self, spectral_data, domain):
        settings = self._domain_settings(domain)

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

    @staticmethod
    def _edge_graph(edges):
        edge_graph = nx.Graph()
        edge_graph.add_nodes_from(range(len(edges)))
        incident_edges = defaultdict(list)

        for edge_index, (u, v) in enumerate(edges):
            incident_edges[u].append(edge_index)
            incident_edges[v].append(edge_index)

        for edge_indices in incident_edges.values():
            edge_graph.add_edges_from(combinations(edge_indices, 2))

        return edge_graph

    def _full_matrix_values(self, G, domain, spectral_data):

        items = spectral_data["items"]
        edges = spectral_data["edges"]
        eigenvalues = spectral_data["eigenvalues"]
        eigenvectors = spectral_data["eigenvectors"]
        times = spectral_data["times"]

        size = len(items)
        values = np.zeros((size,len(times),size), dtype=np.float64)

        if domain == "node":
            item_to_index = {item:index for index,item in enumerate(items)}
            distances = [
                {
                    item_to_index[target]:distance
                    for target,distance in nx.single_source_shortest_path_length(G,root).items()
                }
                for root in items
            ]
        else:
            edge_graph = self._edge_graph(edges)
            distances = [nx.single_source_shortest_path_length(edge_graph,root) for root in range(size)]

        for time_index, time_value in tqdm(enumerate(times), total=len(times),
                                           desc=f"{domain.capitalize()} heat signatures", unit="time",
                                           dynamic_ncols=True):
            heat_matrix = (eigenvectors * np.exp(-eigenvalues * time_value)) @ eigenvectors.T

            if domain == "edge":
                heat_matrix = np.abs(heat_matrix)

            for root in range(size):
                ordered_values = sorted(
                    (
                        distances[root].get(target, size + 1),
                        heat_matrix[root, target],
                    )
                    for target in range(size)
                )

                values[root, time_index] = [entry[-1] for entry in ordered_values]

        return values.reshape(size,-1)


    def _compute_signatures(self, G, domain):
        spectral_data = self._spectral_data(G, domain)
        settings = self._domain_settings(domain)
        signature_mode = settings["signature_mode"]

        if signature_mode == "coefficients":
            values = self._coefficient_values(spectral_data, domain)
        elif signature_mode == "diagonal":
            values = self._heat_diagonal_values(spectral_data)
        else:
            values = self._full_matrix_values(G, domain, spectral_data)

        return {item: tuple(values[index].ravel()) for index, item in enumerate(spectral_data["items"])}

    def _truncate_signatures(self, signatures, domain):
        settings = self._domain_settings(domain)
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
            truncated[item] = tuple(int(value) for value in np.trunc(values * resolutions))

        return truncated

    def _signatures_without_bfs(self, G, node_signatures, edge_signatures):
        root_signatures = {}

        for vertex in G.nodes():
            node_group = (node_signatures[vertex],)
            edge_group = tuple(sorted(edge_signatures[self._edge_key(vertex,neighbor)] for neighbor in G.neighbors(vertex))) if edge_signatures else ()
            root_signatures[vertex] = ((node_group,edge_group),)

        return root_signatures

    def _signatures_with_bfs(self, G, node_signatures, edge_signatures):
        root_signatures = {}

        for root in G.nodes():
            distances = nx.single_source_shortest_path_length(G,root,cutoff=self.bfs_depth + 1)
            node_layers = [[] for _ in range(self.bfs_depth + 1)]
            within_layer_edges = [[] for _ in range(self.bfs_depth + 1)]
            cross_layer_edges = [[] for _ in range(self.bfs_depth + 1)]

            for vertex,depth in distances.items():
                if depth <= self.bfs_depth:
                    node_layers[depth].append(node_signatures[vertex])

            if edge_signatures:
                for u,v in G.edges():
                    if u not in distances or v not in distances:
                        continue

                    depth_u = distances[u]
                    depth_v = distances[v]
                    edge_signature = edge_signatures[self._edge_key(u,v)]

                    if depth_u == depth_v and depth_u <= self.bfs_depth:
                        within_layer_edges[depth_u].append(edge_signature)
                    elif abs(depth_u - depth_v) == 1:
                        lower_depth = min(depth_u,depth_v)

                        if lower_depth <= self.bfs_depth:
                            cross_layer_edges[lower_depth].append(edge_signature)

            root_signatures[root] = tuple(
                (
                    tuple(sorted(node_layers[depth])),
                    tuple(sorted(within_layer_edges[depth])),
                    tuple(sorted(cross_layer_edges[depth])),
                )
                for depth in range(self.bfs_depth + 1)
            )

        return root_signatures

    def compute_vertex_signatures(self,G):
        node_signatures = self._compute_signatures(G, "node")
        edge_signatures = self._compute_signatures(G, "edge") if self.include_edges else {}

        if self.comparison_mode == "fixed_resolution":
            node_signatures = self._truncate_signatures(node_signatures, "node")

            if edge_signatures:
                edge_signatures = self._truncate_signatures(edge_signatures, "edge")

        if self.bfs_depth is None:
            return self._signatures_without_bfs(G, node_signatures, edge_signatures)

        return self._signatures_with_bfs(G, node_signatures, edge_signatures)

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

    def _vertex_signatures_match(self, vertex_signature_v, vertex_signature_u):

        def collections_match(signatures_v, signatures_u):
            if len(signatures_v) != len(signatures_u):
                return False

            for item_signature_v,item_signature_u in zip(signatures_v,signatures_u):
                distance = max(self._weird_norm(item_signature_v,item_signature_u),self._weird_norm(item_signature_u,item_signature_v))

                if distance >= self.weird_norm_threshold:
                    return False

            return True



        if self.bfs_depth is None:
            (node_signatures_v,incident_edge_signatures_v), = vertex_signature_v
            (node_signatures_u,incident_edge_signatures_u), = vertex_signature_u

            return collections_match(node_signatures_v,node_signatures_u) and collections_match(incident_edge_signatures_v,incident_edge_signatures_u)

        for bfs_level_v,bfs_level_u in zip(vertex_signature_v,vertex_signature_u):
            node_signatures_v,within_level_edge_signatures_v,next_level_edge_signatures_v = bfs_level_v
            node_signatures_u,within_level_edge_signatures_u,next_level_edge_signatures_u = bfs_level_u

            if not collections_match(node_signatures_v,node_signatures_u):
                return False
            if not collections_match(within_level_edge_signatures_v,within_level_edge_signatures_u):
                return False
            if not collections_match(next_level_edge_signatures_v,next_level_edge_signatures_u):
                return False

        return True

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

    def compute_shared_vertex_ids(self, G, H):
        if self.comparison_mode != "fixed_resolution":
            raise ValueError("Shared IDs currently require fixed_resolution.")

        next_id = 0

        # Node signatures
        G_node_signatures = self._compute_signatures(G, "node")
        H_node_signatures = self._compute_signatures(H, "node")

        G_node_signatures = self._truncate_signatures(G_node_signatures, "node")
        H_node_signatures = self._truncate_signatures(H_node_signatures, "node")

        G_node_ids, H_node_ids, next_id = self.assign_shared_ids(
            G_node_signatures, H_node_signatures, next_id
        )

        del G_node_signatures, H_node_signatures

        # Edge signatures
        G_edge_ids = {}
        H_edge_ids = {}

        if self.include_edges:
            G_edge_signatures = self._compute_signatures(G, "edge")
            H_edge_signatures = self._compute_signatures(H, "edge")

            G_edge_signatures = self._truncate_signatures(G_edge_signatures, "edge")
            H_edge_signatures = self._truncate_signatures(H_edge_signatures, "edge")

            G_edge_ids, H_edge_ids, next_id = self.assign_shared_ids(
                G_edge_signatures, H_edge_signatures, next_id
            )

            del G_edge_signatures, H_edge_signatures

        if self.bfs_depth is None:
            G_root_signatures = self._signatures_without_bfs(G, G_node_ids, G_edge_ids)
            H_root_signatures = self._signatures_without_bfs(H, H_node_ids, H_edge_ids)
        else:
            G_root_signatures = self._signatures_with_bfs(G, G_node_ids, G_edge_ids)
            H_root_signatures = self._signatures_with_bfs(H, H_node_ids, H_edge_ids)

        del G_node_ids, H_node_ids, G_edge_ids, H_edge_ids

        # Give each complete BFS signature an ID too
        G_vertex_ids, H_vertex_ids, next_id = self.assign_shared_ids(
            G_root_signatures, H_root_signatures, next_id
        )

        del G_root_signatures, H_root_signatures

        return G_vertex_ids, H_vertex_ids

    def compute_equivalence_classes(self, root_signatures) -> EquivalenceClasses:

        if self.comparison_mode == "fixed_resolution":
            equivalence_classes = defaultdict(list)

            for vertex, signature in tqdm(root_signatures.items(), total=len(root_signatures),
                                          desc="Grouping signatures", unit="vertex", dynamic_ncols=True):
                equivalence_classes[signature].append(vertex)

            return equivalence_classes


        similarity_graph = nx.Graph()
        similarity_graph.add_nodes_from(root_signatures.keys())

        number_of_vertices = len(root_signatures)
        number_of_pairs = number_of_vertices * (number_of_vertices - 1) // 2
        vertex_pairs = tqdm(combinations(root_signatures.keys(), 2), total=number_of_pairs,
                            desc="Weird-norm comparisons", unit="pair", dynamic_ncols=True)

        for vertex_v, vertex_u in vertex_pairs:
            if self._vertex_signatures_match(root_signatures[vertex_v], root_signatures[vertex_u]):
                similarity_graph.add_edge(vertex_v, vertex_u)

        equivalence_classes = defaultdict(list)

        for component_index, component in enumerate(nx.connected_components(similarity_graph)):
            equivalence_classes[component_index] = sorted(component)

        return equivalence_classes
