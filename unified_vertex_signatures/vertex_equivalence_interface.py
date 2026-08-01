from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Hashable

import networkx as nx


EquivalenceClasses = defaultdict[Hashable, list[int]]


class VertexEquivalenceInterface(ABC):
    @abstractmethod
    def compute_equivalence_classes(self, G: nx.Graph) -> EquivalenceClasses:

        pass