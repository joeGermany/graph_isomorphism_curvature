from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Hashable

import networkx as nx


EquivalenceClasses = defaultdict[Hashable, list[int]]


class VertexEquivalenceInterface(ABC):

    @abstractmethod
    def compute_vertex_signatures(self,G:nx.Graph):
        pass

    @abstractmethod
    def compute_equivalence_classes(self,root_signatures)->EquivalenceClasses:
        pass