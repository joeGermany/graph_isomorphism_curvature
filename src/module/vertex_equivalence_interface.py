from abc import ABC,abstractmethod
from typing import Optional
import networkx as nx


class VertexEquivalenceInterface(ABC):

    @abstractmethod
    def compute_vertex_signatures(self,G:nx.Graph,H:Optional[nx.Graph]=None):
        pass