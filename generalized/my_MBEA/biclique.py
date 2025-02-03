from typing import List, Optional
from .vertex import Vertex
from .bipartite_graph import BipartiteGraph

class Biclique(BipartiteGraph):
    def __init__(self, left_v: Optional[List[Vertex]] = None, right_v: Optional[List[Vertex]] = None):
        super().__init__()
        self.is_maximal: bool = False
        
        if left_v and right_v:
            self.left_nodes = left_v
            self.right_nodes = right_v
            self.left_neighbours = []
            self.right_neighbours = []

            # Create edges between left and right nodes
            for left in self.left_nodes:
                for right in self.right_nodes:
                    try:
                        Vertex.add_edge(left, right)
                    except RuntimeError:
                        pass  # Suppress edge creation errors

            # Populate neighbours lists
            self.left_neighbours = [left.get_neighbours() for left in self.left_nodes]
            self.right_neighbours = [right.get_neighbours() for right in self.right_nodes]

    def to_string_biclique(self) -> str:
        left_nodes_str = ' '.join(str(node.get_label()) for node in self.left_nodes)
        right_nodes_str = ' '.join(str(node.get_label()) for node in self.right_nodes)
        return f"{left_nodes_str} <-> {right_nodes_str}"