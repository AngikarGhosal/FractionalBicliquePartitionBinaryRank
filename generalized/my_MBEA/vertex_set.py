from typing import List
from .vertex import Vertex

class VertexSet(Vertex):
    def __init__(self, nodes_in: List[Vertex] = None):
        super().__init__()
        self.set_v: List[Vertex] = nodes_in.copy() if nodes_in else []

    def get_set_v(self) -> List[Vertex]:
        return self.set_v

    def get_size(self) -> int:
        return len(self.set_v)

    def get_vertex(self, i: int) -> Vertex:
        return self.set_v[i]

    def add_vertex(self, v: Vertex):
        if v not in self.set_v:
            self.set_v.append(v)

    def remove_vertex(self, v: Vertex):
        self.set_v.remove(v)

    def sort_by_num_of_neighbours(self) -> List[Vertex]:
        self.set_v.sort(key=lambda x: x.get_neighbours_size())
        return self.set_v

    def is_equal(self, other: 'VertexSet') -> bool:
        return self.set_v == other.set_v

    def is_set_empty(self) -> bool:
        return len(self.set_v) == 0

    def to_string_vertex_set(self) -> str:
        return ' '.join(str(v.get_label()) for v in self.set_v)