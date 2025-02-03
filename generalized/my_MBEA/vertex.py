from typing import List, Optional

class Vertex:
    def __init__(self, label: int = None):
        self.label: int = label
        self.neighbours: List[Vertex] = []

    def get_neighbours(self) -> List['Vertex']:
        return self.neighbours

    def get_label(self) -> int:
        return self.label

    def _add_neighbour(self, v: 'Vertex'):
        if v in self.neighbours:
            raise RuntimeError(f"Vertex::add_neighbour - vertex is already a neighbour.")
        self.neighbours.append(v)

    def remove_neighbour(self, v: 'Vertex'):
        if v in self.neighbours:
            self.neighbours.remove(v)

    @staticmethod
    def add_edge(v1: 'Vertex', v2: 'Vertex'):
        try:
            v1._add_neighbour(v2)
        except RuntimeError:
            raise RuntimeError(f"Vertex::add_edge - {v2.label} is already a neighbour of {v1.label}")

        try:
            v2._add_neighbour(v1)
        except RuntimeError:
            raise RuntimeError(f"Vertex::add_edge - {v1.label} is already a neighbour of {v2.label}")

    def remove_edge(self, v1: 'Vertex', v2: 'Vertex'):
        v1.remove_neighbour(v2)
        v2.remove_neighbour(v1)

    def get_neighbours_size(self) -> int:
        return len(self.neighbours)

    def number_of_neighbours_of_v_in_set(self, set_vertices: List['Vertex']) -> int:
        return sum(1 for vertex in set_vertices if self.is_neighbour(vertex))

    def is_neighbour(self, other_v: 'Vertex') -> bool:
        return other_v in self.neighbours

    def is_equal(self, other_v: 'Vertex') -> bool:
        return self.label == other_v.label

    def is_member(self, vertex_set: List['Vertex']) -> bool:
        return any(self.label == v.label for v in vertex_set)

    def __lt__(self, other: 'Vertex') -> bool:
        return self.get_neighbours_size() < other.get_neighbours_size()