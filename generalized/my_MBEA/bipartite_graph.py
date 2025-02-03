from typing import List, Any
from .vertex import Vertex

class BipartiteGraph:
    def __init__(self, inc_mat: List[List[int]] = None):
        self.incidence_matrix: List[List[int]] = inc_mat
        self.left_nodes: List[Vertex] = []
        self.right_nodes: List[Vertex] = []
        self.left_neighbours: List[List[Vertex]] = []
        self.right_neighbours: List[List[Vertex]] = []

        if inc_mat:
            self._initialize_graph()

    def _initialize_graph(self):
        self.check_input(self.incidence_matrix)

        left_start = 1
        right_start = len(self.incidence_matrix) + 1

        transposed = self.transpose(self.incidence_matrix)

        # Create left and right nodes
        self.left_nodes = [Vertex(left_start + i) for i in range(len(self.incidence_matrix))]
        self.right_nodes = [Vertex(right_start + i) for i in range(len(transposed))]

        # Add edges
        for i, row in enumerate(self.incidence_matrix):
            for j, val in enumerate(row):
                if val == 1:
                    left = self.left_nodes[i]
                    right = self.right_nodes[j]
                    try:
                        Vertex.add_edge(left, right)
                    except RuntimeError as e:
                        print(e)

        # Populate neighbours lists
        self.left_neighbours = [left.get_neighbours() for left in self.left_nodes]
        self.right_neighbours = [right.get_neighbours() for right in self.right_nodes]

    def check_input(self, inc_mat: List[List[int]]):
        row_size = len(inc_mat[0])
        for row in inc_mat:
            if len(row) != row_size:
                print("Each row should be of same length")
            for elem in row:
                if elem not in [0, 1]:
                    print("Should be 1/0")

    @staticmethod
    def transpose(mat: List[List[int]]) -> List[List[int]]:
        return [
            [row[i] for row in mat]
            for i in range(len(mat[0]))
        ]

    def get_incidence_matrix(self) -> List[List[int]]:
        return self.incidence_matrix

    def get_left_nodes(self) -> List[Vertex]:
        return self.left_nodes

    def get_right_nodes(self) -> List[Vertex]:
        return self.right_nodes

    def get_left_neighbours(self) -> List[List[Vertex]]:
        return self.left_neighbours

    def get_right_neighbours(self) -> List[List[Vertex]]:
        return self.right_neighbours

    def get_neighbourhood_string(self, neighbours: List[List[Vertex]]) -> str:
        return '\n'.join(' '.join(str(v.get_label()) for v in nv_list) for nv_list in neighbours)

    def print_neighbourhoods(self):
        print(self.get_neighbourhood_string(self.left_neighbours))
        print("\n")
        print(self.get_neighbourhood_string(self.right_neighbours))

    @staticmethod
    def print_list(ls: List[Vertex]):
        for v in ls:
            print(v.get_label())

    def print_graph(self):
        self.print_list(self.get_left_nodes())
        self.print_list(self.get_right_nodes())