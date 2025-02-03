from typing import List, Optional, Tuple
from .vertex import Vertex
from .vertex_set import VertexSet
from .biclique import Biclique
from .bipartite_graph import BipartiteGraph

class BicliqueFinder(Biclique):
    def __init__(self, in_graph: BipartiteGraph):
        super().__init__()
        self.graph = in_graph
        self.init_l = VertexSet(in_graph.get_left_nodes())
        self.init_p = VertexSet(in_graph.get_right_nodes())
        self.init_r = VertexSet()
        self.init_q = VertexSet()
        self.maximal_bicliques: List[Biclique] = []
        self.found_all: bool = False
        self.vertex_set = set()  # Equivalent to Java's HashSet<Pair>

    def find_maximal_bicliques(self, alg_type: str):
        if alg_type == "standard":
            self._biclique_find(self.init_l, self.init_r, self.init_p, self.init_q)
            self.found_all = True

        elif alg_type == "MBC":
            self.init_p.sort_by_num_of_neighbours()
            self._biclique_find_im_p(self.init_l, self.init_r, self.init_p, self.init_q)
            self.found_all = True
            self._find_minimum_biclique_cover()

    def _biclique_find(self, in_l: VertexSet, in_r: VertexSet, in_p: VertexSet, in_q: VertexSet):
        # Create copies of input sets to avoid modifying originals
        l = VertexSet(in_l.get_set_v())
        r = VertexSet(in_r.get_set_v())
        p = VertexSet(in_p.get_set_v())
        q = VertexSet(in_q.get_set_v())

        while not p.is_set_empty():
            x = p.get_vertex(0)
            
            # Create R prime
            r_prime = VertexSet(r.get_set_v())
            r_prime.add_vertex(x)

            # Create L prime (vertices in L that are neighbours of x)
            l_prime = VertexSet([u for u in l.get_set_v() if u.is_neighbour(x)])

            # Prepare P prime and Q prime
            p_prime = VertexSet()
            q_prime = VertexSet()

            is_max = True

            # Check Q vertices
            for v in q.get_set_v():
                num_l_prime_neighbours = v.number_of_neighbours_of_v_in_set(l_prime.get_set_v())
                
                if num_l_prime_neighbours == l_prime.get_size():
                    is_max = False
                    break
                elif num_l_prime_neighbours > 0:
                    q_prime.add_vertex(v)

            if is_max:
                # Process P vertices
                for v in p.get_set_v():
                    if v.is_equal(x):
                        continue

                    num_l_prime_neighbours = v.number_of_neighbours_of_v_in_set(l_prime.get_set_v())
                    
                    if num_l_prime_neighbours == l_prime.get_size():
                        r_prime.add_vertex(v)
                    elif num_l_prime_neighbours > 0:
                        p_prime.add_vertex(v)

                # Create and add biclique
                bcq = Biclique(l_prime.get_set_v(), r_prime.get_set_v())
                bcq.is_maximal = True
                print(bcq.to_string_biclique())
                self.maximal_bicliques.append(bcq)

                # Recursive call if P prime is not empty
                if not p_prime.is_set_empty():
                    self._biclique_find(l_prime, r_prime, p_prime, q_prime)

            # Move x from P to Q
            p.remove_vertex(x)
            q.add_vertex(x)

    def _biclique_find_im_p(self, in_l: VertexSet, in_r: VertexSet, in_p: VertexSet, in_q: VertexSet):
        # Create copies of input sets
        l = VertexSet(in_l.get_set_v())
        r = VertexSet(in_r.get_set_v())
        p = VertexSet(in_p.get_set_v())
        q = VertexSet(in_q.get_set_v())

        while not p.is_set_empty():
            x = p.get_vertex(0)
            
            # Create R prime and initialize sets
            r_prime = VertexSet(r.get_set_v())
            r_prime.add_vertex(x)

            l_prime = VertexSet()
            overline_l_prime = VertexSet(l.get_set_v())
            c = VertexSet()

            # Find L prime (neighbours of x in L)
            for u in l.get_set_v():
                if u.is_neighbour(x):
                    l_prime.add_vertex(u)
                    overline_l_prime.remove_vertex(u)

            c.add_vertex(x)

            # Prepare P prime and Q prime
            p_prime = VertexSet()
            q_prime = VertexSet()

            is_max = True

            # Check Q vertices
            for v in q.get_set_v():
                num_l_prime_neighbours = v.number_of_neighbours_of_v_in_set(l_prime.get_set_v())
                
                if num_l_prime_neighbours == l_prime.get_size():
                    is_max = False
                    break
                elif num_l_prime_neighbours > 0:
                    q_prime.add_vertex(v)

            if is_max:
                # Process P vertices
                for v in p.get_set_v():
                    if v.is_equal(x):
                        continue

                    num_l_prime_neighbours = v.number_of_neighbours_of_v_in_set(l_prime.get_set_v())
                    
                    if num_l_prime_neighbours == l_prime.get_size():
                        r_prime.add_vertex(v)
                        
                        # Check overline L prime neighbours
                        num_overline_l_prime_neighbours = v.number_of_neighbours_of_v_in_set(overline_l_prime.get_set_v())
                        if num_overline_l_prime_neighbours == 0:
                            c.add_vertex(v)
                    elif num_l_prime_neighbours > 0:
                        p_prime.add_vertex(v)

                # Check for duplicate bicliques
                is_present = 0
                bcq = Biclique(l_prime.get_set_v(), r_prime.get_set_v())
                bcq.is_maximal = True

                for v1 in bcq.get_left_nodes():
                    for v2 in bcq.get_right_nodes():
                        vertex_pair = (v1.get_label(), v2.get_label())
                        if vertex_pair in self.vertex_set:
                            is_present += 1
                        self.vertex_set.add(vertex_pair)

                # Add biclique if not a duplicate
                if is_present != (len(bcq.get_left_nodes()) * len(bcq.get_right_nodes())):
                    self.maximal_bicliques.append(bcq)

                # Recursive call
                if not p_prime.is_set_empty():
                    self._biclique_find_im_p(l_prime, r_prime, p_prime, q_prime)

            # Move vertices from C to Q and remove from P
            for v in c.get_set_v():
                q.add_vertex(v)
                p.remove_vertex(v)

    def _find_minimum_biclique_cover(self):
        # Get left nodes and sort by number of neighbours
        left_node_list = VertexSet(self.graph.get_left_nodes())
        sorted_left_nodes = left_node_list.sort_by_num_of_neighbours()

        for v in sorted_left_nodes:
            for b in self.maximal_bicliques:
                # Check if vertex is in the biclique's left nodes
                if v in b.get_left_nodes():
                    present = 0
                    b_right = b.get_right_nodes()
                    
                    # Count number of neighbours in the biclique's right nodes
                    for right in v.get_neighbours():
                        if right in b_right:
                            present += 1

                    # If all neighbours are covered
                    if present == v.get_neighbours_size():
                        print(b.to_string_biclique())
                        
                        # Remove edges between left and right nodes
                        for left in b.get_left_nodes():
                            for right in b.get_right_nodes():
                                left.remove_neighbour(right)

                        # Check if all vertices have no neighbours
                        abs_count = sum(1 for node in sorted_left_nodes if node.get_neighbours_size() == 0)
                        if abs_count == len(sorted_left_nodes):
                            return

                        break

    def get_maximal_bicliques(self) -> Optional[List[Biclique]]:
        if self.found_all:
            return self.maximal_bicliques
        print("Not found yet")
        return None

    def get_lrpq_init(self) -> str:
        res = ' '.join(str(v.get_label()) for v in self.init_l.get_set_v()) + '\n'
        res += ' '.join(str(v.get_label()) for v in self.init_r.get_set_v()) + '\n'
        res += ' '.join(str(v.get_label()) for v in self.init_p.get_set_v()) + '\n'
        res += ' '.join(str(v.get_label()) for v in self.init_q.get_set_v()) + '\n'
        return res

    def get_num_bicliques(self) -> int:
        return len(self.maximal_bicliques) if self.found_all else 0

    def to_string_biclique_f(self) -> Optional[str]:
        if self.found_all:
            return '\n'.join(b.to_string_biclique() for b in self.maximal_bicliques)
        return None