import sys
from typing import List
from .biclique_finder import BicliqueFinder
from .bipartite_graph import BipartiteGraph

def main():
    if len(sys.argv) < 3:
        print("Usage: python solver.py <input_file> <algorithm_type>")
        sys.exit(1)

    # Read adjacency matrix from file
    adj_matrix: List[List[int]] = []
    with open(sys.argv[1], 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip().split()]
            adj_matrix.append(row)

    # Create BicliqueFinder and find maximal bicliques
    biclique_finder = BicliqueFinder(BipartiteGraph(adj_matrix))
    biclique_finder.find_maximal_bicliques(sys.argv[2])

if __name__ == "__main__":
    main()