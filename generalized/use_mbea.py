import sys
from typing import List
from my_MBEA.bipartite_graph import BipartiteGraph
from my_MBEA.biclique_finder import BicliqueFinder

def read_matrix_from_file(filename: str) -> List[List[int]]:
    """
    Read adjacency matrix from a text file.
    
    Expected file format:
    - Each line represents a row of the matrix
    - Values are space or tab-separated integers (0 or 1)
    
    Example file content:
    0 1 1 0
    1 0 0 1
    1 0 1 0
    0 1 0 1
    """
    adj_matrix: List[List[int]] = []
    
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Split line by whitespace and convert to integers
                row = [int(x) for x in line.strip().split()]
                adj_matrix.append(row)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except ValueError:
        print("Error: Invalid matrix format. Ensure all values are integers (0 or 1).")
        sys.exit(1)
    
    return adj_matrix

def main():
    # Check if correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <matrix_file> <algorithm_type>")
        print("Algorithm types: 'standard' or 'MBC'")
        sys.exit(1)

    # Get input file and algorithm type from command line
    matrix_file = sys.argv[1]
    algorithm_type = sys.argv[2]

    # Read adjacency matrix from file
    adj_matrix = read_matrix_from_file(matrix_file)

    # Create BipartiteGraph
    graph = BipartiteGraph(adj_matrix)

    # Create BicliqueFinder
    biclique_finder = BicliqueFinder(graph)

    # Find maximal bicliques
    biclique_finder.find_maximal_bicliques(algorithm_type)

    # Print results
    # print("\nMaximal Bicliques:")
    # biclique_finder_results = biclique_finder.to_string_biclique_f()
    # if biclique_finder_results:
    #     print(biclique_finder_results)
    # else:
    #     print("No bicliques found.")

if __name__ == "__main__":
    main()