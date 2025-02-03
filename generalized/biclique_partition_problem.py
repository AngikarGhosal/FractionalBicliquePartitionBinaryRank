import argparse
import numpy as np
import os
import subprocess
from itertools import chain, combinations, product
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import scipy as sp
import scipy.optimize as sop
import argparse
import multiprocessing
from functools import partial


def load_matrix(file_path):
    if file_path.endswith('.npz'):
        npz_file = np.load(file_path)
        return npz_file[npz_file.files[0]]
    else:
        with open(file_path, 'r') as file:
            content = file.read().strip()

        content = content.replace('[', '').replace(']', '').strip()
        
        lines = content.split('\n')
        matrix = []
        for line in lines:
            line = line.replace(',', ' ')
            matrix.append([int(x) for x in line.split()])

        return np.array(matrix)

def save_matrix_to_file(matrix, file_path):
    with open(file_path, 'w') as f:
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')

def run_external_script(input_file, output_file):
    try:
        # Create the command string
        command = f"python3 use_mbea.py {input_file} standard"
        
        # Open output file
        with open(output_file, 'w') as output_f:
            # Run the command and capture output
            subprocess.run(command, shell=True, stdout=output_f, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the script: {e}")


def parse_biclique_line(line):
    """Parse a biclique line into two lists of integers."""
    left_part, right_part = line.split('<->')
    left_nodes = list(map(int, left_part.split()))
    right_nodes = list(map(int, right_part.split()))
    return left_nodes, right_nodes

def generate_subbicliques(biclique_line):
    """Generate all sub-bicliques from a biclique line."""
    left_nodes, right_nodes = parse_biclique_line(biclique_line)

    # Generate all non-empty subsets of left_nodes and right_nodes
    left_subsets = list(chain.from_iterable(combinations(left_nodes, r) for r in range(1, len(left_nodes) + 1)))
    right_subsets = list(chain.from_iterable(combinations(right_nodes, r) for r in range(1, len(right_nodes) + 1)))

    # Create all possible sub-bicliques
    sub_bicliques = set()
    for left_subset in left_subsets:
        for right_subset in right_subsets:
            if len(left_subset)>1 or len(right_subset)>1:
                sub_biclique = (tuple(sorted(left_subset)), tuple(sorted(right_subset)))
                sub_bicliques.add(sub_biclique)

    return sub_bicliques

def read_maximal_bicliques(file_path):
    """Read maximal bicliques from a file."""
    with open(file_path, 'r') as file:
        return file.read().strip().split('\n')

def write_all_subbicliques_to_file(sub_bicliques, file_path):
    """Write all sub-bicliques to the specified file."""
    with open(file_path, 'w') as file:
        for left_subset, right_subset in sorted(sub_bicliques):
            left_part = ' '.join(map(str, left_subset))
            right_part = ' '.join(map(str, right_subset))
            file.write(f"{left_part} <-> {right_part}\n")

def generate_and_save_subbicliques(idx):
    """Generate all sub-bicliques for each maximal biclique and save them."""
    v1_file_path = f"MaxBicliques/M{idx}_FirstV1.txt"
    v0_file_path = f"MaxBicliques/M{idx}_FirstV0.txt"
    
    maximal_bicliques = read_maximal_bicliques(v1_file_path)
    all_subbicliques = set()

    for biclique_line in maximal_bicliques:
        subbicliques = generate_subbicliques(biclique_line)
        all_subbicliques.update(subbicliques)

    write_all_subbicliques_to_file(all_subbicliques, v0_file_path)

def create_adjacency_matrix(num_rows, num_cols, left_nodes, right_nodes):
    """Create an adjacency matrix from biclique nodes."""
    matrix = np.zeros((num_rows, num_cols), dtype=int)
    for left_node in left_nodes:
        for right_node in right_nodes:
            matrix[left_node - 1, right_node - num_rows - 1] = 1
    return matrix



def get_biclique_from_list_of_edges(list_of_edges):
    first_set = set()
    second_set = set()
    for x in list_of_edges:
        first_set.add(int(x[0]))
        second_set.add(int(x[1]))
    first_list = sorted(list(first_set))
    second_list = sorted(list(second_set))
    first_list = [str(i) for i in first_list]
    second_list = [str(i) for i in second_list]
    V = ' '.join(first_list) + ' <-> ' + ' '.join(second_list)
    return V


def process_biclique(biclique_set, edges, y, threshold, number_of_nodes):
    print("currently at", biclique_set)
    V = []
    Biclique_Edges = []
    for i in biclique_set[0]:
        for j in biclique_set[1]:
            index = edges.index((i,j))
            V.append(y.X[index])
            Biclique_Edges.append((i,j))
    V = np.array(V)

    Length = len(V)
    new_model = gp.Model("nextbiclique")
    new_model.setParam('OutputFlag', 0)  # Suppress Gurobi output
    Bool_Var = new_model.addMVar(shape=Length, vtype=GRB.BINARY, name="Bool_var")
    Node_Var = new_model.addMVar(shape=number_of_nodes, vtype=GRB.BINARY, name="Node_var")
    
    # The setup is same for all the max bicliques.
    new_model.setObjective(V.T@Bool_Var, GRB.MAXIMIZE)

    for i in range(Length):
        a = int(Biclique_Edges[i][0])
        b = int(Biclique_Edges[i][1])
        new_model.addConstr(Node_Var[a-1] >= Bool_Var[i])
        new_model.addConstr(Node_Var[b-1] >= Bool_Var[i])
        new_model.addConstr(Node_Var[a-1] + Node_Var[b-1] <= 1 + Bool_Var[i])

    new_model.optimize()

    print(f"Maximal Biclique: {biclique_set}")
    print(f"Optimal value: {new_model.ObjVal}")

    list_to_return = []
    for s in range(new_model.SolCount):
        new_model.params.SolutionNumber = s
        if new_model.PoolObjVal > threshold:
            one_zero_vector = Bool_Var.Xn

            flag_all_one = all(e == 1 for e in one_zero_vector)
            new_biclique_edges = []
            if not flag_all_one:
                new_biclique_edges = [Biclique_Edges[i] for i, e in enumerate(one_zero_vector) if e == 1]
            S = get_biclique_from_list_of_edges(new_biclique_edges)
            if S != ' <-> ':
                list_to_return.append(S + '\n')

    return new_model.ObjVal, list_to_return  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', type=str, required=True, help='Path to the text file containing the list of matrix files')
    parser.add_argument('--startiter', type=int, help='an integer to specify starting interation', default=1)
    parser.add_argument('--enditer', type=int, help='an integer to specify ending interation', default=10)
    parser.add_argument('--storefolder', type=str, help='path to the folder where bicliques are stored', default='Bicliques')
    print("BEGINNING")
    args = parser.parse_args()
    
    storefolder = args.storefolder
    startiter = args.startiter
    enditer = args.enditer
    
    matrices = []
    with open(args.filelist, 'r') as file:
        file_paths = file.read().splitlines()

    # Create directories if they do not exist
    matrices_folder = "Matrices"
    max_bicliques_folder = "MaxBicliques"
    bicliques_folder = "Bicliques"
    os.makedirs(matrices_folder, exist_ok=True)
    os.makedirs(max_bicliques_folder, exist_ok=True)
    os.makedirs(bicliques_folder, exist_ok=True)

    for idx, file_path in enumerate(file_paths, start=1):
        matrix = load_matrix(file_path)
        matrices.append(matrix)

        # Save matrix to text file
        matrix_file_name = f"M{idx}.txt"
        matrix_file_path = os.path.join(matrices_folder, matrix_file_name)
        save_matrix_to_file(matrix, matrix_file_path)

        # Run the external script and store output
        output_file_name = f"M{idx}_FirstV1.txt"
        output_file_path = os.path.join(max_bicliques_folder, output_file_name)
        
        run_external_script(matrix_file_path, output_file_path)
    
    # Generate all sub-bicliques and save them
    for idx, file_path in enumerate(file_paths, start=1):
        generate_and_save_subbicliques(idx)
    
    # Compute the Kronecker product of all matrices
    kron_product = matrices[0]
    for matrix in matrices[1:]:
        kron_product = np.kron(kron_product, matrix)

    # Save kron_product to Bicliques/M.txt
    kron_file_path = os.path.join(bicliques_folder, "M.txt")
    save_matrix_to_file(kron_product, kron_file_path)

    # Run the external script on the full Kronecker matrix
    first_v0_file_path = os.path.join(bicliques_folder, "FirstV0.txt")
    v0_file_path = os.path.join(bicliques_folder, "V0.txt")
    run_external_script(kron_file_path, first_v0_file_path)

    # Generate Kronecker products of maximal bicliques
    num_matrices = len(file_paths)
    maximal_bicliques_list = []
    matrix_shapes = []

    for idx in range(1, num_matrices + 1):
        v1_file_path = f"MaxBicliques/M{idx}_FirstV1.txt"
        maximal_bicliques = read_maximal_bicliques(v1_file_path)
        maximal_bicliques_list.append(maximal_bicliques)
        
        matrix_file_path = f"Matrices/M{idx}.txt"
        matrix = load_matrix(matrix_file_path)
        matrix_shapes.append(matrix.shape)

    full_bicliques = set()
    for biclique_combo in product(*maximal_bicliques_list):
        adjacency_product = None
        for idx, biclique_line in enumerate(biclique_combo):
            left_nodes, right_nodes = parse_biclique_line(biclique_line)
            num_rows, num_cols = matrix_shapes[idx]
            adjacency_matrix = create_adjacency_matrix(num_rows, num_cols, left_nodes, right_nodes)

            if adjacency_product is None:
                adjacency_product = adjacency_matrix
            else:
                adjacency_product = np.kron(adjacency_product, adjacency_matrix)
        # Extract and format the biclique from adjacency_product
        left_indices, right_indices = np.nonzero(adjacency_product)
        left_indices_adjusted = left_indices + 1
        right_indices_adjusted = right_indices + adjacency_product.shape[0] + 1  # Adjust right indices
        # Remove duplicates by converting to sets
        left_indices_set = sorted(set(left_indices_adjusted))
        right_indices_set = sorted(set(right_indices_adjusted))

        # Form the biclique string
        full_biclique_str = f"{' '.join(map(str, left_indices_set))} <-> {' '.join(map(str, right_indices_set))}"
        full_bicliques.add(full_biclique_str)

    # Write unique full bicliques to file
    first_v1_file_path = os.path.join(bicliques_folder, "FirstV1.txt")
    v1_file_path = os.path.join(bicliques_folder, "V1.txt")

    with open(first_v1_file_path, 'w') as f:
        for biclique in sorted(full_bicliques):
            f.write(biclique + '\n')

    with open(first_v0_file_path, 'r') as file1:
        lines = file1.readlines()
    with open(v0_file_path, 'w') as file2:
        for line in lines:
            modified_line = line.rstrip('\n') + ':0\n'
            file2.write(modified_line)

    with open(first_v1_file_path, 'r') as file1:
        lines = file1.readlines()
    with open(v1_file_path, 'w') as file2:
        for line in lines:
            modified_line = line.rstrip('\n') + ':0\n'
            file2.write(modified_line)
    edges = []
    for i in range(kron_product.shape[0]):
        for j in range(kron_product.shape[1]):
            if kron_product[i, j] != 0:
                edges.append((i+1, j+1+(kron_product.shape[0])))

    adjacency_sets = {}
    for u, v in edges:
        if u not in adjacency_sets:
            adjacency_sets[u] = set()
        if v not in adjacency_sets:
            adjacency_sets[v] = set()
        adjacency_sets[u].add(v)
        adjacency_sets[v].add(u)

    biclique_data = {}

    ###########
    ##Rest of the Code is Same From Here
    ###########

    # Read the initial biclique data
    initial_file = f"{storefolder}/V{startiter}.txt"
    with open(initial_file, 'r') as file:
        for line in file:
            biclique, count = line.strip().split(':')
            biclique_data[biclique] = int(count)

    Name_of_File_to_open = f"{storefolder}/V0.txt"
    with open(Name_of_File_to_open, 'r') as filename:
        first_max_bicliques = [line.split(':')[0].strip() for line in filename]
    first_max_bicliques_as_sets = []
    for key in first_max_bicliques:
        A, B = key.split('<->')
        Asplit = [int(a) for a in A.strip().split()]
        Bsplit = [int(b) for b in B.strip().split()]
        first_max_bicliques_as_sets.append((set(Asplit), set(Bsplit)))

    best_lower_bound = -float('inf')
    corresponding_value = None

    for VAL in range(startiter, enditer+1):
        print("CURRENT VAL IS ", VAL)
        print("\n\n\n\n")

        max_bicliques_vals = []
        max_bicliques = []
        for biclique, count in biclique_data.items():
            max_bicliques_vals.append(count)
            max_bicliques.append(biclique)

        max_bicliques_as_sets = []
        for key in max_bicliques:
            A, B = key.split('<->')
            Asplit = [int(a) for a in A.strip().split()]
            Bsplit = [int(b) for b in B.strip().split()]
            max_bicliques_as_sets.append((set(Asplit), set(Bsplit)))

        adjacency_matrix=np.array([[0 for i in range((len(edges))+1)] for j in range(len(max_bicliques_as_sets)+1)])
        for i, biclique in enumerate(max_bicliques_as_sets):
            for j, edge in enumerate(edges):
                if edge[0] in biclique[0] and edge[1] in biclique[1]:
                    adjacency_matrix[i+1][j+1] = 1

        A = adjacency_matrix[1:, 1:].T.copy()
        m, n = A.shape
        print(m, n)


        ones_m = np.ones((m,), dtype=int)

        model = gp.Model("biclique")
        y = model.addMVar(shape=m, lb=-float('inf')*ones_m, name="y")
        model.setObjective(ones_m @ y, GRB.MAXIMIZE)
        threshold = 1.000001
        number_of_nodes = kron_product.shape[0] + kron_product.shape[1]


        ones_n = np.ones((n,), dtype=int)
        model.addConstr(A.T @ y <= ones_n)
        model.setParam("Crossover", 0)
        model.optimize()

        print(f"Optimal value: {model.ObjVal}")
        print(f"Optimal solution: y = {y.X}")

        dual_answer = A.T @ y.X
        for index in range(len(dual_answer)):
            element = dual_answer[index]
            print(index, element, "HERE")
            if element < 0.99999:
                max_bicliques_vals[index] += 1
            print(A.T @ y.X)

        process_biclique_partial = partial(process_biclique,  
                                            edges=edges,  
                                            y=y.X,  
                                            threshold=threshold,  
                                            number_of_nodes=number_of_nodes)

        with multiprocessing.Pool() as pool:
            results = pool.map(process_biclique_partial, first_max_bicliques_as_sets)

        max_obj_value = max(res[0] for res in results)
        print(f"Max objective value from bicliques: {max_obj_value}")

        scaled_obj_val = model.ObjVal / max_obj_value
        print(f"Scaled dual objective value: {scaled_obj_val}")

        # Update best lower bound if the current scaled objective value is better
        if scaled_obj_val > best_lower_bound:
            best_lower_bound = scaled_obj_val
            corresponding_value = VAL

        bicliquestrings=[res[1] for res in results]
        New_Biclique_String_List = list(set(chain(*bicliquestrings)))
        bicliques_added = len(New_Biclique_String_List)

        print(New_Biclique_String_List)

        # Update biclique_data
        for biclique in New_Biclique_String_List:
            biclique = biclique.strip()
            if biclique not in biclique_data:
                biclique_data[biclique] = 0
            elif biclique in max_bicliques:
                position = max_bicliques.index(biclique)
                if max_bicliques_vals[position] < 3:
                    biclique_data[biclique] = max_bicliques_vals[position]
                else:
                    print(f"ELIMINATING {biclique} at {VAL}")
                    biclique_data.pop(biclique, None)
                    
        intermediate_file = f"{storefolder}/V{VAL+1}.txt"
        with open(intermediate_file, 'w') as file:
            for biclique, count in biclique_data.items():
                file.write(f"{biclique}:{count}\n")

    # After all iterations, write the final results
    final_file = f"{storefolder}/V{enditer+1}.txt"
    with open(final_file, 'w') as file:
        for biclique, count in biclique_data.items():
            file.write(f"{biclique}:{count}\n")

    print(f"Final biclique data written to {final_file}")

if __name__ == '__main__':
    main()