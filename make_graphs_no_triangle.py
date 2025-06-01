import random
import itertools
import numpy as np
from tqdm import tqdm

def upper_triangle_indices(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def count_triangles(graph, n):
    """Returns a dictionary mapping edge (i, j) to number of triangles it is part of."""
    triangle_counts = {}
    for i, j in upper_triangle_indices(n):
        if graph[i][j]:
            count = 0
            for k in range(n):
                if k != i and k != j:
                    if k < i:
                        a = graph[k][i]
                    else:
                        a = graph[i][k]
                    if k < j:
                        b = graph[k][j]
                    else:
                        b = graph[j][k]
                    if a and b:
                        count += 1
            triangle_counts[(i, j)] = count
    return triangle_counts

def has_triangle(graph, n):
    """Checks whether there's any triangle in the graph."""
    for i, j, k in itertools.combinations(range(n), 3):
        if (graph[i][j] if i < j else graph[j][i]) and \
           (graph[i][k] if i < k else graph[k][i]) and \
           (graph[j][k] if j < k else graph[k][j]):
            return True
    return False

def edge_forms_triangle(graph, n, u, v):
    """Check if adding edge (u, v) would form a triangle."""
    for w in range(n):
        if w == u or w == v:
            continue
        uw = graph[u][w] if u < w else graph[w][u]
        vw = graph[v][w] if v < w else graph[w][v]
        if uw and vw:
            return True
    return False

def generate_graph(n=20):
    # Start with a complete graph
    graph = np.zeros((n, n), dtype=int)
    for i, j in upper_triangle_indices(n):
        graph[i][j] = 1

    # Remove edges in triangles
    while has_triangle(graph, n):
        triangle_counts = count_triangles(graph, n)
        if not triangle_counts:
            break
        max_tri = max(triangle_counts.values())
        edges = [e for e, count in triangle_counts.items() if count == max_tri]
        edge_to_remove = random.choice(edges)
        i, j = edge_to_remove
        graph[i][j] = 0

    # Add edges without forming triangles
    possible_edges = [(i, j) for i, j in upper_triangle_indices(n) if graph[i][j] == 0]
    random.shuffle(possible_edges)
    for i, j in possible_edges:
        if not edge_forms_triangle(graph, n, i, j):
            graph[i][j] = 1

    return graph

def generate_graphs(num_graphs=20, n=20):
    graphs = []
    for _ in tqdm(range(num_graphs)):
        graph = generate_graph(n)
        graphs.append(graph)
    return graphs

def print_graph(graph):
    for i in range(len(graph)):
        row = ''.join(str(graph[i][j]) if j > i else ' ' for j in range(len(graph)))
        print(row)
    print("\n")

def save_graphs(graphs, filename="graphs.txt"):
    with open(filename, "w") as f:
        for graph in graphs:
            for i in range(len(graph)):
                row = ''.join(str(graph[i][j]) for j in range(len(graph)))
                f.write(row + " ")
            f.write("\n")

def load_graphs(filename="graphs.txt"):
    graphs = []
    with open(filename, "r") as f:
        for line in f:
            graph = [[]]
            idx=0
            for char in line.strip():
                if char == '1':
                    graph[idx].append(1)
                elif char == '0':
                    graph[idx].append(0)
                else:
                    idx+=1
                    graph.append([])
            graphs.append(np.array(graph).reshape(1,len(graph)* len(graph)))
    return graphs

def main():
    # Generate and print graphs
    num_graphs = 3000  # Number of graphs to generate
    N=20
    graphs = generate_graphs(num_graphs=num_graphs, n=N)
    for idx, g in enumerate(graphs):
        print(f"Graph {idx + 1}:")
        print_graph(g)
    
    # Save the graphs to a file
    
    save_graphs(graphs, filename= f"triangle_free_graphs/graphs_{N}_num{num_graphs}.txt")
    
    

if __name__ == "__main__":
    main()
