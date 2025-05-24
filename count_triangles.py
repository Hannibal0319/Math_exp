#count trinagles in graph based on adjacency matrix
import numpy as np
import networkx as nx
import torch


def count_triangles(adj_matrix):
    """
    Counts the number of triangles in a graph represented by its adjacency matrix.

    Parameters:
        adj_matrix (numpy.ndarray): The adjacency matrix of the graph.

    Returns:
        int: The number of triangles in the graph.
    """
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)

    # Count the number of triangles, but if multiple node in one triangle, count only once
    # This is done by summing the number of triangles for each node and dividing by 3
   
    triangle_count = nx.triangles(G)
    triangle_count = sum(nx.triangles(G).values()) // 3

    return triangle_count

def multiply(A, B, C,V):
    for i in range(V):
        for j in range(V):
            C[i][j] = 0
            for k in range(V):
                C[i][j] += A[i][k] * B[k][j]

# Utility function to calculate 
# trace of a matrix (sum of 
# diagonal elements) 
def getTrace(graph,V):
    trace = 0
    for i in range(V):
        trace += graph[i][i] 
    return trace

# Utility function for calculating 
# number of triangles in graph 
def triangleInGraph(graph,V):
    
    # To Store graph^2 
    aux2 = [[None] * V for i in range(V)]

    # To Store graph^3 
    aux3 = [[None] * V for i in range(V)]

    # Initialising aux 
    # matrices with 0
    for i in range(V):
        for j in range(V):
            aux2[i][j] = aux3[i][j] = 0

    # aux2 is graph^2 now printMatrix(aux2) 
    multiply(graph, graph, aux2,V) 

    # after this multiplication aux3 is 
    # graph^3 printMatrix(aux3) 
    multiply(graph, aux2, aux3,V) 

    trace = getTrace(aux3,V) 
    return trace // 6

if __name__ == "__main__":
    # Example adjacency matrix
    V=20
    adj_matrix = np.array([
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    # Count triangles
    triangle_count = triangleInGraph(adj_matrix,V)
    print(f"Number of triangles in the graph: {triangle_count}")
    