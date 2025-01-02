import numpy as np
import gudhi
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

def graph_alpha(pts, n_layer = 1, cut = np.inf, draw = False):
    
    # Get a graph from alpha shape
    pts_list = pts.tolist()
    n_node = len(pts_list)
    alpha_complex = gudhi.AlphaComplex(points=pts_list)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=cut**2)
    skeleton = simplex_tree.get_skeleton(1)
    initial_graph = nx.Graph()
    initial_graph.add_nodes_from([i for i in range(n_node)])
    for s in skeleton:
        if len(s[0]) == 2:
            initial_graph.add_edge(s[0][0], s[0][1])
    # Extend the graph for the specified layers
    extended_graph = nx.Graph()
    extended_graph.add_nodes_from(initial_graph)
    extended_graph.add_edges_from(initial_graph.edges)
    if n_layer == 2:
        for i in range(n_node):
            for j in initial_graph.neighbors(i):
                for k in initial_graph.neighbors(j):
                    extended_graph.add_edge(i,k)
    elif n_layer == 3:
        for i in range(n_node):
            for j in initial_graph.neighbors(i):
                for k in initial_graph.neighbors(j):
                    for l in initial_graph.neighbors(k):
                        extended_graph.add_edge(i,l)
    if n_layer >= 4:
        print("Setting n_layer to greater than 3 may results in too large neighborhoods")    

    # Remove self edges
    for i in range(n_node):
        try:
            extended_graph.remove_edge(i,i)
        except:
            pass

    # Draw the graph
    if draw:
        pos_dict = {i:pts_list[i] for i in range(n_node)}
        nx.draw_networkx(extended_graph, pos=pos_dict, with_labels=False, node_size=1, node_color='dimgrey', edge_color='darkgrey')
        plt.axis('equal'); plt.axis('off'); plt.show()

    # Get the sparse adjacency matrix
    A = nx.to_scipy_sparse_matrix(extended_graph, format='csr')

    return A
