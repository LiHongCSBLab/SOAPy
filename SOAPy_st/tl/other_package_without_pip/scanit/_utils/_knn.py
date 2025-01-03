import numpy as np
from scipy import sparse
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import networkx as nx
import matplotlib.pyplot as plt

def graph_knn(pts, k = 10, cut = np.inf, draw = False):
    # Assymetric knn graph
    A_knn = kneighbors_graph(pts, n_neighbors=k, mode='connectivity')
    # Make it symetric
    A_knn_sym = ((A_knn + A_knn.T).astype(bool)).astype(int)
    # Apply the cutoff
    if not np.isinf(cut):
        A_rn = radius_neighbors_graph(pts, radius=cut, mode='connectivity')
        A_knn_sym = A_knn_sym.multiply(A_rn)
    # Plot the graph
    if draw:
        G = nx.from_scipy_sparse_matrix(A_knn_sym)
        pts_list = list(pts)
        n_node = len(pts_list)
        pos_dict = {i:pts_list[i] for i in range(n_node)}
        nx.draw_networkx(G, pos=pos_dict, with_labels=False, node_size=10, node_color='dimgrey', edge_color='darkgrey')
        plt.axis('equal'); plt.axis('off'); plt.show()

    return A_knn_sym