import umap
import anndata
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS, smacof

from .._utils import graph_alpha
from .._utils import graph_knn
from .._utils import estimate_cutoff_knn
from .._utils import rep_dgi
from .._utils import rep_gae

def spatial_graph(
    adata: anndata.AnnData,
    method="alpha shape", 
    alpha_n_layer=1, 
    cut=None, 
    estimate_cut=True, 
    knn_n_neighbors=10,
    draw = False
):
    pts = adata.obsm['spatial']
    if estimate_cut:
        cut = estimate_cutoff_knn(pts, k=knn_n_neighbors)
    else:
        cut = np.inf
    if method == "alpha shape":
        A = graph_alpha(pts, cut=cut, n_layer=alpha_n_layer, draw=draw)
    elif method == "knn":
        A = graph_knn(pts, cut=cut, k=knn_n_neighbors, draw=draw)
    
    adata.obsp['scanit-graph'] = A

# def spatial_deconvolution(
#     adata: anndata.AnnData,
#     method='laplacian'
# ):


def spatial_representation(
    adata: anndata.AnnData,
    n_h = 32,
    model = 'dgi',
    n_epoch = 1000,
    lr = 0.001,
    print_step = 500,
    torch_seed = None,
    python_seed = None,
    numpy_seed = None,
    device=None,
    data_slot = None,
    n_consensus = 1,
    projection = 'mds',
    n_comps_proj = 15,
    n_nb_proj = 15,
    extra_embeddings = None,
    extra_embedding_weights = None
):
    A = adata.obsp['scanit-graph']
    if not data_slot is None:
        X_processed = np.array( adata.obsm[data_slot] )
    else:
        X_processed = np.array( adata.X )
        
    if model == "dgi":
        if n_consensus == 1 and extra_embeddings is None:
            X_embed = rep_dgi(n_h, X_processed, A, 
                n_epoch=n_epoch, lr=lr, print_step=print_step, 
                torch_seed=torch_seed, python_seed=python_seed,
                numpy_seed=numpy_seed, device=device)
        else:
            X_embeds = []

            np.random.seed(torch_seed)
            torch_seeds = np.random.choice(10000, size=n_consensus, replace=False)
            np.random.seed(python_seed)
            python_seeds = np.random.choice(10000, size=n_consensus, replace=False)
            np.random.seed(numpy_seed)
            numpy_seeds = np.random.choice(10000, size=n_consensus, replace=False)
            
            for i in range(n_consensus):
                X_embed = rep_dgi(n_h, X_processed, A, 
                    n_epoch=n_epoch, lr=lr, print_step=print_step, 
                    torch_seed=torch_seeds[i], python_seed=python_seeds[i],
                    numpy_seed=numpy_seeds[i], device=device)
                X_embeds.append(X_embed)
            if not extra_embeddings is None:
                for extra_embedding in extra_embeddings:
                    X_embeds.append(extra_embedding)
            
            if extra_embedding_weights is None:
                embeds_weights = np.ones(len(X_embeds)) / float(len(X_embeds))
            else:
                embeds_weights = []
                for i in range(n_consensus):
                    embeds_weights.append( (1-np.sum(extra_embedding_weights))/float(n_consensus) )
                embeds_weights.extend(extra_embedding_weights)
                embeds_weights = np.array(embeds_weights, float)
    elif model == "gae":
        X_embed = rep_gae(n_h, X_processed, A, n_epoch=n_epoch)

    if n_consensus > 1 or not extra_embeddings is None:
        n_spot = X_embed.shape[0]
        W_consensus = np.zeros([n_spot, n_spot])
        for i in range(len(X_embeds)):
            W = distance_matrix(X_embeds[i], X_embeds[i])
            W_consensus += W * embeds_weights[i]
        if projection == 'mds':
            # X_embed,_ = smacof(W_consensus, n_components=n_comps_proj, n_jobs=-1)
            model = MDS(n_components=n_comps_proj, dissimilarity='precomputed', n_jobs=-1, random_state=python_seed)
            X_embed = model.fit_transform(W_consensus)
        elif projection == 'umap':
            model = umap.UMAP(n_components=n_comps_proj, metric='precomputed', n_neighbors=n_nb_proj)
            X_embed = model.fit_transform(W_consensus)

    adata.obsm['X_scanit'] = X_embed
    if n_consensus > 1:
        adata.obsp['D_scanit'] = W_consensus