import copy
import pandas as pd
import torch
import scanpy as sc
from typing import Optional, Union
from .utils import _set_R_environment
from typing import Optional, Literal
import anndata as ad


class _STAGATE2Domain(object):

    def __init__(self,
                 adata: sc.AnnData,
                 inplace: bool = True,
                 ):
        if inplace:
            self.adata = adata
        else:
            self.adata = copy.deepcopy(self.adata)

    def get_Spatial_domain(self,
                           rad_cutoff=None,
                           model: Literal['Radius', 'KNN'] = 'Radius',
                           n_epochs: int = 1500,
                           lr: float = 0.0001,
                           device: Optional[str] = None,
                           ):

        import STAGATE_pyG
        adata = self.adata

        STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff, model=model)
        # STAGATE_pyG.Stats_Spatial_Net(adata)

        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        STAGATE_pyG.train_STAGATE(adata,
                                  n_epochs=n_epochs,
                                  lr=lr,
                                  device=torch.device(device))

        return adata

    def mclust_R(self,
                 num_cluster: int,
                 R_HOME: Optional[str] = None,
                 R_USER: Optional[str] = None,
                 used_obsm: str = 'STAGATE',
                 key_added: str = 'cluster',
                 random_seed: int = 2020,
                 ):

        import STAGATE_pyG

        adata = self.adata
        _set_R_environment(R_HOME=R_HOME, R_USER=R_USER)
        sc.pp.neighbors(adata, use_rep='STAGATE')
        sc.tl.umap(adata)

        # clust
        adata = STAGATE_pyG.mclust_R(adata,
                                     used_obsm=used_obsm,
                                     num_cluster=num_cluster,
                                     random_seed=random_seed)

        adata.obs.rename(columns={'mclust': key_added})
        return adata.obs
        # obs_df = adata.obs.dropna()

    def louvain(self,
                resolution: float = 0.5,
                key_added: str = 'cluster',
                ):

        adata = self.adata

        sc.pp.neighbors(adata, use_rep='STAGATE')
        sc.tl.umap(adata)
        # louvain
        sc.tl.louvain(adata, resolution=resolution, key_added=key_added)

        return adata.obs


def domain_from_STAGATE(adata: sc.AnnData,
                        cluster: Literal['m_clust', 'louvain', None] = 'm_cluster',
                        num_cluster: Optional[int] = None,
                        R_HOME: Optional[str] = None,
                        R_USER: Optional[str] = None,
                        device: Optional[str] = None,
                        model_graph: Literal['Radius', 'KNN'] = 'Radius',
                        rad_cutoff=None,
                        n_epochs: int = 1500,
                        lr: float = 0.0001,
                        inplace: bool = True,
                        used_obsm: str = 'STAGATE',
                        key_added: str = 'cluster_domain',
                        random_seed: int = 2020,
                        resolution_louvain: float = 0.5,
                        ) -> Union[tuple[ad.AnnData, pd.DataFrame], None]:
    """

    Parameters
    ----------
    adata
        anndata.Anndata object.
    cluster
        clustering method.
    num_cluster
        number of clusters (if 'cluster' is m_cluster)
    R_HOME
        PathLike, the location of R (if 'cluster' is m_cluster).
    R_USER
        PathLike, the location of R (if 'cluster' is m_cluster).
    device
        See torch.device.
    model_graph
        The network construction model.
    rad_cutoff
        The number of nearest neighbors when model='KNN'
    n_epochs
        Number of total epochs (STAGATE training).
    lr
        Learning rate for AdamOptimizer (STAGATE training).
    inplace

    used_obsm
    key_added
    random_seed
    resolution_louvain

    Returns
    -------

    """

    New_STAGATE = _STAGATE2Domain(adata,
                                  inplace=inplace)
    adata = New_STAGATE.get_Spatial_domain(rad_cutoff=rad_cutoff,
                                           model=model_graph,
                                           n_epochs=n_epochs,
                                           lr=lr,
                                           device=device
                                           )
    if cluster == 'm_clust':
        adata_obs = New_STAGATE.mclust_R(num_cluster=num_cluster,
                                         R_HOME=R_HOME,
                                         R_USER=R_USER,
                                         used_obsm=used_obsm,
                                         key_added=key_added,
                                         random_seed=random_seed,
                                         )
    elif cluster == 'louvain':
        adata_obs = New_STAGATE.louvain(resolution=resolution_louvain,
                                        key_added=key_added)
    else:
        adata_obs = None

    if not inplace:
        return adata, adata_obs
