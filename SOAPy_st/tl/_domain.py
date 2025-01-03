import copy
import pandas as pd
import scanpy as sc
import numpy as np
from typing import Optional, Union
from ..utils import _scale, _add_info_from_sample, _get_info_from_sample, _check_adata_type
from typing import Optional, Literal
import anndata
import logging as logg
import os

__all__ = ["domain_from_unsupervised", "domain_from_local_moran", "global_moran", "cal_aucell"]


class _SpatialDomain(object):

    def __init__(self,
                 adata: sc.AnnData,
                 domain_method: str,
                 inplace: bool = True,
                 ):

        if domain_method == 'stagate':
            self.domain_emb = 'STAGATE'
        elif domain_method == 'graphst':
            self.domain_emb = 'emb'
        elif domain_method == 'scanit':
            self.domain_emb = 'X_scanit'

        if inplace:
            self.adata = adata
        else:
            self.adata = copy.deepcopy(self.adata)

    def get_stagate_domain(self,
                           graph_model='Radius',
                           rad_cutoff=None,
                           k_cutoff=None,
                           **kwargs
                           ):

        from .other_package_without_pip import STAGATE_pyG as STAGATE

        adata = self.adata

        STAGATE.Cal_Spatial_Net(adata, model=graph_model, k_cutoff=k_cutoff, rad_cutoff=rad_cutoff)
        # STAGATE_pyG.Stats_Spatial_Net(adata)

        STAGATE.train_STAGATE(adata, **kwargs)

        return adata

    def get_graphST_domain(self,
                           **kwargs
                           ):

        from .other_package_without_pip.GraphST import GraphST
        from sklearn.decomposition import PCA

        adata = self.adata
        model = GraphST.GraphST(adata, **kwargs)

        # train model
        adata = model.train()

        pca = PCA(n_components=20, random_state=42)
        embedding = pca.fit_transform(adata.obsm[self.domain_emb].copy())
        adata.obsm['emb_pca'] = embedding
        self.domain_emb = 'emb_pca'
        self.adata = adata

        return adata

    def get_scanit_domain(self,
                          graph_model='knn',
                          alpha_n_layer=1,
                          k_cutoff=10,
                          **kwargs
                          ):
        from .other_package_without_pip import scanit

        adata = self.adata
        adata.X = adata.X.toarray()
        scanit.tl.spatial_graph(adata, method=graph_model, knn_n_neighbors=k_cutoff, alpha_n_layer=alpha_n_layer)
        scanit.tl.spatial_representation(adata, **kwargs)
        self.adata = adata

        return adata

    def mclust_R(self,
                 num_cluster: int,
                 key_added: str = 'domain',
                 modelNames='EEE',
                 random_seed: int = 2020,
                 ):

        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri
        adata = self.adata

        np.random.seed(random_seed)
        robjects.r.library("mclust")
        rpy2.robjects.numpy2ri.activate()
        r_random_seed = robjects.r['set.seed']
        r_random_seed(random_seed)
        rmclust = robjects.r['Mclust']

        res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[self.domain_emb]), num_cluster, modelNames)
        mclust_res = np.array(res[-2])

        adata.obs[key_added] = mclust_res
        adata.obs[key_added] = adata.obs[key_added].astype('int')
        adata.obs[key_added] = adata.obs[key_added].astype('category')

        return adata

    def louvain(self,
                resolution: float = 0.5,
                key_added: str = 'domain',
                ):

        adata = self.adata
        sc.pp.neighbors(adata, use_rep=self.domain_emb)
        sc.tl.umap(adata)
        # louvain
        sc.tl.louvain(adata, resolution=resolution, key_added=key_added)

        return adata


def domain_from_unsupervised(
        adata: anndata.AnnData,
        domain_method: Literal['stagate', 'graphst', 'scanit'] = 'stagate',
        graph_model: str = None,
        k_cutoff: Optional[int] = None,
        rad_cutoff: Optional[float] = None,
        alpha_n_layer: Optional[int] = None,
        cluster_method: Literal['m_clust', 'louvain'] = 'm_clust',
        cluster_number: int = 10,
        cluster_key: str = 'domain',
        random_seed: int = 2020,
        resolution_louvain: float = 0.5,
        spatial_in_obsm: str = 'spatial',
        inplace: bool = True,
        **kwargs
) -> anndata.AnnData:
    """
    Generate spatial domains using unsupervised learning methods.
    This function supports multiple spatial domain identification algorithms, including STAGATE, GraphST, and ScanIT,
    and provides two clustering methods (mclust and Louvain) to cluster the spatial domains.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.

    domain_method : Literal['stagate', 'graphst', 'scanit'], optional (default: 'stagate')
        The method used to generate spatial domains. Available options are:
        - 'stagate': Use the STAGATE algorithm to generate spatial domains.
        - 'graphst': Use the GraphST algorithm to generate spatial domains.
        - 'scanit': Use the ScanIT algorithm to generate spatial domains.

    graph_model : str, optional (default: None)
        The model used to construct the spatial graph. For STAGATE, options are 'Radius' or 'KNN';
        for ScanIT, options are 'alpha shape' or 'knn'.

    k_cutoff : Optional[int], optional (default: None)
        The number of KNN neighbors used to construct the spatial graph. Only valid when graph_model is 'KNN' or 'knn'.

    rad_cutoff : Optional[float], optional (default: None)
        The radius cutoff used to construct the spatial graph. Only valid when graph_model is 'Radius'.

    alpha_n_layer : Optional[int], optional (default: None)
        The number of alpha layers used in the ScanIT algorithm. Only valid when domain_method is 'scanit'.

    cluster_method : Literal['m_clust', 'louvain'], optional (default: 'm_clust')
        The clustering algorithm used. Available options are:
        - 'm_clust': Use the mclust algorithm for clustering.
        - 'louvain': Use the Louvain algorithm for clustering.

    cluster_number : int, optional (default: 10)
        The number of clusters. Only valid when cluster_method is 'm_clust'.

    cluster_key : str, optional (default: 'domain')
        The key in adata.obs where the clustering results will be stored.

    random_seed : int, optional (default: 2020)
        Random seed for reproducibility.

    resolution_louvain : float, optional (default: 0.5)
        The resolution parameter for the Louvain algorithm. Only valid when cluster_method is 'louvain'.

    spatial_in_obsm : str, optional (default: 'spatial')
        The key in adata.obsm where spatial coordinates are stored.

    inplace : bool, optional (default: True)
        Whether to modify the AnnData object in place. If False, a modified copy is returned.

    **kwargs : dict
        Additional parameters passed to the specific algorithms.


    Returns
    -------
    - :attr:`anndata.AnnData.obs` ``domain`` - The cluster of spatial domain.

    """

    adata = _check_adata_type(adata, spatial_in_obsm, inplace)
    if spatial_in_obsm != 'spatial':
        # stagate only recognize coordinate information from adata.obsm['spatial']
        adata.obsm['spatial'] = adata.obsm[spatial_in_obsm]

    New_SpatialDomain = _SpatialDomain(adata, domain_method=domain_method, inplace=inplace)

    if domain_method == 'stagate':

        if graph_model == None:
            graph_model = 'Radius'
        elif graph_model == 'knn':
            graph_model = 'KNN'

        assert (graph_model in ['Radius', 'KNN']), 'graph_model of STAGATE must in [\'Radius\', \'KNN\']'

        adata = New_SpatialDomain.get_stagate_domain(
            graph_model=graph_model,
            k_cutoff=k_cutoff,
            rad_cutoff=rad_cutoff,
            **kwargs
            )
    elif domain_method == 'graphst':
        adata = New_SpatialDomain.get_graphST_domain(
            **kwargs
            )
    elif domain_method == 'scanit':

        if graph_model == None or graph_model == 'KNN':
            graph_model = 'knn'

        assert (graph_model in ['alpha shape', 'knn']), 'graph_model of scanit must in [\'alpha shape\', \'knn\']'

        adata = New_SpatialDomain.get_scanit_domain(
            graph_model=graph_model,
            alpha_n_layer=alpha_n_layer,
            k_cutoff=k_cutoff,
            **kwargs
            )

    if cluster_method == 'm_clust':
        New_SpatialDomain.mclust_R(num_cluster=cluster_number,
                                   key_added=cluster_key,
                                   random_seed=random_seed,
                                   )
    elif cluster_method == 'louvain':
        New_SpatialDomain.louvain(resolution=resolution_louvain,
                                  key_added=cluster_key
                                  )
    else:
        logg.error(f'{cluster_method} is not in [\'m_clust\', \'louvain\']', exc_info=True)
        raise ValueError()

    return adata


def domain_from_local_moran(
        adata: anndata.AnnData,
        score_key: str,
        moran_label_key: str = 'hotspot',
        k: int = 6,
        pvalue_filter: float = 0.05,
        attribute_filter: Optional[float] = None,
        spatial_lag_filter: Optional[float] = None,
        zscore: bool = True,
        fdr: bool = True,
        scale: Union[str, float] = 'hires',
        spatial_in_obsm: str = 'spatial',
        inplace: bool = True,
) -> anndata.AnnData:
    """
    Using the local moran method generate the spatial domain.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    score_key : str
        The key for the Moran index need to be computed. It could be gene name ,key in adata.obs_index or AUCell name.
    moran_label_key : str
        The key for storing Moran labels in adata.obs.
    k : int, optional
        Number of nearest neighbors in KNN.
    pvalue_filter : float, optional
        Threshold for determining hotspot regions based on the p-value of the local Moran's I statistic.
        A smaller value indicates higher statistical significance, typically set to 0.05 or another significance level.
        For a detailed explanation, see https://pysal.org/esda/generated/esda.Moran_Local.html#esda.Moran_Local
    attribute_filter : float, optional
        Threshold for attribute filtering (AUCell score).
        If provided, it affects how observations are labeled as "Hotspot" or "Not" based on whether the attribute value
        is greater than or equal to this threshold.
        For a detailed explanation, see https://pysal.org/esda/generated/esda.Moran_Local.html#esda.Moran_Local
    spatial_lag_filter : float, optional
        Threshold for the average attribute of neighboring spots.
        If provided, it affects how observations are labeled as "Hotspot" or "Not" based on whether the spatial lag
        value of the local Moran's I statistic is greater than or equal to this threshold.
    zscore : bool, optional
        If True, calculate the z-score of the attribute.
    fdr : bool, optional
        If True, the p-values were corrected for FDR.
    scale : Union[str, float], optional
        scale used in subsequent analyses. If it's Visium data it can also be HE image labels (hires or lower).
        Most of the time you don't need to change this.
    spatial_in_obsm : str, optional
        The key of spatial coordinates in adata.obsm.
    inplace : bool, optional
        If True, Modify directly in the original adata.

    Returns
    -------
    - :attr:`anndata.AnnData.uns` ``['SOAPy']['local_moran']`` - Local Moran information for each spot

    """
    import libpysal
    from esda.moran import Moran_Local
    from libpysal.weights.spatial_lag import lag_spatial
    from pyscenic.binarization import derive_threshold

    adata = _check_adata_type(adata, spatial_in_obsm, inplace)
    if type(scale) != float:
        scale = _scale(adata, None, scale)

    df_pixel = adata.obsm[spatial_in_obsm]
    if isinstance(df_pixel, np.ndarray):
        df_pixel = pd.DataFrame(df_pixel)
    df_pixel = df_pixel.iloc[:, [0, 1]]

    obs = copy.deepcopy(adata.obs)
    obs['x'] = list(df_pixel.iloc[:, 0] * scale)
    obs['y'] = list(df_pixel.iloc[:, 1] * scale)
    coord_df = obs[['y', 'x']]

    try:
        score = _get_info_from_sample(adata, sample_id=None, key='aucell', printf=False)
        score = score[score_key]
    except KeyError:
        if score_key in adata.obs.columns:
            score = adata.obs[score_key]
        else:
            var_names = adata.var_names.tolist()
            index = var_names.index(score_key)
            express = adata.X
            # if type(adata.X) is np.ndarray:
            #     express = adata.X
            # else:
            #     express = adata.X.toarray()
            score = {score_key: express[:, index].tolist()}
            score = pd.DataFrame(score, index=adata.obs.index)

    w = libpysal.weights.KNN(coord_df, k=k)
    moran = Moran_Local(score.astype(np.float64), w)

    if zscore:
        local_moran_df = pd.DataFrame({'Attribute': moran.z,
                                       'Spatial Lag': lag_spatial(moran.w, moran.z),
                                       'Moran_Is': moran.Is,
                                       'Cluster': moran.q,
                                       'P_value': moran.p_sim})
    else:
        local_moran_df = pd.DataFrame({'Attribute': score.tolist(),
                                       'Spatial Lag': lag_spatial(moran.w, score.tolist()),
                                       'Moran_Is': moran.Is,
                                       'Cluster': moran.q,
                                       'P_value': moran.p_sim})

    if fdr:
        from statsmodels.stats.multitest import fdrcorrection
        _, p_adjust = fdrcorrection(local_moran_df['P_value'])
        local_moran_df['P_value'] = p_adjust

    if attribute_filter is None:
        attribute_filter = derive_threshold(local_moran_df, 'Attribute')
    print('attribute_filter:', attribute_filter)
    if spatial_lag_filter is None:
        spatial_lag_filter = derive_threshold(local_moran_df, 'Spatial Lag')
    print('spatial_lag_filter:', spatial_lag_filter)

    local_moran_df['Label'] = [
        'Hotspot' if (local_moran_df.loc[i, 'P_value'] <= pvalue_filter) and (
                    local_moran_df.loc[i, 'Attribute'] >= attribute_filter) and (
                    local_moran_df.loc[i, 'Spatial Lag'] >= spatial_lag_filter) else 'Not' for i in
        range(local_moran_df.shape[0])]

    local_moran_df.index = coord_df.index
    adata.obs['Moran_domain'] = local_moran_df['Label']

    local_moran = {'LocalMoran': local_moran_df,
                   'attribute_filter': attribute_filter,
                   'spatial_lag_filter': spatial_lag_filter,
                   'pvalue_filter': pvalue_filter}

    adata.obs[moran_label_key] = local_moran_df['Label'].tolist()
    _add_info_from_sample(adata, sample_id=None, keys='local_moran', add=local_moran)

    return adata


def global_moran(
        adata: anndata.AnnData,
        score_labels: Union[list, str, None] = None,
        k: int = 6,
        scale: Union[str, float] = 'hires',
        spatial_in_obsm: str = 'spatial',
        inplace: bool = True,
) -> anndata.AnnData:
    """
    The global Moran index is calculated based on selected genes or indicators such as AUCell score.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    score_labels : str
        The label or label lists for the Moran index need to be computed.
        It could be gene name in adata.uns_names or in adata.uns['SOAPy']['aucell'] (the result of cal_aucell()).
    k : int, optional
        Number of nearest neighbors in KNN
    scale : Union[str, float], optional
        The scaling factor for distance scaling. If it's Visium data it can also be HE image labels (hires or lower).
        Most of the time you don't need to change this.
    spatial_in_obsm : str, optional
        Keyword of coordinate information in obsm.
    inplace : bool, optional
        Whether to change the original adata.

    Returns
    -------
    - :attr:`anndata.AnnData.uns` ``['SOAPy']['global_moran']`` - Global Moran information for each keyword

    """
    from tqdm import tqdm
    from esda.moran import Moran
    import libpysal
    from shapely import geometry
    import geopandas as gpd

    adata = _check_adata_type(adata, spatial_in_obsm, inplace)

    if score_labels is None:
        data_df = _get_info_from_sample(adata, sample_id=None, key='aucell')
    else:
        if isinstance(score_labels, str):
            score_labels = [score_labels]
        for score_label in score_labels:
            data_df = pd.DataFrame(index=adata.obs.index)
            try:
                score = _get_info_from_sample(adata, sample_id=None, key='aucell')
                data_df[score_label] = score[score_label].tolist()
            except KeyError:
                var_names = adata.var_names.tolist()
                if score_label in var_names:
                    data_df[score_label] = adata[:, score_label].X.toarray().T[0]
                else:
                    logg.error(f'{score_label} is not in the aucell and gene list, please check. '
                               f'This keyword has been skipped')
                    continue

    if type(scale) != float:
        scale = _scale(adata, None, scale)

    df_pixel = adata.obsm[spatial_in_obsm]
    if isinstance(df_pixel, np.ndarray):
        df_pixel = pd.DataFrame(df_pixel)
    df_pixel = df_pixel.iloc[:, [0, 1]]

    obs = copy.deepcopy(adata.obs)
    obs['x'] = list(df_pixel.iloc[:, 0] * scale)
    obs['y'] = list(df_pixel.iloc[:, 1] * scale)
    coord_df = obs[['y', 'x']]

    coord_df = coord_df.loc[data_df.index, :]
    coord_point = [geometry.Point(coord_df.iloc[i, 0], coord_df.iloc[i, 1]) for i in range(coord_df.shape[0])]

    geo_df = gpd.GeoDataFrame(data=data_df,
                              geometry=coord_point)
    w = libpysal.weights.KNN.from_dataframe(geo_df, k=k)

    # Global Moran index
    geo_mat = geo_df.values
    moran = [Moran(geo_mat[:, i].astype('float64'), w) for i in tqdm(range(geo_df.shape[1] - 1))]
    moran_I = [m.I for m in moran]
    moran_p = [m.p_sim for m in moran]
    global_moran = pd.DataFrame({'Moran_I': moran_I,
                                 'P_value': moran_p})

    global_moran.index = data_df.columns.tolist()[0:data_df.shape[1] - 1]

    _add_info_from_sample(adata, sample_id=None, keys='global_moran', add=global_moran)

    return adata


def cal_aucell(
        adata: anndata.AnnData,
        signatures: dict,
        gene_percent: float = 0.05,
        inplace: bool = True,
        **kwargs
) -> anndata.AnnData:
    """
    AUCell scores were calculated for given gene lists.
    For more information about AUCell, see https://www.nature.com/articles/s41596-020-0336-2

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    signatures : dict
        The name of each gene set and the corresponding gene list.
    gene_percent : float, optional
        The fraction of the ranked genome to take into account for the calculation of the area Under the recovery Curve.
    inplace : bool, optional
        Whether to change the original adata.
    kwargs:
        Other Parameters of pyscenic.aucell.aucell.

    Returns
    -------
        - :attr:`anndata.AnnData.uns` ``['SOAPy']['global_moran']`` - AUCell score of signatures
    """
    from ctxcore.genesig import GeneSignature
    from pyscenic.aucell import aucell

    adata = _check_adata_type(adata, 'spatial', inplace)

    exp_mat=pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
    # generate GeneSignature list
    gs = [GeneSignature(name=k, gene2weight=v) for (k, v) in signatures.items()]
    gene_thres = pd.Series(np.count_nonzero(exp_mat, axis=1)).quantile(gene_percent)/exp_mat.shape[1]
    aucell_df = aucell(exp_mat, gs, auc_threshold=gene_thres, **kwargs)
    _add_info_from_sample(adata, sample_id=None, keys='aucell', add=aucell_df)

    return adata

