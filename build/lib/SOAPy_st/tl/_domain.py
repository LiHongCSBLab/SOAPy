import copy
import pandas as pd
import torch
import scanpy as sc
import numpy as np
from typing import Optional, Union
from ..utils import _scale, _add_info_from_sample, _get_info_from_sample, _check_adata_type
from typing import Optional, Literal
import anndata
import logging as logg

__all__ = ["domain_from_STAGATE", "domain_from_local_moran", "global_moran", "cal_aucell"]


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
                           **kwargs
                           ):

        import STAGATE
        adata = self.adata

        STAGATE.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)
        # STAGATE_pyG.Stats_Spatial_Net(adata)

        STAGATE.train_STAGATE(adata, **kwargs)

        return adata

    def mclust_R(self,
                 num_cluster: int,
                 key_added: str = 'domain',
                 random_seed: int = 2020,
                 ):

        import STAGATE

        adata = self.adata
        # sc.pp.neighbors(adata, use_rep='STAGATE')
        # sc.tl.umap(adata)

        # clust
        adata = STAGATE.mclust_R(adata,
                                 used_obsm='STAGATE',
                                 num_cluster=num_cluster,
                                 random_seed=random_seed)

        adata.obs.rename(columns={'mclust': key_added}, inplace=True)
        return adata
        # obs_df = adata.obs.dropna()

    def louvain(self,
                resolution: float = 0.5,
                key_added: str = 'domain',
                ):

        adata = self.adata

        sc.pp.neighbors(adata, use_rep='STAGATE')
        sc.tl.umap(adata)
        # louvain
        sc.tl.louvain(adata, resolution=resolution, key_added=key_added)

        return adata


def domain_from_STAGATE(
        adata: anndata.AnnData,
        cluster_method: Literal['m_clust', 'louvain'] = 'm_clust',
        cluster_number: int = 10,
        cluster_key: str = 'domain',
        rad_cutoff: Optional[float] = None,
        random_seed: int = 2020,
        resolution_louvain: float = 0.5,
        spatial_in_obsm: str = 'spatial',
        inplace: bool = True,
        **kwargs
) -> anndata.AnnData:
    """
    Using the STAGATE method generate the spatial domain.
    Detailed methods for STAGATE are available at https://stagate.readthedocs.io/en/latest/T1_DLPFC.html

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    cluster_method : Literal['m_clust', 'louvain']
        cluster method.
    cluster_number : int
        number of clusters (if 'cluster' is m_cluster)
    cluster_key : str
        Store the new label name for the domain category in adata.
    rad_cutoff : float, optional
        radius cutoff of spatial neighborhood.
    random_seed : int
        Random seed used in m_cluster.
    resolution_louvain : float
        resolution used in louvain cluster.
    spatial_in_obsm : str
        The key of spatial coordinates in adata.obsm
    inplace : bool
        If True, Modify directly in the original adata.
    **kwargs : ANY
        Parameters of STAGATE.train_STAGATE().

    Returns
    -------
    - :attr:`anndata.AnnData.obs` ``domain`` - The cluster of spatial domain.

    """

    adata = _check_adata_type(adata, spatial_in_obsm, inplace)
    if spatial_in_obsm != 'spatial':
        # stagate only recognize coordinate information from adata.obsm['spatial']
        adata.obsm['spatial'] = adata.obsm[spatial_in_obsm]

    New_STAGATE = _STAGATE2Domain(adata, inplace=inplace)

    adata = New_STAGATE.get_Spatial_domain(
        rad_cutoff=rad_cutoff,
        **kwargs
        )

    if cluster_method == 'm_clust':
        New_STAGATE.mclust_R(num_cluster=cluster_number,
                             key_added=cluster_key,
                             random_seed=random_seed,
                             )
    elif cluster_method == 'louvain':
        New_STAGATE.louvain(resolution=resolution_louvain,
                            key_added=cluster_key)
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

