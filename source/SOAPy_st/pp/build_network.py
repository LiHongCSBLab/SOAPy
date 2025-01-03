import warnings
import copy
import anndata
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Literal
from .utils import _filter_of_graph, _preprocessing_of_graph
from ..utils import _scale, _graph, _add_info_from_sample, _check_adata_type

__all__ = ["make_network"]


def make_network(
        adata: anndata.AnnData,
        sample_key: Optional[str] = None,
        sample: Union[str, int, list, None] = None,
        cluster_key: str = 'clusters',
        method: Literal['radius', 'knn', 'regular', 'neighbor'] = 'knn',
        cutoff: Union[float, int] = 6,
        max_quantile: Union[float, int] = 98,
        exclude: Union[str, dict] = None,
        scale: Union[str, float] = 'hires',
        spatial_in_obsm: str = 'spatial',
        inplace: bool = True
) -> anndata.AnnData:
    """
    A function to create a network based on spatial information.
    We offer four different ways to build a network: KNN network, range radiation network, regular network and First-order neighbor network

    'exclude' is a parameter to exclude categories that cannot form an edge, you can set 'same' and 'different' to
    specifies the same/different clusters may not be connected.If you want to define a custom class of points that
    can't be connected as edges, pass it as a dictionary.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    sample_key : str, optional
        Batch's key in adata.obs.
    sample : Union[str, int, list], optional
        The sample number for which the network needs to be built.
    cluster_key : str, optional
        The column label of clusters in adata.obs.
    method : str, optional
        the method to make network, select in 'Radius' and 'KNN'.
    cutoff : Union[float, int], optional
        In KNN network and regular network, cutoff means number of neighbors to use. In range radiation network, cutoff means range of parameter space to use
    max_quantile : Union[float, int], optional
        In Neighbor network, Order the distance of all sides, and more than max_quantile% of the sides will be removed
    exclude : Union[str, float], optional
        Excluding categories that cannot form an edge.
    scale : Union[str, float], optional
        Scale used in subsequent analyses. If it's Visium data it can also be HE image labels (hires or lower).
        Most of the time you don't need to change this.
    spatial_in_obsm : str, optional
        Keyword of coordinate information in obsm.
    inplace : bool, optional
        Whether to change the original adata.

    Returns
    -------
    :attr:`~anndata.AnnData.uns['SOAPy']`
        SOAPy generated parameters
    :attr:`~anndata.AnnData.uns['SOAPy']['indices']`
        adjacency matrix of network
    :attr:`~anndata.AnnData.uns['SOAPy']['distance']`
        distance matrix of network
    :attr:`~anndata.AnnData.uns['SOAPy']['edges']`
        edges of network

    """
    adata = _check_adata_type(adata, spatial_in_obsm, inplace)

    df_pixel = adata.obsm[spatial_in_obsm]
    df_pixel = pd.DataFrame(df_pixel)
    df_pixel = df_pixel.iloc[:, [0, 1]]

    if sample_key is None:
        if type(scale) != float:
            scale = _scale(adata, None, scale)

        indices, distances = _graph(
            col=list(df_pixel.iloc[:, 0] * scale),
            row=list(df_pixel.iloc[:, 1] * scale),
            method=method,
            cutoff=cutoff,
            max_quantile=max_quantile
        )

        _add_info_from_sample(adata,
                              keys=['indices', 'distance', 'edges'],
                              add=_get_edge(adata.obs, indices, distances, cluster_key, exclude))
    else:
        df_pixel['sample'] = adata.obs[sample_key].tolist()

        if sample is None:
            sample = df_pixel['sample'].unique().tolist()
        if not isinstance(sample, list):
            sample = [sample]
        for index in sample:
            df_pixel_sample = df_pixel[df_pixel['sample'] == index]
            obs = adata.obs[adata.obs[sample_key] == index]

            if type(scale) != float:
                scale = _scale(adata, index, scale)

            indices, distances = _graph(
                col=df_pixel_sample.iloc[:, 0] * scale,
                row=df_pixel_sample.iloc[:, 1] * scale,
                method=method,
                cutoff=cutoff,
                max_quantile=max_quantile
            )

            _add_info_from_sample(adata,
                                  sample_id=index,
                                  keys=['indices', 'distance', 'edges'],
                                  add=_get_edge(obs, indices, distances, cluster_key, exclude))

    return adata


def _get_edge(obs, indices, distances, cluster_label, exclude):
    """
    By adding constraints to the network, the modified advanced matrix and edge are obtained
    """
    if exclude is not None:
        indices, distances = _filter_of_graph(
            adata=obs,
            indices=indices,
            distances=distances,
            cluster_label=cluster_label,
            exclude=exclude
        )

    edge = _preprocessing_of_graph(
        clu_value=obs[cluster_label].values,
        indices=indices,
        distances=distances
    )
    return indices, distances, edge