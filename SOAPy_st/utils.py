import copy
import anndata
import numpy as np
import pandas as pd
import logging as logg
from anndata import AnnData
from typing import Optional, Union, Any


def _add_info_from_sample(
        adata: AnnData,
        sample_id: Optional[str] = None,
        keys: Union[str, list, None] = None,
        add: Any = None,
) -> AnnData:
    """
    Storing information
    """

    if isinstance(keys, str):
        keys = [keys]
        add = [add]

    try:
        adata.uns['SOAPy']
    except KeyError:
        adata.uns['SOAPy'] = {}
        logg.warning('adata has not been initialized, and adata.uns[\'SOAPy\'] has been established')

    if sample_id is None:
        for index, key in enumerate(keys):
            adata.uns['SOAPy'][key] = add[index]
            logg.info(f'{key} information has been incorporated into adata.uns[\'SOAPy\'][{key}]')
    else:
        try:
            adata.uns['SOAPy'][sample_id]
        except KeyError:
            adata.uns['SOAPy'][sample_id] = {}
        for index, key in enumerate(keys):
            adata.uns['SOAPy'][sample_id][key] = add[index]
            logg.info(f'{key} information from {sample_id} has been incorporated into adata.uns[\'SOAPy\'][{sample_id}][{key}]')

    return adata


def _get_info_from_sample(
        adata: AnnData,
        sample_id: Optional[str] = None,
        key: Optional[str] = None,
        printf: bool = True
):
    """
    Getting information
    """

    if 'SOAPy' not in adata.uns.keys():
        if printf:
            logg.error('SOAPy was not found in adata.uns', exc_info=True)
        raise KeyError()

    if sample_id is None:
        if key not in adata.uns['SOAPy'].keys():
            if printf:
                logg.error(f'{key} analysis of {sample_id} was not present, please conduct {key} analysis first',
                           exc_info=True)
            raise KeyError()
        data = adata.uns['SOAPy'][key]
    else:
        if key not in adata.uns['SOAPy'][sample_id].keys():
            if printf:
                logg.error(f'{key} analysis of {sample_id} was not present, please conduct {key} analysis first',
                           exc_info=True)
            raise KeyError()
        data = adata.uns['SOAPy'][sample_id][key]

    return data


def _scale(
        adata: AnnData,
        library_id: Optional[str] = None,
        res: Optional[str] = None,
) -> float:
    """
    Get the scale of the coordinates to the drawing map
    """
    try:
        if library_id is None:
            library_id = list(adata.uns['spatial'].keys())[0]
        scale = adata.uns['spatial'][library_id]['scalefactors']['tissue_' + res + '_scalef']
    except KeyError:
        scale = 1.0

    return scale


def _neighbor_network(
        points: np.ndarray,
        max_quantile: int = 98,
):

    def map_dis_to_points(point_pairs, dis_matrix, points_num):
        indices = [[] for i in range(points_num)]
        distances = [[] for i in range(points_num)]
        for pair in point_pairs:
            pi, pj, pk = pair
            indices[pi].append(pj)
            indices[pi].append(pk)
            indices[pj].append(pi)
            indices[pj].append(pk)
            indices[pk].append(pi)
            indices[pk].append(pj)
        indices = [list(set(neigh)) for neigh in indices]
        for index, neighs in enumerate(indices):
            for point in neighs:
                distances[index].append(dis_matrix[index, point])
        return distances, indices

    def filter_upper(distances, indices, max_):
        dis_flatten = []
        for dis_neigh in distances:
            dis_flatten += dis_neigh
        max = np.percentile(dis_flatten, max_)
        for pi, neigh in enumerate(indices):
            for index, pj in enumerate(neigh):
                if distances[pi][index] > max:
                    distances[pi][index] = -1
                    indices[pi][index] = -1
        distances = [[dis for dis in dis_n if not dis < 0] for dis_n in distances]
        indices = [[pointj for pointj in nei_n if not pointj < 0] for nei_n in indices]
        return distances, indices

    from scipy.spatial import Delaunay
    from scipy.spatial import distance
    import numpy as np

    tri = Delaunay(points)

    points = np.array(points, dtype=np.float32)
    dis_matrix = distance.cdist(points, points, 'euclidean')

    distances, indices = map_dis_to_points(tri.simplices, dis_matrix, len(points))
    distances, indices = filter_upper(distances, indices, max_quantile)

    distances = np.array([np.array(dis) for dis in distances], dtype=object)
    indices = np.array([np.array(ind) for ind in indices], dtype=object)
    return distances, indices


def _graph(col, row, method, cutoff=None, max_quantile=None):
    """
    Build an adjacent network

    """
    from sklearn.neighbors import NearestNeighbors

    def _KNN_neighbors(crd, k):
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nbrs.fit(crd)
        distances, indices = nbrs.kneighbors(crd)
        return distances, indices

    def _radius_neighbors(crd, radius):
        nbrs = NearestNeighbors(radius=radius, metric="euclidean")
        nbrs.fit(crd)
        distances, indices = nbrs.radius_neighbors(crd)
        return distances, indices

    X = list(zip(col, row))
    if method == 'knn':
        distances, indices = _KNN_neighbors(crd=X, k=cutoff)
    elif method == 'radius':
        distances, indices = _radius_neighbors(crd=X, radius=cutoff)
    elif method == 'regular':
        distances, indices = _KNN_neighbors(crd=X, k=cutoff)
        dist = distances.flatten()
        dist_cutoff = np.median(dist) * 1.2
        mask = np.where(distances < dist_cutoff, True, False)

        distances_correct = []
        indices_correct = []
        for row in range(mask.shape[0]):
            auxiliary_dist = []
            auxiliary_indices = []
            for col in range(mask.shape[1]):
                if mask[row, col]:
                    auxiliary_dist.append(distances[row, col])
                    auxiliary_indices.append(indices[row, col])
            distances_correct.append(np.array(auxiliary_dist))
            indices_correct.append(np.array(auxiliary_indices))
        distances = distances_correct
        indices = indices_correct
    elif method == 'neighbor':
        distances, indices = _neighbor_network(np.array(list(zip(col, row))), max_quantile)
    else:
        raise ValueError("The method must be in ['radius', 'knn', 'regular' ,'neighbor']")

    return indices, distances


def _check_adata_type(
        adata: anndata.AnnData,
        spatial_in_obsm: str,
        inplace: bool,
) -> anndata.AnnData:

    from scipy.sparse import csr_matrix

    if not inplace:
        adata = copy.deepcopy(adata)

    adata.X = csr_matrix(adata.X)

    if spatial_in_obsm in adata.obsm.keys():
        if isinstance(adata.obsm[spatial_in_obsm], pd.DataFrame):
            adata.obsm[spatial_in_obsm] = adata.obsm[spatial_in_obsm].values
    else:
        pass

    return adata