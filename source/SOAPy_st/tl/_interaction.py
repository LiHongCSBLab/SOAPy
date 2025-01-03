import copy
import anndata
import numpy as np
import pandas as pd
from typing import Optional, Union, Literal
from ..utils import _add_info_from_sample, _get_info_from_sample, _check_adata_type
from .utils import (
    _count_edge,
    _randomize_helper,
    Iterators,
    _best_k,
)
import numba as nb
from joblib import Parallel, delayed
import logging as logg

__all__ = ['neighborhood_analysis', 'infiltration_analysis', 'get_c_niche']


class cell_network(object):
    """
    Collate the most basic neighborhood information.
    """

    def __init__(self,
                 adata,
                 cluster_key,
                 sample_key: Optional[str] = None,
                 sample=None,
                 ):
        if sample_key is not None:
            self.adata = adata[adata.obs[sample_key] == sample, :]
        else:
            self.adata = adata
        self._edge = _get_info_from_sample(self.adata, sample_id=sample, key='edges')
        self.cluster_key = cluster_key
        self.cluster = adata.obs[cluster_key]
        list_cluster = self.cluster.unique().tolist()
        new_list = [str(elem) for elem in list_cluster]
        self._cell_type_map = {v: i for i, v in enumerate(sorted(new_list))}
        self._species_of_clusters = len(self._cell_type_map)

    @property
    def cell_type(self):
        return self._cell_type_map.keys()

    @property
    def species_of_cell_type(self):
        return self._species_of_clusters


class cell_cell_interaction(cell_network):
    """
    Cell-cell interaction class
    """

    def __init__(self,
                 adata: anndata.AnnData,
                 cluster_key: str = 'clusters',
                 exclude: Union[str, dict] = None,
                 sample_key: Optional[str] = None,
                 sample=None,
                 ):

        super().__init__(adata, cluster_key, sample_key, sample)
        self.exclude = exclude

    def _randomize(self,
                   n_jobs: int,
                   num: int):
        adata = copy.deepcopy(self.adata)

        Iterator = Iterators(num,
                             adata,
                             self.cluster_key,
                             self._species_of_clusters,
                             self._cell_type_map)
        perms = Parallel(n_jobs=n_jobs)(delayed(_randomize_helper)(param[0], param[1], param[2], param[3])
                                        for param in Iterator)

        return perms

    def neighborhood_analysis(self,
                              n_jobs: int,
                              num: int,
                              method: str):

        @nb.njit()
        def enhance(matrix: np.ndarray, species: int):
            matrix_sum = np.sum(matrix, axis=0)
            for i in range(species):
                matrix[i, i] = matrix_sum[i] - matrix[i, i]
            for i in range(species):
                for j in range(i):
                    matrix[i, j] = matrix[i, j] / (matrix[i, i] + matrix[j, j] + 1)
                    matrix[j, i] = matrix[j, i] / (matrix[i, i] + matrix[j, j] + 1)
            for i in range(species):
                matrix[i, i] = 0

            return matrix

        mat_edge = _count_edge(self._edge, self._species_of_clusters, self._cell_type_map)
        perms = self._randomize(n_jobs=n_jobs, num=num)

        if method == 'included':
            pass
        elif method == 'excluded':
            mat_edge = enhance(mat_edge, self._species_of_clusters)
            species_list = [self._species_of_clusters] * num
            perms = Parallel(n_jobs=n_jobs)(delayed(enhance)(perm, spec) for perm, spec in zip(perms, species_list))
        perms = np.array(perms)
        zscore_array = (mat_edge - perms.mean(axis=0, dtype=np.float32)) / perms.std(axis=0, dtype=np.float32, ddof=1)
        zscore_df = pd.DataFrame(data=zscore_array, index=list(self._cell_type_map.keys()),
                                 columns=list(self._cell_type_map.keys()))

        return zscore_df

    def infiltration_analysis(self):

        # infiltration score
        mat_edge = _count_edge(self._edge, self._species_of_clusters, self._cell_type_map)
        mat_score = np.zeros((self.species_of_cell_type, self.species_of_cell_type))
        for index in range(self.species_of_cell_type):
            for column in range(self.species_of_cell_type):
                if index == column:
                    continue
                mat_score[index, column] = mat_edge[index, column] / min(mat_edge[index, index],
                                                                        mat_edge[column, column])

        pd_score = pd.DataFrame(mat_score, index=list(self._cell_type_map.keys()),
                                columns=list(self._cell_type_map.keys()))
        # self.adata.uns['SOAPy']['infiltration_score'] = pd_score

        # cell type number
        cluster_num = {}

        for c in self._cell_type_map.keys():
            cluster_num[c] = len(np.where(self.cluster == c)[0])

        return pd_score, cluster_num


class niche(cell_network):
    """
    Niche analysis. support multi-analysis.
    """

    def __init__(self,
                 adata: anndata.AnnData,
                 cluster_key: str = 'clusters',
                 sample_key: Optional[str] = None,
                 sample: str = None
                 ):

        super().__init__(adata, cluster_key, sample_key, sample=sample)
        self.sample = sample
        self.mat_neigh = None

    @property
    def i_niche(self):
        edge = self._edge
        mat_neigh = np.zeros((len(self.adata.obs_names), self._species_of_clusters))
        for index in edge.index:
            mat_neigh[int(edge.loc[index, 'point_1']), self._cell_type_map[edge.at[index, 'cluster_2']]] += 1
            mat_neigh[int(edge.loc[index, 'point_2']), self._cell_type_map[edge.at[index, 'cluster_1']]] += 1

        for index in range(len(self.adata.obs_names)):
            mat_neigh[index, self._cell_type_map[self.cluster[index]]] += 1

        pd_i_niche = pd.DataFrame(mat_neigh, index=self.adata.obs_names, columns=list(self.cell_type))

        for index in range(mat_neigh.shape[0]):
            if sum(mat_neigh[index, :]) != 0:
                mat_neigh[index, :] = mat_neigh[index, :] / sum(mat_neigh[index, :])
        self.mat_neigh = mat_neigh
        _add_info_from_sample(self.adata, sample_id=self.sample, keys='mat_neigh', add=mat_neigh)

        return pd_i_niche


class mult_sample_niche(object):
    def __init__(self,
                 adata: anndata.AnnData,
                 sample: Union[str, list, None],
                 celltype_key: str = 'clusters',
                 sample_key: Optional[str] = None,
                 ):
        list_i_niche = []
        list_barcode = []
        in_sample_cluster = []
        in_sample_list = []
        for sample_id in sample:
            i_niche_single = niche(adata, celltype_key, sample_key, sample=sample_id)
            list_i_niche.append(i_niche_single.i_niche)
            in_sample_cluster += i_niche_single.adata.obs[celltype_key].tolist()
            in_sample_list += i_niche_single.adata.obs[sample_key].tolist()
            list_barcode += i_niche_single.adata.obs_names.tolist()

        all_cluster = adata.obs[celltype_key].tolist()

        self.cluster = celltype_key
        self.in_sample_cluster = in_sample_cluster
        self.in_sample_list = in_sample_list
        self.barcode = list_barcode
        new_list = [str(elem) for elem in all_cluster]
        celltype_unique = np.unique(new_list)
        self._cell_type_map = {v: i for i, v in enumerate(sorted(celltype_unique))}
        self._species_of_clusters = len(self._cell_type_map)

        dict_num_spots = {}
        i_mult_niche = []
        for index, mat_niche in enumerate(list_i_niche):

            dict_num_spots[sample[index]] = mat_niche.shape[0]

            if self._species_of_clusters != mat_niche.shape[1]:
                mat_label = mat_niche.columns.tolist()
                label_add_mat = list(set(all_cluster) - set(mat_label))
                if len(label_add_mat) > 0:
                    for label in label_add_mat:
                        mat_niche[label] = [0.0 for i in range(mat_niche.shape[0])]

            if index == 0:
                i_mult_niche = copy.deepcopy(mat_niche)
                continue

            i_mult_niche = pd.concat([i_mult_niche, mat_niche])

        i_mult_niche = i_mult_niche.reindex(columns=np.unique(all_cluster), fill_value=0)
        self.mult_niche = i_mult_niche
        self.num_niche = dict_num_spots

    def mult_c_niche(self,
                     k_max,
                     sdbw: bool = True) -> pd.DataFrame:
        res, km = _best_k(self.mult_niche.values, k_max, sdbw)
        df_mult_niche = self.mult_niche
        # df_mult_niche = pd.DataFrame(data=self.mult_niche, columns=list(self._cell_type_map.keys()))

        df_mult_niche['sample'] = self.in_sample_list
        df_mult_niche['barcode'] = self.barcode
        df_mult_niche['celltype'] = self.in_sample_cluster
        df_mult_niche['C_niche'] = res

        return df_mult_niche


def neighborhood_analysis(
        adata: anndata.AnnData,
        method: Literal['excluded', 'include'] = 'excluded',
        cluster_key: str = 'clusters',
        sample_key: Optional[str] = None,
        sample: Union[int, str, list, None] = None,
        n_jobs: Optional[int] = None,
        n_iters: int = 1000,
        inplace: bool = True,
) -> anndata.AnnData:
    """
    Compute neighborhood enrichment Z-score by permutation test.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    method : str, optional
        'included': Z-scores of edges between two cell types were counted directly after randomization.
        'excluded': After randomization, remove self-connected edges between cells of the same type and calculate the z-score of edges between two cell types.
    cluster_key : str, optional
        The label of cluster in adata.obs.
    sample_key : str, optional
        The keyword of sample id in adata.obs.columns.
    sample: Union[int, str, list, None],
        The samples involved in calculating infiltration score.
    n_jobs : int, optional
        The maximum number of concurrently running jobs.
    n_iters : int, optional
        Number of rounds of random grouping for permutation tests.
    inplace : bool, optional
        Whether to change the original adata.

    Returns
    -------
    - :attr:`anndata.AnnData.uns` ``['SOAPy']['include_method' or 'exclude_method']['dic_crd_poly']`` - neighborhood score

    """
    if sample is not None and sample_key is None:
        logg.error(f'Mult-sample niche analysis cannot be specified without a given sample key')
        raise ValueError
    if sample is None and sample_key is not None:
        sample = adata.obs[sample_key].unique().tolist()
        logg.info(f'Use all samples in the niche calculation')
    if not isinstance(sample, list):
        sample = [sample]

    adata = _check_adata_type(adata, 'spatial', inplace)

    for sample_id in sample:
        if sample_id is not None:
            bdata = adata[adata.obs[sample_key] == sample_id, :].copy()
            bdata.uns['SOAPy'] = {}
            bdata.uns['SOAPy'] = copy.deepcopy(adata.uns['SOAPy'][sample_id])
        else:
            bdata = adata
        new_cci = cell_cell_interaction(bdata, cluster_key)
        zscore = new_cci.neighborhood_analysis(n_jobs=n_jobs, num=n_iters, method=method)
        _add_info_from_sample(adata, sample_id=sample_id, keys=method + '_score', add=zscore)

    return adata


def infiltration_analysis(
        adata: anndata.AnnData,
        cluster_key: str = 'clusters',
        sample_key: Optional[str] = None,
        sample: Union[int, str, list, None] = None,
        inplace: bool = True,
) -> anndata.AnnData:
    """
    The infiltration score was calculated by the infiltration of non-parenchymal cells into parenchymal cells.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    cluster_key : str, optional
        The label of cluster in adata.obs.
    sample_key : str, optional
        The keyword of sample id in adata.obs.columns.
    sample: Union[int, str, list, None],
        The samples involved in calculating infiltration score.
    inplace : bool, optional
        Whether to change the original adata.

    Returns
    -------
    - :attr:`anndata.AnnData.uns` ``['SOAPy']['infiltration score']['dic_crd_poly']`` - infiltration score

    """
    adata = _check_adata_type(adata, 'spatial', inplace)


    if sample is not None and sample_key is None:
        logg.error(f'Mult-sample niche analysis cannot be specified without a given sample key')
        raise ValueError
    if sample is None and sample_key is not None:
        sample = adata.obs[sample_key].unique().tolist()
        logg.info(f'Use all samples in the niche calculation')
    if not isinstance(sample, list):
        sample = [sample]

    for sample_id in sample:
        if sample_id is not None:
            bdata = adata[adata.obs[sample_key] == sample_id, :].copy()
            bdata.uns['SOAPy'] = {}
            bdata.uns['SOAPy'] = copy.deepcopy(adata.uns['SOAPy'][sample_id])
        else:
            bdata = adata

        new_cci = cell_cell_interaction(bdata, cluster_key)
        infiltration_score, cluster_num = new_cci.infiltration_analysis()

        _add_info_from_sample(adata,
                              sample_id=sample_id,
                              keys=['infiltration_analysis', 'cluster_number'],
                              add=[infiltration_score, cluster_num])
    _add_info_from_sample(adata, sample_id=None, keys='infiltration_sample', add=sample)

    return adata


# @decorator_helper
def get_c_niche(
        adata: anndata.AnnData,
        k_max: int,
        niche_key: str = 'C_niche',
        celltype_key: str = 'clusters',
        sample_key: Optional[str] = None,
        sample: Union[str, list] = None,
        sdbw: bool = False,
        inplace: bool = True,
) -> anndata.AnnData:
    """
    The C-niche is calculated using the cell type of the neighborhood.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    k_max:
        If sdbw is true, k_max is the maximum number of c-niche; If sdbw is false, it represents the number of c-niche
    niche_key : str, optional
        Add the keyword of niche in adata.obs.columns.
    celltype_key : str, optional
        The keyword of spot cluster in adata.obs.columns.
    sample_key : str, optional
        The keyword of sample id in adata.obs.columns.
    sample: Union[str, list, None],
        The samples involved in calculating the niche.
    sdbw : bool, optional
        Automated cluster number screening using sdbw.
    inplace : bool, optional
        Whether to change the original adata.

    Returns
    -------
    - :attr:`anndata.AnnData.obs` ``[niche]`` - c-niche of each spot
    - :attr:`anndata.AnnData.uns` ``[niche]`` - i-niche of each spot (cell type composition of neighbors)

    """
    adata = _check_adata_type(adata, 'spatial', inplace)

    if isinstance(sample, str):
        sample = [sample]
    if sample is not None and sample_key is None:
        logg.error(f'Mult-sample niche analysis cannot be specified without a given sample key')
        raise ValueError
    if sample is None and sample_key is not None:
        sample = list(adata.obs[sample_key].unique())
        logg.info(f'Use all samples in the niche calculation')

    assert celltype_key in adata.obs.columns

    niche_ = mult_sample_niche(adata=adata, sample=sample, celltype_key=celltype_key, sample_key=sample_key)
    niche_inf = niche_.mult_c_niche(k_max=k_max, sdbw=sdbw)
    _add_info_from_sample(adata=adata, sample_id=None, keys='niche', add=niche_inf)

    if sample is not None:
        adata.obs[niche_key] = None
        adata.obs.loc[np.isin(adata.obs[sample_key], sample), niche_key] = niche_inf['C_niche'].tolist()
    else:
        adata.obs[niche_key] = niche_inf['C_niche'].tolist()

    return adata

