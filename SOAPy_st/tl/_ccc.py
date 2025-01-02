import math
import anndata
import numpy as np
import numba as nb
import pandas as pd
from numba import prange, njit
from anndata import AnnData
from scipy.stats.mstats import gmean
import copy
from typing import Optional, Union, Tuple, Literal, List
from ..utils import _scale, _graph, _add_info_from_sample, _check_adata_type
from tqdm import tqdm
from .utils import adj_pvals, _count_edge, allocation_edge_2_diff_cell_type
from ..pp.utils import _preprocessing_of_graph
import logging as logg
import re
import os

__all__ = ['cell_level_communications', 'cell_type_level_communication', 'lr_pairs']


class lr_pairs():
    """
    Class for ligand and receptor pairs
    """

    def __init__(
            self,
            lr_data: Union[pd.DataFrame, Literal['human', 'mouse']] = 'human',
            Annotation_key: Optional[str] = 'annotation',
            ligand_key: Optional[str] = 'ligand_symbol',
            receptor_key: Optional[str] = 'receptor_symbol',
            ):
        """
        Parameters
        ----------
        lr_data : pd.DataFrame
            Ligand and receptor information database. Default: CellChat human database
        Annotation_key : str
            The key of Annotation (Contact or secretory) in lr_data
        ligand_key : str
            The key of ligand in lr_data
        receptor_key : str
            The key of receptor in lr_data
        """

        self.ligand_key = ligand_key
        self.receptor_key = receptor_key
        self.Annotation_key = Annotation_key

        if lr_data is 'human':
            path = os.path.dirname(os.path.realpath(__file__))
            lr_data = pd.read_csv(f'{path}/datasets/cci/human/Human-2020-Jin-LR-pairs.csv', index_col=0, header=0)
            lr_data = lr_data.replace('ECM-Receptor', 'Cell-Cell Contact')
        elif lr_data is 'mouse':
            path = os.path.dirname(os.path.realpath(__file__))
            lr_data = pd.read_csv(f'{path}/datasets/cci/mouse/Mouse-2020-Jin-LR-pairs.csv', index_col=0, header=0)
            lr_data = lr_data.replace('ECM-Receptor', 'Cell-Cell Contact')

        if Annotation_key is None:
            self.lr_data = lr_data.loc[:, [ligand_key, receptor_key]]
        else:
            self.lr_data = lr_data.loc[:, [ligand_key, receptor_key, Annotation_key]]

        self.complexes = {}
        self.gene_name = []

    def get_complexes(self,
                      complex_sep='&',
                      ):
        """
        Get the names of all the complexes

        Parameters
        ----------
        complex_sep
            A separator used to separate subunits in the database, default: '&'

        Returns
        -------
        self.complexes
            A dict for complexes name and its subunits (key: complexes name, value: subunits)
        self.gene_name
            A list for all genes' name
        """

        col_a = self.ligand_key
        col_b = self.receptor_key
        complexes = {}
        gene_name = []

        for idx, row in self.lr_data.iterrows():
            prot_a = row[col_a]
            prot_b = row[col_b]

            gene_name.append(prot_a)
            gene_name.append(prot_b)

            if complex_sep in prot_a:
                comp = set([l for l in prot_a.split(complex_sep)])
                complexes[prot_a] = comp
            if complex_sep in prot_b:
                comp = set([r for r in prot_b.split(complex_sep)])
                complexes[prot_b] = comp

        self.complexes = complexes
        self.gene_name = list(set(gene_name))


def get_used_lr(
        lr_pairs: lr_pairs,
        key=None
) -> tuple[list, list]:
    """
    Collate and return information of ligand-receptor pairs
    """
    lr_names = []
    lr_used = []
    ln = lr_pairs.ligand_key
    rn = lr_pairs.receptor_key
    if key is None:
        for id, row in lr_pairs.lr_data:
            lr_names.append(row[ln] + ':' + row[rn])
            lr_used.append([row[ln], row[rn]])
    else:
        for id, row in lr_pairs.lr_data.iterrows():
            if row[lr_pairs.Annotation_key] == key:
                lr_names.append(row[ln] + ':' + row[rn])
                lr_used.append([row[ln], row[rn]])
    return lr_names, lr_used


def _lr_helper(
        adata: anndata.AnnData,
        lr_pairs: lr_pairs,
        scale: Union[str, float],
        key: str,
        method: str,
        norm: bool = True,
        percent_of_drop: Union[int, float] = 0,
        radius: Optional[float] = None,
        max_quantile: Optional[float] = 98,
        agg_method: str = 'gmean',
        spatial_in_obsm: str = 'spatial',
):
    """
    Data pre-processing
    """
    # Get spatial information
    df_pixel = adata.obsm[spatial_in_obsm]
    if isinstance(df_pixel, np.ndarray):
        df_pixel = pd.DataFrame(df_pixel)
        df_pixel = df_pixel.iloc[:, [0, 1]]
    row = list(df_pixel.iloc[:, 1] * scale)
    col = list(df_pixel.iloc[:, 0] * scale)

    indices, distances = _graph(
        col, row,
        method=method,
        cutoff=radius,
        max_quantile=max_quantile,
    )

    # lr-pairs information
    lr_names, lr_used = get_used_lr(lr_pairs, key)
    if not isinstance(adata.X, np.ndarray):
        exp = adata.X.toarray()
    else:
        exp = adata.X
    dict_gene = {gene: index for index, gene in enumerate(adata.var_names)}
    if lr_pairs.complexes is not None:
        exp, dict_gene = update_exp(exp,
                                    dict_gene,
                                    lr_pairs.complexes,
                                    agg_method,
                                    lr_pairs.gene_name,
                                    norm,
                                    percent_of_drop,
                                    )

    return indices, distances, lr_names, lr_used, exp.astype(np.float32), dict_gene


def update_exp(
        exp: np.ndarray,
        dict_gene: dict,
        complexes: dict,
        agg_method: str,
        used_gene: list,
        norm: bool,
        percent_of_drop: Union[int, float]
):
    """
    Update expression by combining subunit from database. if there are 'ligand1 : receptor1 and receptor2' in dataset.
     'receptor1 and receptor2' expression will be generated by the amount of 'receptor1' and 'receptor2' expression

    Parameters
    ----------
    exp : np.ndarray
        The expression of all the ligand and receptor in anndata.
    dict_gene : dict
        The dict of ligand and receptor.
    complexes : dict
        The dict of complexes and their subunits.
    agg_method : str
        Method for calculating complex expression. Default: geometric mean.
    used_gene : list
        All the genes that are used.
    norm : bool
        Whether 0-1 normalization was performed on the expression volume data.
    percent_of_drop : Union[int, float]
        Percentage of extreme values removed from normalization

    Returns
    -------
    exp_new
        The expression after the addition of complex
    dict_gene_new
        The dict_gene after the addition of complex

    """
    gene_name = list(dict_gene.keys())
    exp = copy.deepcopy(exp)
    for k, v in complexes.items():
        if all(g in gene_name for g in v):
            dict_gene[k] = len(dict_gene)
            index = [dict_gene[gene] for gene in v]
            if agg_method == 'min':
                update_ = np.min(exp[:, index], axis=1)
            elif agg_method == 'mean':
                update_ = np.mean(exp[:, index], axis=1)
            elif agg_method == 'gmean':
                update_ = gmean(exp[:, index], axis=1)
            else:
                ValueError("{} is not a valid agg_method".format(agg_method))
            exp = np.insert(exp, exp.shape[1], np.array(update_), axis=1)

    dict_gene_new = {}
    i = 0
    index_used = []
    for gene in list(set(used_gene)):
        if gene in dict_gene.keys():
            index_used.append(dict_gene[gene])
            dict_gene_new[gene] = i
            i += 1
    exp_new = exp[:, index_used]

    if norm:
        exp_new = ((exp_new - np.percentile(exp_new, percent_of_drop, axis=0)) /
                  (np.percentile(exp_new, 100 - percent_of_drop, axis=0) - np.percentile(exp_new, percent_of_drop, axis=0)))
        exp_new = np.where(exp_new > 1, 1.0, exp_new)
        exp_new = np.where(exp_new < 0, 0.0, exp_new)
        exp_new = np.nan_to_num(exp_new)

    return exp_new.astype(np.float32), dict_gene_new


def get_sample_lr_score(
        lr_used: list,
        species: str,
        exp: np.ndarray,
        dict_gene: dict,
        gene_used: list,
        indices: list,
        distances: list,
        lr_num: int,
        obs_num: int,
        func: bool,
        n_shuffle: int,
):
    """
    All communication scores for a certain communication type
    """
    if func:
        mode = 'contact'
    else:
        mode = 'secretory'

    without_neighbor = 0
    neigh_num = 0
    i=1
    for i, j in enumerate(indices):
        if not isinstance(j, np.ndarray):
            indices[i] = np.array([], dtype=np.int32)
            distances[i] = np.array([], dtype=np.float32)
            without_neighbor += 1
        else:
            indices[i] = np.array(indices[i], dtype=np.int32)
            distances[i] = np.array(distances[i], dtype=np.float32)
        neigh_num += len(indices[i])
    print(f'In {mode} mode, The average number of neighbors is {neigh_num/i}')
    print(f'In {mode} mode, total of {without_neighbor} spots have no neighbors')

    indices = nb.typed.List(indices)
    distances = nb.typed.List(distances)
    spots_score = np.zeros(shape=(obs_num, lr_num), dtype=np.float32)
    spots_p = np.ones(shape=(obs_num, lr_num), dtype=np.float32) * (n_shuffle + 1)

    with tqdm(
            total=lr_num,
            desc=f"{lr_num} {mode} ligand-receptor pairs.",
            bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:

        for index_lr in prange(lr_num):
            ligand, receptor = str(lr_used[index_lr][0]), str(lr_used[index_lr][1])
            if ligand not in gene_used or receptor not in gene_used:
                spots_score[:, index_lr] = 0
                pbar.update(1)
                continue

            l_index = int(dict_gene[ligand])
            r_index = int(dict_gene[receptor])
            if species == 'ligand':
                exp_index = exp[:, l_index]
                exp_neighbor = exp[:, r_index]
            else:
                exp_index = exp[:, l_index]
                exp_neighbor = exp[:, r_index]

            spots_score[:, index_lr], spots_p[:, index_lr] = get_one_lr_score(
                obs_num=obs_num,
                exp_index=exp_index,
                exp_neighbor=exp_neighbor,
                neighbors=indices,
                distances=distances,
                n=n_shuffle,
                func=func
            )
            pbar.update(1)

    return spots_score, spots_p / (n_shuffle + 1)


@njit()
def get_one_lr_score(
        obs_num: int,
        exp_index: np.ndarray,
        exp_neighbor: np.ndarray,
        neighbors: np.ndarray,
        distances: np.ndarray,
        n: int,
        func: bool,
):
    """
    Compute the cell-cell communication score for a pair of ligand receptors in spot level.
    """
    scores_all_spots = np.zeros(shape=obs_num, dtype=np.float32)
    p_all_spots = np.ones(shape=obs_num, dtype=np.int32) * (n + 1)
    exps_index = np.zeros(shape=(exp_index.shape[0], n + 1), dtype=np.float32)
    exps_neigh = np.zeros(shape=(exp_neighbor.shape[0], n + 1), dtype=np.float32)
    exps_index[:, 0] = exp_index
    exps_neigh[:, 0] = exp_neighbor

    for index in range(n):
        np.random.shuffle(exp_index)
        np.random.shuffle(exp_neighbor)
        exps_index[:, index + 1] = exp_index
        exps_neigh[:, index + 1] = exp_neighbor

    for index_spot in prange(obs_num):
        neighbor = neighbors[index_spot]
        if len(neighbor) == 0:
            continue
        else:
            if func:
                lr_score = np.sum(exps_index[index_spot, :] * exps_neigh[neighbor, :], axis=0, dtype=np.float32)
            else:
                d = distances[index_spot]
                d_reshaped = np.reshape(d + np.float32(1.0), (d.shape[0], 1))
                lr_score = np.sum(exps_index[index_spot, :] * (exps_neigh[neighbor, :] / d_reshaped), axis=0,
                                  dtype=np.float32)
            scores_all_spots[index_spot] = lr_score[0]

            count_shuffle = 0
            score_true = lr_score[0]
            for score in lr_score:
                if score >= score_true:
                    count_shuffle += 1
            p_all_spots[index_spot] = count_shuffle

    return scores_all_spots, p_all_spots


def get_contact_score(
        adata: AnnData,
        lr_pairs,
        species,
        key='contact',
        scale=1.0,
        agg_method='gmean',
        norm: bool = True,
        percent_of_drop: Union[int, float] = 0,
        spatial_in_obsm='spatial',
        max_quantile=95,
        n_iters: int = 1000,
):
    """
    contact type communication score in spot level
    """

    # if type(n_jobs) != type(None):
    #     nb.set_num_threads(n_jobs)

    indices, distances, lr_names, lr_used, exp, dict_gene = _lr_helper(adata=adata,
                                                                       lr_pairs=lr_pairs,
                                                                       scale=scale,
                                                                       key=key,
                                                                       method='neighbor',
                                                                       max_quantile=max_quantile,
                                                                       agg_method=agg_method,
                                                                       norm=norm,
                                                                       percent_of_drop=percent_of_drop,
                                                                       spatial_in_obsm=spatial_in_obsm,
                                                                       )

    print(f'There are {len(lr_names)} cell-cell contact ligand-receptor pairs in sample')

    obs_names = adata.obs_names.tolist()
    lr_score, pvalue = get_sample_lr_score(
        lr_used=lr_used,
        species=species,
        exp=exp,
        dict_gene=dict_gene,
        gene_used=list(dict_gene.keys()),
        indices=indices,
        distances=distances,
        lr_num=len(lr_used),
        obs_num=len(obs_names),
        func=True,
        n_shuffle=n_iters
    )

    return lr_score, pvalue, lr_names


def get_secretory_score(
        adata: AnnData,
        lr_pairs,
        species,
        key='secretory',
        scale=1.0,
        agg_method='gmean',
        norm: bool = True,
        percent_of_drop: Union[int, float] = 0,
        spatial_in_obsm='spatial',
        radius=100,
        n_iters: int = 1000,
):
    """
    secretory type communication score in spot level
    """

    # if type(n_jobs) != type(None):
    #     nb.set_num_threads(n_jobs)

    indices, distances, lr_names, lr_used, exp, dict_gene = _lr_helper(adata=adata,
                                                                       lr_pairs=lr_pairs,
                                                                       scale=scale,
                                                                       key=key,
                                                                       method='radius',
                                                                       radius=radius,
                                                                       max_quantile=None,
                                                                       agg_method=agg_method,
                                                                       norm=norm,
                                                                       percent_of_drop=percent_of_drop,
                                                                       spatial_in_obsm=spatial_in_obsm,
                                                                       )

    print(f'There are {len(lr_names)} secreted ligand-receptor pairs in sample')

    obs_names = adata.obs_names.tolist()
    lr_score, pvalue = get_sample_lr_score(
        lr_used=lr_used,
        species=species,
        exp=exp,
        dict_gene=dict_gene,
        gene_used=list(dict_gene.keys()),
        indices=indices,
        distances=distances,
        lr_num=len(lr_used),
        obs_num=len(obs_names),
        func=False,
        n_shuffle=n_iters
    )

    return lr_score, pvalue, lr_names


def cell_level_communications(
        adata: AnnData,
        lr_pairs: lr_pairs,
        species: Literal['ligand', 'receptor'] = 'ligand',
        agg_method: Literal['min', 'mean', 'gmean'] = 'gmean',
        scale: Union[str, float] = 'hires',
        sample: Union[str, int, list, None] = None,
        sample_key: Optional[str] = None,
        contact_key: str = 'contact',
        contact_max_quantile: int = 95,
        secretory_key: str = 'secretory',
        secretory_radius: Union[int, float] = 100,
        fdr_axis: Literal['spot', 'lr'] = 'spot',
        pval_adj_cutoff: float = 0.05,
        adj_method: str = 'fdr_bh',
        norm: bool = True,
        percent_of_drop: Union[int, float] = 0,
        spatial_in_obsm: str = 'spatial',
        n_iters: int = 1000,
        inplace: bool = True,
) -> anndata.AnnData:
    """
    A permutation test of ligand-receptor expression across every spot.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    lr_pairs : lr_pairs
        database of ligand-receptor pairs.
    species : Literal['ligand', 'receptor'], optional
        'ligand': The central cell is the ligand cell and the neighbor cells are the receptor cells.
        'receptor': The central cell is the receptor cell and the neighbor cells are the ligand cells.
    agg_method : str, optional
        Integrated subunit to calculate ligand or receptor.
        'min': ligand or receptor = min(sub1, sub2, sub3, ...)
        'mean': ligand or receptor = mean(sub1, sub2, sub3, ...)
        'gmean': geometrical mean, ligand or receptor = np.gmean(sub1, sub2, sub3, ...)
    scale : Union[str, float], optional
        scale used in subsequent analyses. If it's Visium data it can also be HE image labels (hires or lower).
        Most of the time you don't need to change this
    sample : Union[str, int, list], optional
        Samples for which communication scores need to be calculated
    sample_key : str, optional
        The keyword of sample id in adata.obs.columns.
    contact_key : str, optional
        The tag name that represents the contact type in the LR database
    contact_max_quantile : int
        In Neighbor network, Order the distance of all sides, and more than max_quantile% of the sides will be removed
    secretory_key : str, optional
        The tag name that represents the secretory type in the LR database
    secretory_radius : Union[int, float]
        The maximum distance considered for the secretory type
    fdr_axis : Literal['spot', 'lr']
        Dimensions of applying multiple hypothesis testing
        'spot': tested in each spot
        'lr': tested in each ligand-receptor pair
        None: no multiple hypothesis testing
    pval_adj_cutoff : float
        Cutoff for spot to be significant based on adjusted p-value.
    adj_method : str
        Any method supported by statsmodels.stats.multitest.multipletests;
        https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
    norm : bool
        Whether 0-1 normalization was performed on the expression volume data.
    percent_of_drop : Union[int, float]
        Percentage of extreme values removed from normalization
    spatial_in_obsm : str
        The key of spatial coordinates in adata.obsm
    n_iters : int
        Number of permutations for the permutation-test
    inplace : bool, optional
        Whether to change the original adata.

    Returns
    -------
    - :attr:`anndata.AnnData.uns` ``['SOAPy']['ligand_spot_comm_score' or 'receptor_spot_comm_score']``
        - 'contact_names': Names of used contact type lR pairs.
        - 'secretory_names': Names of used secretory type lR pairs.
        - 'contact_affinity': The affinity of contact type lR pairs.
        - 'secretory_affinity': The affinity of secretory type lR pairs.

    """
    adata = _check_adata_type(adata, spatial_in_obsm, inplace)

    if isinstance(sample, str):
        sample = [sample]
    if sample is not None and sample_key is None:
        logg.error(f'Mult-sample niche analysis cannot be specified without a given sample key')
        raise ValueError
    if sample is None and sample_key is not None:
        sample = list(np.unique(adata.obs[sample_key]))
        logg.info(f'Use all samples in the niche calculation')
    if sample is None and sample_key is None:
        sample = [None]

    for sample_id in sample:
        if sample_id is None:
            bdata = adata
        else:
            bdata = adata[adata.obs[sample_key] == sample, :]

        if type(scale) != float:
            scale = _scale(bdata, None, scale)

        if contact_key is None:
            lr_score_c, pvalue_c, contact_names = None, None, None
            update_p_data_c = None
        else:
            lr_score_c, pvalue_c, contact_names = get_contact_score(
                adata=bdata,
                lr_pairs=lr_pairs,
                species=species,
                key=contact_key,
                scale=scale,
                agg_method=agg_method,
                spatial_in_obsm=spatial_in_obsm,
                max_quantile=contact_max_quantile,
                norm=norm,
                percent_of_drop=percent_of_drop,
                n_iters=n_iters,
            )
            update_p_data_c = adj_pvals(lr_score_c, pvalue_c, pval_adj_cutoff, fdr_axis, adj_method)

        if secretory_key is None:
            lr_score_s, pvalue_s, secretory_names = None, None, None
            update_p_data_s = None
        else:
            lr_score_s, pvalue_s, secretory_names = get_secretory_score(
                adata=bdata,
                lr_pairs=lr_pairs,
                species=species,
                key=secretory_key,
                scale=scale,
                agg_method=agg_method,
                spatial_in_obsm=spatial_in_obsm,
                radius=secretory_radius,
                norm=norm,
                percent_of_drop=percent_of_drop,
                n_iters=n_iters,
            )
            update_p_data_s = adj_pvals(lr_score_s, pvalue_s, pval_adj_cutoff, fdr_axis, adj_method)

        communication_ = {
            'contact_names': contact_names,
            'secretory_names': secretory_names,
            'contact_affinity': update_p_data_c,
            'secretory_affinity': update_p_data_s,
        }

        _add_info_from_sample(adata, sample_id=sample_id, keys=species+'_cell_comm_score', add=communication_)

    return adata


###################################
###################################
@njit()
def get_celltype_score(
        exp_ligand, exp_receptor, len_celltype, clusters, edges, distances, func, n_iters,
):
    """
    Compute the cell-cell communication score for a pair of ligand receptors in cell type level.
    """

    lr_score = np.zeros(shape=(n_iters, len_celltype, len_celltype), dtype=np.float32)
    lr_p = np.zeros(shape=(len_celltype, len_celltype), dtype=np.float32)
    shuffle_ligand = np.copy(exp_ligand)
    shuffle_receptor = np.copy(exp_receptor)
    # index = np.arange(len(clusters), dtype=np.int32)

    for i in range(n_iters):
        if i != 0:

            clusters_i = np.copy(clusters)
            shuffle_ligand_i = np.zeros_like(shuffle_ligand)
            shuffle_receptor_i = np.zeros_like(shuffle_receptor)

            for j in range(len_celltype):
                index_j = np.random.permutation(np.where(clusters == j)[0])
                shuffle_ligand_i[np.where(clusters == j)[0]] = shuffle_ligand[index_j]
                shuffle_receptor_i[np.where(clusters == j)[0]] = shuffle_receptor[index_j]

        else:
            clusters_i = np.copy(clusters)
            shuffle_ligand_i = np.copy(shuffle_ligand)
            shuffle_receptor_i = np.copy(shuffle_receptor)
        lr_score_i = np.zeros(shape=(len_celltype, len_celltype), dtype=np.float32)

        for index_edge in range(edges.shape[0]):
            spoti, spotj = edges[index_edge, 0], edges[index_edge, 1]
            d = distances[index_edge]
            cti = clusters_i[spoti]
            ctj = clusters_i[spotj]
            if func:
                lr_score_i[cti, ctj] += shuffle_ligand_i[spoti] * shuffle_receptor_i[spotj]
                lr_score_i[ctj, cti] += shuffle_ligand_i[spotj] * shuffle_receptor_i[spoti]
            else:
                lr_score_i[cti, ctj] += shuffle_ligand_i[spoti] * shuffle_receptor_i[spotj] / (
                            d + np.float32(1.0))
                lr_score_i[ctj, cti] += shuffle_ligand_i[spotj] * shuffle_receptor_i[spoti] / (
                            d + np.float32(1.0))

        lr_score[i, :, :] = lr_score_i

    lr_score_truth = lr_score[0, :, :]
    for i in range(n_iters):
        p = np.where(lr_score[i, :, :] >= lr_score_truth, 1, 0)
        lr_p = lr_p + p

    return lr_p, lr_score_truth


def lr_score_with_cluster(
        adata: anndata.AnnData,
        lr_pairs: lr_pairs,
        clusters: list,
        celltype: list,
        scale: float,
        method: str,
        radius: Union[int, float] = None,
        max_quantile: Union[int, float] = None,
        k: Union[int, float] = 1,
        m: Union[int, float] = 1,
        key: str = None,
        agg_method: str = 'gmean',
        norm: bool = True,
        percent_of_drop: Union[int, float] = 0,
        spatial_in_obsm: str = 'spatial',
        n_iters: int = 1000,
        func: bool = True,
) -> [np.ndarray, np.ndarray, list]:
    if func:
        mode = 'contact'
    else:
        mode = 'secretory'

    # Get spatial information
    indices, distances, lr_names, lr_used, exp, dict_gene = _lr_helper(adata=adata,
                                                                       lr_pairs=lr_pairs,
                                                                       scale=scale,
                                                                       key=key,
                                                                       method=method,
                                                                       radius=radius,
                                                                       max_quantile=max_quantile,
                                                                       agg_method=agg_method,
                                                                       norm=norm,
                                                                       percent_of_drop=percent_of_drop,
                                                                       spatial_in_obsm=spatial_in_obsm,
                                                                       )

    without_neighbor = 0
    neigh_num = 0
    i = 0
    for i, j in enumerate(indices):
        neigh_num += len(indices[i])

    print(f'In {mode} mode, The average number of neighbors is {neigh_num/i}')
    print(f'In {mode} mode, total of {without_neighbor} spots have no neighbors')

    for index, name in enumerate(celltype):
        index = np.int32(index)
        clusters = [index if i == name else i for i in clusters]
    clusters = np.array(clusters, dtype=np.int32)

    n_ct = np.zeros(shape=len(celltype), dtype=np.int32)
    for index_i in range(len(celltype)):
        n_ct[index_i] = np.sum(clusters == index_i)

    pd_edges = _preprocessing_of_graph(
        clu_value=clusters,
        indices=indices,
        distances=distances
    )
    count_edges = _count_edge(edge=pd_edges,
                              species_of_clusters=len(celltype),
                              cell_type_dict={ct: ct for ct in range(len(celltype))})

    count_diff_edges = np.sum(count_edges, axis=0) - count_edges.diagonal()
    expectation_edges = allocation_edge_2_diff_cell_type(count_diff_edges)

    edges = np.zeros(shape=(pd_edges.shape[0], 2))
    edges[:, 0] = pd_edges['point_1']
    edges[:, 1] = pd_edges['point_2']
    edges = edges.astype(np.int32)
    distances = pd_edges['distance'].values

    lr_names_new = []
    lr_used_new = []
    for index, name in enumerate(lr_names):
        gene_in_lr = re.split(':|&', name)
        if all(gene in adata.var_names for gene in gene_in_lr):
            lr_names_new.append(name)
            lr_used_new.append(lr_used[index])

    lr_ct_strength = np.zeros(shape=(len(lr_names_new), len(celltype), len(celltype)), dtype=np.float32)
    lr_ct_affinity = np.ones(shape=(len(lr_names_new), len(celltype), len(celltype)), dtype=np.float32) * (n_iters+1)

    with tqdm(
            total=len(lr_names_new),
            desc=f"{len(lr_names_new)} {mode} ligand-receptor pairs.",
            bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:

        for index, name in enumerate(lr_names_new):
            l, r = lr_used_new[index]

            lr_ct_affinity[index, :, :], lr_ct_strength[index, :, :] = get_celltype_score(
                exp_ligand=np.copy(exp[:, dict_gene[l]]),
                exp_receptor=np.copy(exp[:, dict_gene[r]]),
                len_celltype=len(celltype),
                clusters=clusters,
                edges=edges,
                distances=distances,
                func=func,
                n_iters=n_iters
            )

            l_all_mean = np.mean(exp[:, dict_gene[l]])
            r_all_mean = np.mean(exp[:, dict_gene[r]])

            for index_i in range(len(celltype)):
                for index_j in range(len(celltype)):
                    l_ct1 = np.mean(exp[np.where(clusters == index_i), dict_gene[l]])
                    r_ct2 = np.mean(exp[np.where(clusters == index_j), dict_gene[r]])

                    exp_score = np.power((l_ct1 / l_all_mean) * (r_ct2 / r_all_mean), k)
                    exp_score = np.nan_to_num(exp_score)
                    if index_i == index_j:
                        lr_ct_strength[index, index_i, index_j] = exp_score
                    else:
                        count_edge_ij = count_edges[index_i, index_j]
                        count_score = np.power(count_edge_ij/expectation_edges[index_i, index_j], m)
                        count_score = 2*count_score / (count_score + np.float32(1.0))
                        lr_ct_strength[index, index_i, index_j] = exp_score*count_score

            pbar.update(1)

    lr_ct_affinity = lr_ct_affinity / n_iters

    return [lr_ct_strength, lr_ct_affinity, lr_names_new]


def select_communication_in_celltype(
        affinity: np.ndarray,
        strength: np.ndarray,
        affinity_cutoff: float,
        strength_cutoff: float,
        celltype: list
) -> pd.DataFrame:
    """
    Count the cell-cell communication between cell types with affinity(p) and strength(s).
    """
    if affinity_cutoff is not None:
        communication_intensity_a = np.where(affinity < affinity_cutoff, 1, 0)
    else:
        communication_intensity_a = np.ones_like(affinity)

    if strength_cutoff is not None:
        communication_intensity_s = np.where(strength > strength_cutoff, 1, 0)
    else:
        communication_intensity_s = np.ones_like(strength)

    communication_intensity = communication_intensity_s & communication_intensity_a
    communication_intensity = np.sum(communication_intensity, axis=0)
    communication_intensity = pd.DataFrame(communication_intensity, index=celltype, columns=celltype)
    return communication_intensity


def cell_type_level_communication(
        adata: AnnData,
        lr_pairs: lr_pairs,
        cluster_key: str = 'cluster',
        agg_method: str = 'gmean',
        scale: Union[str, float] = 'hires',
        sample: Union[str, int, list, None] = None,
        sample_key: Optional[str] = None,
        affinity_cutoff: Optional[float] = 0.05,
        strength_cutoff: Optional[float] = 2.0,
        contact_key: Optional[str] = 'contact',
        contact_max_quantile: int = 95,
        secretory_key: Optional[str] = 'secretory',
        secretory_radius: Union[int, float] = 100,
        k: Union[str, float] = 1,
        m: Union[str, float] = 1,
        norm: bool = True,
        percent_of_drop: Union[int, float] = 0,
        spatial_in_obsm: str = 'spatial',
        n_iters: int = 1000,
        inplace: bool = True,
) -> anndata.AnnData:
    """
    Cell type ligand-receptor algorithm composed of two indexes: affinity and strength.
    The affinity is calculated from the p-value of the permutation-test.
    The strength is a combination score of the ratio of the expression and the number of edges to the expectation.

    Parameters
    ----------
    adata : anndata.Anndata
        An AnnData object containing spatial omics data and spatial information.
    lr_pairs : lr_pairs
        database of ligand-receptor pairs
    cluster_key : str
        The label of cluster in adata.obs.
    agg_method : str
        Integrated subunit to calculate ligand or receptor.
        'min': ligand or receptor = min(sub1, sub2, sub3, ...)
        'mean': ligand or receptor = mean(sub1, sub2, sub3, ...)
        'gmean': geometrical mean, ligand or receptor = np.gmean(sub1, sub2, sub3, ...)
    scale : Union[str, float]
        scale used in subsequent analyses. If it's Visium data it can also be HE image labels (hires or lower).
        Most of the time you don't need to change this.
    sample : Union[str, int, list], optional
        Samples for which communication scores need to be calculated.
    sample_key : str, optional
        The keyword of sample id in adata.obs.columns.
    affinity_cutoff : float
        The cutoff of affinity.
    strength_cutoff : Optional[float]
        The cutoff of strength.
    contact_key : str, optional
        The tag name that represents the contact type in the LR database
    contact_max_quantile : int
        In Neighbor network, Order the distance of all sides, and more than max_quantile% of the sides will be removed
    secretory_key : str, optional
        The tag name that represents the secretory type in the LR database
    secretory_radius : Union[int, float]
        The maximum distance considered for the secretory type
    k: Union[str, float]
        The weight of the expression function
    m: Union[str, float]
        The weight of the edge function
    norm : bool
        Whether 0-1 normalization was performed on the expression volume data.
    percent_of_drop : Union[int, float]
        Percentage of extreme values removed from normalization
    spatial_in_obsm : str
        The key of spatial coordinates in adata.obsm
    n_iters : int
        Number of permutations for the permutation-test
    inplace : bool, optional
        Whether to change the original adata.

    Returns
    -------
    - :attr:`anndata.AnnData.uns` ``['SOAPy']['ligand_spot_comm_score' or 'receptor_spot_comm_score']``
        - 'celltype': cell type in used sample.
        - 'contact': cell-cell communication between cell types under contact type LR.
            - 'names': Names of used contact type lR pairs.
            - 'sig_celltype': How cell types communicate with each other under contact conditions.
            - 'strength': The strength of contact type lR pairs.
            - 'affinity': The affinity of contact type lR pairs.
        - 'secretory': cell-cell communication between cell types under secretory type LR.
            - 'names': Names of used secretory type lR pairs.
            - 'sig_celltype': How cell types communicate with each other under secretory conditions.
            - 'strength': The strength of secretory type lR pairs.
            - 'affinity': The affinity of secretory type lR pairs.

    """
    adata = _check_adata_type(adata, spatial_in_obsm, inplace)

    if isinstance(sample, str):
        sample = [sample]
    if sample is not None and sample_key is None:
        logg.error(f'Mult-sample niche analysis cannot be specified without a given sample key')
        raise ValueError
    if sample is None and sample_key is not None:
        sample = list(np.unique(adata.obs[sample_key]))
        logg.info(f'Use all samples in the niche calculation')
    if sample is None and sample_key is None:
        sample = [None]

    for sample_id in sample:
        if sample_id is None:
            bdata = adata
        else:
            bdata = adata[adata.obs[sample_key] == sample, :].copy()

        if type(scale) != float:
            scale = _scale(bdata, None, scale)

        clusters = adata.obs[cluster_key].tolist()
        celltype = list(np.unique(clusters))

        if contact_key is None:
            c_sig_celltype = None
            c_name = None
            contact_strength_ct = None
            contact_affinity_ct = None
        else:
            contact_strength_ct, contact_affinity_ct, c_name = lr_score_with_cluster(
                adata=bdata,
                clusters=clusters,
                celltype=celltype,
                scale=scale,
                method='neighbor',
                k=k,
                m=m,
                max_quantile=contact_max_quantile,
                lr_pairs=lr_pairs,
                key=contact_key,
                agg_method=agg_method,
                norm=norm,
                percent_of_drop=percent_of_drop,
                spatial_in_obsm=spatial_in_obsm,
                n_iters=n_iters,
                func=True
            )

            c_sig_celltype = select_communication_in_celltype(affinity=contact_affinity_ct,
                                                              strength=contact_strength_ct,
                                                              affinity_cutoff=affinity_cutoff,
                                                              strength_cutoff=strength_cutoff,
                                                              celltype=list(celltype))

        if secretory_key is None:
            s_sig_celltype = None
            s_name = None
            secretory_strength_ct = None
            secretory_affinity_ct = None
        else:
            secretory_strength_ct, secretory_affinity_ct, s_name = lr_score_with_cluster(
                adata=bdata,
                clusters=clusters,
                celltype=celltype,
                scale=scale,
                method='radius',
                radius=secretory_radius,
                k=k,
                m=m,
                lr_pairs=lr_pairs,
                key=secretory_key,
                agg_method=agg_method,
                norm=norm,
                percent_of_drop=percent_of_drop,
                spatial_in_obsm=spatial_in_obsm,
                n_iters=n_iters,
                func=False
            )

            s_sig_celltype = select_communication_in_celltype(affinity=secretory_affinity_ct,
                                                              strength=secretory_strength_ct,
                                                              affinity_cutoff=affinity_cutoff,
                                                              strength_cutoff=strength_cutoff,
                                                              celltype=list(celltype))

        communication_ = {
            'celltype': celltype,
            'contact': {
                'names': c_name,
                'sig_celltype': c_sig_celltype,
                'strength': contact_strength_ct,
                'affinity': contact_affinity_ct
            },
            'secretory': {
                'names': s_name,
                'sig_celltype': s_sig_celltype,
                'strength': secretory_strength_ct,
                'affinity': secretory_affinity_ct
            }
        }
        _add_info_from_sample(adata, sample_id=sample_id, keys='celltype_comm_score', add=communication_)

    return adata
