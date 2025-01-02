import os
import numpy as np
import pandas as pd
from typing import Optional, Union
from anndata import AnnData
import numba as nb
import logging as logg


def _assert_variable_in_list(
        variable: str,
        target_list,
        list_name: str,
) -> None:
    if variable not in target_list:
        raise KeyError(f"Spatial basis `{variable}` not found in `{list_name}`.")


def set_R_environment(
        R_HOME: Optional[str] = None,
        R_USER: Optional[str] = None
) -> None:
    """
    Setting R's environment

    Parameters
    ----------
    R_HOME
        PathLike, the location of R (if 'cluster' is m_cluster).
    R_USER
        PathLike, the location of R (if 'cluster' is m_cluster).
    """

    if R_HOME is not None:
        os.environ['R_HOME'] = R_HOME
    if R_USER is not None:
        os.environ['R_USER'] = R_USER


def _best_k(data, k, sdbw):
    """
    The optimal number of clusters is selected by sdbw index
    """
    from sklearn.cluster import KMeans
    from s_dbw import S_Dbw

    data = data.astype(np.float32)

    if sdbw and k != 2:
        s_Dbw = {}
        for k_i in range(2, k+1):
            if data.shape[0] <= k_i:
                break
            km = KMeans(n_clusters=k_i)
            clusters_res = km.fit_predict(data)
            s_Dbw[k_i] = S_Dbw(data, clusters_res)

        min_k = list(s_Dbw.keys())[list(s_Dbw.values()).index(min(list(s_Dbw.values())))]
    else:
        min_k = k

    km = KMeans(n_clusters=min_k)
    res = km.fit_predict(data)
    logg.info(f'The k in cluster is {min_k}', exc_info=True)
    return res, min_k


@nb.jit()
def _filter_of_graph(adata: AnnData,
                     cluster_label: str,
                     exclude: Union[str, dict]) -> AnnData:

    if type(exclude) == str:
        assert exclude in ['same', 'different']
    if type(exclude) == dict:
        exclude = [(i, j) for i in exclude.keys()
                   for j in exclude.values()] + \
                  [(i, j) for j in exclude.keys()
                   for i in exclude.values()]

    distances, neighborhoods = adata.uns['SOAPy']['distance'], adata.uns['SOAPy']['indices']

    for i, neigh in enumerate(neighborhoods):
        point_1 = neigh[0]
        for j, point_2 in enumerate(neigh):
            index_point1 = adata.obs.index[point_1]
            index_point2 = adata.obs.index[point_2]
            if exclude == 'same' and \
                    adata.obs.loc[index_point1, cluster_label] == adata.obs.loc[index_point2, cluster_label]:
                distances[i, j] = -1
                neighborhoods[i, j] = -1
            if exclude == 'different' and \
                    adata.obs.loc[index_point1, cluster_label] != adata.obs.loc[index_point2, cluster_label]:
                distances[i, j] = -1
                neighborhoods[i, j] = -1
            if (adata.obs.loc[index_point1, cluster_label], adata.obs.loc[index_point2, cluster_label]) in exclude:
                distances[i, j] = -1
                neighborhoods[i, j] = -1

    adata.uns['SOAPy']['distance'], adata.uns['SOAPy']['indices'] = distances, neighborhoods
    return adata


# @nb.jit
def _preprocessing_of_graph(adata: AnnData,
                            cluster_label: str,
                            ) -> pd.DataFrame:
    """

    Parameters
    ----------
    adata
    cluster_label

    Returns
    -------

    """

    distances, neighborhoods = adata.uns['SOAPy']['distance'], adata.uns['SOAPy']['indices']
    edges = []

    obs = adata.obs[cluster_label]
    obs_value = obs.values
    for neigh in neighborhoods:
        if len(neigh) == 0:
            continue
        point_1 = neigh[0]
        for point_2 in neigh:
            if point_2 == point_1 | point_2 == -1:
                continue
            elif point_2 < point_1:
                edge = [point_2, point_1,
                        obs_value[point_2],
                        obs_value[point_1]]
                edges.append(edge)

            elif point_2 > point_1:
                edge = [point_1, point_2,
                        obs_value[point_1],
                        obs_value[point_2]]
                edges.append(edge)

    df_edge = pd.DataFrame(data=np.array(edges), columns=['point_1', 'point_2',
                                                          'cluster_1', 'cluster_2'])

    df_edge.drop_duplicates(subset=['point_1', 'point_2'], inplace=True)
    return df_edge


def _count_edge(edge: pd.DataFrame,
                species_of_clusters: int,
                cell_type_dict: dict,
                ) -> np.ndarray:
    """

    Parameters
    ----------
    edge
    species_of_clusters
    cell_type_dict

    Returns
    -------

    """
    enrich_metrics = np.zeros((species_of_clusters, species_of_clusters), dtype=np.float32)

    for row in edge.itertuples():
        enrich_metrics[cell_type_dict[getattr(row, 'cluster_1')], cell_type_dict[getattr(row, 'cluster_2')]] += 1
        enrich_metrics[cell_type_dict[getattr(row, 'cluster_2')], cell_type_dict[getattr(row, 'cluster_1')]] += 1

    return enrich_metrics


def _randomize_helper(adata,
                      cluster_label,
                      species_of_clusters: int,
                      cell_type_dict: dict,
                      ):
    """

    Parameters
    ----------
    adata
    cluster_label
    species_of_clusters
    cell_type_dict

    Returns
    -------

    """
    Series_cluster = adata.obs[cluster_label]
    iter_cluster = Series_cluster.sample(frac=1.0,
                                         ).reset_index(drop=True)
    adata.obs[cluster_label] = iter_cluster.tolist()
    iter_edge = _preprocessing_of_graph(adata,
                                        cluster_label=cluster_label
                                        )

    enrichment = _count_edge(iter_edge, species_of_clusters, cell_type_dict)
    return enrichment


class Iterators:
    def __init__(self,
                 n_iter: int,
                 *args
                 ):
        self.stop = n_iter
        self.params = []
        for param in args:
            self.params.append(param)

    def __iter__(self):
        self.k = 0
        return self

    def __next__(self):
        if self.k < self.stop:
            self.k += 1
            return self.params
        else:
            raise StopIteration


def insert_to_uns(adata, data, *args):
    pointer = adata.uns
    last_index = args[-1]
    args = args[:-1]
    for index in args:
        try:
            pointer[index]
        except KeyError:
            pointer[index] = {}
        pointer = pointer[index]
    pointer[last_index] = data


def adj_pvals(
    data,
    pvalues_data,
    pval_adj_cutoff: float = 0.05,
    correct_axis: str = 'spot',
    adj_method: str = "fdr_bh",
):
    """
    Performs p-value adjustment and determination of significant spots.
    """
    from statsmodels.stats.multitest import multipletests

    scores = data
    sig_scores = scores.copy()
    ps = pvalues_data
    padjs = np.ones(ps.shape)
    if correct_axis == 'spot':
        for spot_i in range(ps.shape[0]):
            lr_indices = np.where(ps[spot_i, :] != 1)[0]
            if len(lr_indices) > 0:
                spot_ps = ps[spot_i, lr_indices]
                spot_padjs = multipletests(spot_ps, method=adj_method)[1]
                padjs[spot_i, lr_indices] = spot_padjs
                sig_scores[spot_i, lr_indices[spot_padjs >= pval_adj_cutoff]] = 0
    elif correct_axis == 'lr':
        for lr_i in range(ps.shape[1]):
            spot_indices = np.where(ps[:, lr_i] != 1)[0]
            if len(spot_indices) > 0:
                lr_ps = ps[spot_indices, lr_i]
                spot_padjs = multipletests(lr_ps, method=adj_method)[1]
                padjs[spot_indices, lr_i] = spot_padjs
                sig_scores[spot_indices[spot_padjs >= pval_adj_cutoff], lr_i] = 0
    else:
        raise Exception(
            f"Invalid correct_axis input, the p value data only have two dimensions"
        )

    # Counting spots significant per lr #
    lr_counts = (padjs < pval_adj_cutoff).sum(axis=0)

    # Re-ranking LRs based on these counts & updating LR ordering #
    new_order = np.argsort(-lr_counts)
    print(f"Updated adata.uns[lr_summary]")
    scores_ordered = scores[:, new_order]
    sig_scores_ordered = sig_scores[:, new_order]
    ps_ordered = ps[:, new_order]
    padjs_ordered = padjs[:, new_order]

    update_p_data = {}
    keys = ["lr_scores", "lr_sig_scores", "p_vals", "p_adjs"]
    values = [scores_ordered, sig_scores_ordered, ps_ordered, padjs_ordered]
    for i in range(len(keys)):
        update_p_data[keys[i]] = values[i]

    return update_p_data


def allocation_edge_2_diff_cell_type(edges, k_max=10000, error=0.001):
    length = len(edges)
    matrix = np.outer(edges, edges) / np.sum(edges)

    for k in range(k_max):
        matrix_new = np.zeros(shape=(length, length), dtype=np.float32)
        sum_ = matrix.trace()

        for i in range(length):
            for j in range(length):
                if i == j:
                    matrix_new[i, j] = matrix[i, j] * matrix[i, j] / sum_
                else:
                    matrix_new[i, j] = matrix[i, j] + (matrix[i, i] * matrix[j, j]) / sum_

        if np.linalg.norm(matrix_new.diagonal()) <= error:
            matrix = matrix_new
            break
        matrix = matrix_new

    if np.linalg.norm(matrix.diagonal()) > error:
        residue_edge = np.max(matrix.diagonal())
        residue_index = np.argmax(matrix.diagonal())
        for i in range(length):
            if i == residue_index:
                matrix[residue_index, residue_index] = 0
            else:
                matrix[residue_index, i] = matrix[i, residue_index] = \
                    residue_edge * matrix[i, residue_index] / (edges[residue_index] - residue_edge)

    return matrix