import numpy as np
import numba as nb
import pandas as pd
from anndata import AnnData
from typing import Union, Optional, Tuple


@nb.jit
def _filter_of_graph(obs: pd.DataFrame,
                     indices: np.ndarray,
                     distances: np.ndarray,
                     cluster_label: str,
                     exclude: Union[str, dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the edges as 'exclude'

    Returns
    -------
        anndata.Anndata object.

    """
    if type(exclude) == str:
        assert exclude in ['same', 'different']
    if type(exclude) == dict:
        exclude = [(i, j) for i, j in zip(exclude.keys(), exclude.values())] + \
                  [(j, i) for i, j in zip(exclude.keys(), exclude.values())]

    for i, neigh in enumerate(indices):
        point_1 = neigh[0]
        index_point1 = obs.index[point_1]
        for j, point_2 in enumerate(neigh):
            if j == 0:
                continue
            index_point2 = obs.index[point_2]
            if exclude == 'same' and \
                    obs.loc[index_point1, cluster_label] == obs.loc[index_point2, cluster_label]:
                distances[i][j] = -1
                indices[i][j] = -1
                continue
            if exclude == 'different' and \
                    obs.loc[index_point1, cluster_label] != obs.loc[index_point2, cluster_label]:
                distances[i][j] = -1
                indices[i][j] = -1
                continue
            if type(exclude) == dict and \
                    (obs.loc[index_point1, cluster_label], obs.loc[index_point2, cluster_label]) in exclude:
                distances[i][j] = -1
                indices[i][j] = -1
        distances[i] = distances[i][distances[i] >= 0]
        indices[i] = indices[i][indices[i] >= 0]

    return indices, distances


def _preprocessing_of_graph(clu_value: np.ndarray,
                            indices: np.ndarray,
                            distances: np.ndarray,
                            ) -> pd.DataFrame:
    """
    get information of edges
    """
    edges = []

    for index, neigh in enumerate(indices):
        if len(neigh) == 0:
            continue
        for index_2, point_2 in enumerate(neigh):
            if point_2 == index | point_2 == -1:
                continue
            elif point_2 < index:
                edge = [point_2, index,
                        clu_value[point_2],
                        clu_value[index],
                        distances[index][index_2]]
                edges.append(edge)

            elif point_2 > index:
                edge = [index, point_2,
                        clu_value[index],
                        clu_value[point_2],
                        distances[index][index_2]]
                edges.append(edge)

    df_edge = pd.DataFrame(data=np.array(edges), columns=['point_1', 'point_2',
                                                          'cluster_1', 'cluster_2', 'distance'])

    df_edge.drop_duplicates(subset=['point_1', 'point_2'], inplace=True)
    return df_edge
