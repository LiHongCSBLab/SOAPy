"""
Cell communication analysis visualization related code.
Some source code is from stlearn.plotting.cci_plot (),we made some changes to suit our data structure.
"""
import copy
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.colors import Colormap
import seaborn as sns
from scipy.stats.mstats import gmean
import networkx as nx
from anndata import AnnData
import math
from .utils import get_cmap, get_colors
from ._color import _get_palette
from ._chord import chordDiagram
from ._heatmap import _dotplot
from ..utils import _get_info_from_sample
import sys
from typing import Optional, Union, Tuple, Any, Literal
import logging

__all__ = [
    'show_ccc_chordplot',
    'show_ccc_netplot',
    'show_ccc_dotplot',
    'show_ccc_embedding'
]


def _get_lr_data(adata, sample_id, lr_type):

    data = _get_info_from_sample(adata=adata, sample_id=sample_id, key='celltype_comm_score')
    score = data[lr_type]['sig_celltype']
    title = f'cell-cell {lr_type} communication'

    return score, title


def _get_lr_matrix(
        adata: anndata.AnnData,
        lr_type: str,
        sample_id: Optional[str] = None,
        affinity_cutoff: Optional[float] = None,
        strength_cutoff: Optional[float] = None,
        lrs_name: Optional[list] = None,
        cts_name: Optional[list] = None,
        n_top_lrs: Optional[int] = None,
        n_top_cts: Optional[int] = None,
):

    def _3dto2d(data, len_ct, lr_pairs, cts):
        scores = np.zeros(shape=(len(lr_pairs), len_ct * len_ct))
        for index, name in enumerate(lr_pairs):
            p = data[index, :, :]
            k = 0
            for i in range(len_ct):
                for j in range(len_ct):
                    scores[index, k] = p[i, j]
                    k += 1

        scores = pd.DataFrame(scores, index=lr_pairs, columns=cts)
        return scores

    def _best_lr(
            strength: pd.DataFrame,
            affinity: pd.DataFrame,
            lrs_name: Optional[list] = None,
            cts_name: Optional[list] = None,
            affinity_cutoff: Optional[float] = None,
            strength_cutoff: Optional[float] = None,
            n_top_lrs: Optional[int] = None,
            n_top_cts: Optional[int] = None
        ):

        communication_intensity_p = np.where(affinity < affinity_cutoff, 1, 0)
        communication_intensity_s = np.where(strength > strength_cutoff, 1, 0)
        significance = communication_intensity_s & communication_intensity_p

        if lrs_name is None:
            sum_index_lr = np.sum(significance, axis=1)
            sort_index_lr = np.argsort(-sum_index_lr)
            if n_top_lrs is None or n_top_lrs > len(sort_index_lr):
                n_top_lrs = len(sort_index_lr)
            strength = strength.iloc[sort_index_lr[0: n_top_lrs], :]
            affinity = affinity.iloc[sort_index_lr[0: n_top_lrs], :]
        else:
            strength = strength.loc[lrs_name, :]
            affinity = affinity.loc[lrs_name, :]

        if cts_name is None:
            sum_index_cts = np.sum(significance, axis=0)
            sort_index_cts = np.argsort(-sum_index_cts)
            if n_top_cts is None or n_top_cts > len(sort_index_cts):
                n_top_cts = len(sort_index_cts)
            strength = strength.iloc[:, sort_index_cts[0:n_top_cts]]
            affinity = affinity.iloc[:, sort_index_cts[0:n_top_cts]]
        else:
            strength = strength.loc[:, cts_name]
            affinity = affinity.loc[:, cts_name]

        return strength, affinity

    data = _get_info_from_sample(adata=adata, sample_id=sample_id, key='celltype_comm_score')
    ct = data['celltype']
    if lr_type == 'contact':
        affinity = data['contact']['affinity']
        strength = data['contact']['strength']
        pairs = data['contact']['names']
    else:
        affinity = data['secretory']['affinity']
        strength = data['secretory']['strength']
        pairs = data['secretory']['names']

    cts = []
    for i in ct:
        for j in ct:
            cts.append(i + ',' + j)

    strength = _3dto2d(strength, len(ct), pairs, cts)
    affinity = _3dto2d(affinity, len(ct), pairs, cts)

    strength, affinity = _best_lr(strength=strength,
                                  affinity=affinity,
                                  affinity_cutoff=affinity_cutoff,
                                  lrs_name=lrs_name,
                                  cts_name=cts_name,
                                  strength_cutoff=strength_cutoff,
                                  n_top_lrs=n_top_lrs,
                                  n_top_cts=n_top_cts
                                  )

    return strength, affinity


def show_ccc_netplot(
        adata: AnnData,
        sample_id: Union[int, str] = None,
        lr_type: Literal['contact', 'secretory'] = 'contact',
        pos: dict = None,
        cmap: Optional[str] = None,
        font_size: int = 12,
        node_size_exp: int = 1,
        node_size_scaler: int = 1,
        min_counts: int = 0,
        ax: Axes = None,
        figsize: tuple = (8, 8),
        dpi: int = 100,
        show: bool = True,
        save: Optional[str] = None,
        **kwargs
) -> Optional[Axes]:
    """
    Visualize ligand-receptor interactions between two cell types in spatial omics data using network plot.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    figsize : Tuple[float, float], optional
        (Width, height) of the figure.
    dpi : float, optional
        The resolution of the figure.
    title : str, optional
        The title of shown figure.
    ax : Axes
        A matplotlib axes object.
    show : bool
        Show the plot, do not return axis.
    save : Union[str, PathLike], optional
        The path where the image is stored.
    lr_type: str
        The LR pair to visualise the cci network for. If None, will use spot
        cci counts across all LR pairs from adata.uns[f'lr_cci_use_label'].
    pos: dict
        Positions to draw each cell type, format as outputted from running
        networkx.circular_layout(graph). If not inputted will be generated.
    cmap: str
        Cmap to use when generating the cell colors, if not already specified by
        adata.uns[f'use_label_colors'].
    font_size: int
        Size of the cell type labels.
    node_size_scaler: float
        Scaler to multiply by node sizes to increase/decrease size.
    node_size_exp: int
        Increases difference between node sizes by this exponent.
    min_counts: int
        Minimum no. of LR interactions for connection to be drawn.
    kwargs : Any
        Other params of nx.draw_networkx()

    Returns
    -------
    pos: dict
        Dictionary of positions where the nodes are draw if return_pos is True, useful for consistent layouts.

    References
    ----------
    Pham, D. et al. Robust mapping of spatiotemporal trajectories and cell–cell interactions in healthy and diseased
        tissues. Nat Commun 14, 7739 (2023).

    """

    # Either plotting overall interactions, or just for a particular LR #
    int_df, title = _get_lr_data(adata, sample_id, lr_type)
    # Creating the interaction graph #
    all_set = int_df.index.values
    int_matrix = int_df.values
    graph = nx.MultiDiGraph()
    int_bool = int_matrix > min_counts
    int_matrix = int_matrix * int_bool
    for i, cell_A in enumerate(all_set):
        if cell_A not in graph:
            graph.add_node(cell_A)
        for j, cell_B in enumerate(all_set):
            if int_bool[i, j]:
                count = int_matrix[i, j]
                graph.add_edge(cell_A, cell_B, weight=count)

    # Determining graph layout, node sizes, & edge colours #
    if pos is None:
        pos = nx.circular_layout(graph)
    total = sum(sum(int_matrix))
    node_names = list(graph.nodes.keys())
    node_indices = [np.where(all_set == node_name)[0][0] for node_name in node_names]
    node_sizes = np.array(
        [
            (
                ((sum(int_matrix[i, :] + int_matrix[:, i]) - int_matrix[i, i]) / total)
                * 10000
                * node_size_scaler
            )
            ** (node_size_exp)
            for i in node_indices
        ]
    )
    node_sizes[node_sizes == 0] = 0.1

    edges = list(graph.edges.items())
    e_totals = []
    for i, edge in enumerate(edges):
        trans_i = np.where(all_set == edge[0][0])[0][0]
        receive_i = np.where(all_set == edge[0][1])[0][0]
        e_total = (
            sum(list(int_matrix[trans_i, :]) + list(int_matrix[:, receive_i]))
            - int_matrix[trans_i, receive_i]
        )
        e_totals.append(e_total)
    edge_weights = [edge[1]["weight"] / e_totals[i] for i, edge in enumerate(edges)]

    # Determining node colors #
    nodes = np.unique(list(graph.nodes.keys()))
    node_colors = _get_palette(nodes, palette=cmap)
    node_colors = list(node_colors.values())
    if not np.all(np.array(node_names) == nodes):
        nodes_indices = [np.where(nodes == node)[0][0] for node in node_names]
        node_colors = np.array(node_colors)[nodes_indices]

    #### Drawing the graph #####
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    # Adding in the self-loops #
    z = 55
    for i, edge in enumerate(edges):
        cell_type = edge[0][0]
        if cell_type != edge[0][1]:
            continue
        x, y = pos[cell_type]
        angle = math.degrees(math.atan(y / x))
        if x > 0:
            angle = angle + 180
        arc = patches.Arc(
            xy=(x, y),
            width=0.3,
            height=0.025,
            lw=5,
            ec=plt.cm.get_cmap("Blues")(edge_weights[i]),
            angle=angle,
            theta1=z,
            theta2=360 - z,
        )
        ax.add_patch(arc)

    # Drawing the main components of the graph #
    nx.draw_networkx(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        arrowstyle="->",
        arrowsize=50,
        width=5,
        font_size=font_size,
        font_weight="bold",
        edge_color=edge_weights,
        ax=ax,
        **kwargs
    )
    fig.suptitle(title, fontsize=30)
    plt.tight_layout()

    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()
    else:
        return ax


def show_ccc_chordplot(
        adata: AnnData,
        sample: Union[str, list] = None,
        lr_type: Literal['contact', 'secretory'] = 'contact',
        min_ints: int = 2,
        n_top_ccis: int = 10,
        cmap: str = None,
        label_size: int = 10,
        label_rotation: float = 0,
        ax: Axes = None,
        figsize: tuple = (8, 8),
        dpi: int = 100,
        show: bool = True,
        save: Optional[str] = None,
) -> Optional[Axes]:
    """
    Visualize ligand-receptor interactions between two cell types in spatial omics data using chord plot.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    figsize : Tuple[float, float], optional
        (Width, height) of the figure.
    dpi : float, optional
        The resolution of the figure.
    title : str, optional
        The title of shown figure.
    ax : Axes
        A matplotlib axes object.
    show : bool
        Show the plot, do not return axis.
    save : Union[str, PathLike], optional
        The path where the image is stored.
    sample : Union[str, list]
        Name of sample.
    lr_type : Literal['contact', 'secretory']
        Types of ligand-receptor pairs.
    min_ints : int
        Minimum number of interactions for a connection to be considered.
    n_top_ccis : int
        Maximum number of top ligand-receptor pairs to display.
    cmap : str
        Colormap for visualizing cell types.
    label_size : int
        Font size for node labels.
    label_rotation : float
        Rotation angle for node labels.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the chord plot.

    References
    ----------
    Pham, D. et al. Robust mapping of spatiotemporal trajectories and cell–cell interactions in healthy and diseased
        tissues. Nat Commun 14, 7739 (2023).
    """
    # Either plotting overall interactions, or just for a particular LR #
    int_df, title = _get_lr_data(adata, sample, lr_type)

    int_df = int_df.transpose()

    flux = int_df.values
    total_ints = flux.sum(axis=1) + flux.sum(axis=0) - flux.diagonal()
    keep = np.where(total_ints > min_ints)[0]
    # Limit of 10 for good display #
    if len(keep) > n_top_ccis:
        keep = np.argsort(-total_ints)[0:n_top_ccis]
    # Filter any with all zeros after filtering #
    all_zero = np.array(
        [np.all(np.logical_and(flux[i, keep] == 0, flux[keep, i] == 0)) for i in keep]
    )
    keep = keep[all_zero == False]
    if len(keep) == 0:  # If we don't keep anything, warn the user
        return

    flux = flux[:, keep]
    flux = flux[keep, :].astype(float)

    # Add pseudocount to row/column which has all zeros for the incoming
    # so can make the connection between the two
    for i in range(flux.shape[0]):
        if np.all(flux[i, :] == 0):
            flux[i, flux[:, i] > 0] += sys.float_info.min
        elif np.all(flux[:, i] == 0):
            flux[flux[i, :] > 0, i] += sys.float_info.min

    cell_names = int_df.index.values.astype(str)[keep]
    nodes = cell_names

    # Retrieving colors of cell types #
    cluster = np.unique(int_df.index)
    colors = _get_palette(cluster, palette=cmap)
    colors = list(colors.values())

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    nodePos = chordDiagram(flux, ax, lim=1.25, colors=colors)
    ax.axis("off")
    prop = dict(fontsize=label_size, ha="center", va="center")
    for i in range(len(cell_names)):
        x, y = nodePos[i][0:2]
        rotation = nodePos[i][2]
        # Prevent text going upside down at certain rotations
        if (90 > rotation > 18 and label_rotation != 0) or (120 > rotation > 90):
            label_rotation_ = -label_rotation
        else:
            label_rotation_ = label_rotation
        ax.text(
            x, y, nodes[i], rotation=nodePos[i][2] + label_rotation_, **prop
        )  # size=10,
    fig.suptitle(title, fontsize=20)

    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()
    else:
        return ax


def show_ccc_dotplot(
        adata: AnnData,
        sample: Union[str, list] = None,
        lr_type: Literal['contact', 'secretory'] = 'contact',
        affinity_cutoff: float = 0.05,
        strength_cutoff: float = 2.0,
        lrs_name: Optional[list] = None,
        cts_name: Optional[list] = None,
        n_top_lrs: Optional[int] = 10,
        n_top_cts: Optional[int] = 15,
        figsize: Tuple[Any, Any] = (7, 5),
        cmap: Union[Colormap, str] = "Spectral_r",
        strength_min: Optional[float] = None,
        strength_max: Optional[float] = None,
        size: int = 10,
        sig_color: str = '#000000',
        ax: Axes = None,
        show: bool = False,
        save: Optional[str] = None,
        **kwargs
) -> Optional[Axes]:
    """
    Dotplot for cell communication analysis are presented, either specifying the cell type pairs and ligand-receptor
    pairs to be presented or showing the most significant cell type pairs and ligand-receptor pairs after ranking by
    significance level.

    Parameters
    ----------
    adata : anndata.Anndata
        An AnnData object containing spatial omics data and spatial information.
    sample : Union[str, list]
        Name of sample.
    lr_type : Literal['contact', 'secretory']
        Types of ligand-receptor pairs.
    affinity_cutoff : float
        The threshold at which affinity was significant, and affinity below this threshold were considered significant.
    strength_cutoff : float
        The threshold at which strength was significant, and strength above this threshold were considered significant.
    lrs_name : Optional[list]
        Ligand-receptor pairs that need to be show.
        e.g. ['Endothelial cell,Macrophage', 'Macrophage,Endothelial cell']
    cts_name : Optional[list]
        cell type pairs that need to be show.
        e.g. ['FGF2:FGFR1', 'WNT3A:FZD7&LRP5']
    n_top_lrs : int
        Maximum number of ligand-receptor pairs presented, ordered according to the number of cell type pairs for which
        a ligand-receptor pair was significant.
    n_top_cts : int
        The maximum number of cell type pairs shown, ordered according to the number of significant ligand-receptor
        pairs for a cell type pair.
    figsize : Tuple[Any, Any]
        The size of format figure.(width, height)
    cmap : Union[Colormap, str]
        Color map of dotplot.
    strength_min, strength_max : Optional[float]
        The extreme value of the strength that is displayed, the value that is not in the range will be displayed in the
         same color as the corresponding extreme value.
    size : int
        Size of the points in figure.
    sig_color : str
        Points that were considered significant were highlighted by the color.
    ax : matplotlib.figure.Axes
        A matplotlib.axes object.
    show : str
        Show this plot.
    save : Optional[str]
        The path where the image is stored.
    kwargs: Any, optional
        Other params of ax.scatter()

    Returns
    -------
        If show==False, return Axes
    """
    strength, affinity = \
        _get_lr_matrix(adata=adata,
                       affinity_cutoff=affinity_cutoff,
                       strength_cutoff=strength_cutoff,
                       lrs_name=lrs_name,
                       cts_name=cts_name,
                       n_top_lrs=n_top_lrs,
                       n_top_cts=n_top_cts,
                       sample_id=sample,
                       lr_type=lr_type
                       )

    # heatmap
    ax = _dotplot(
        scores=strength,
        pvalue=affinity,
        s_cutoff=strength_cutoff,
        p_cutoff=affinity_cutoff,
        vmin=strength_min,
        vmax=strength_max,
        ax=ax,
        cmap=cmap,
        sig_color=sig_color,
        figsize=figsize,
        s=size,
        **kwargs
    )
    ax.set_title(f'{lr_type} lr pairs')
    ax.set_ylabel('LR-pair')
    ax.set_xlabel('cell type')

    legend_elements = [
        plt.scatter([], [], marker='o', color='silver', edgecolor='#FFFFFF', s=size, label='> 0.5'),
        plt.scatter([], [], marker='o', color='silver', edgecolor='#FFFFFF', s=size * 2, label='> 0.1'),
        plt.scatter([], [], marker='o', color='silver', edgecolor='#FFFFFF', s=size * 3, label='> 0.05'),
        plt.scatter([], [], marker='o', color='silver', edgecolor='#FFFFFF', s=size * 4, label='< 0.05')
        ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.2, 0.0), title='P value')

    plt.tight_layout()

    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()
    else:
        return ax


def show_ccc_embedding(
        adata: anndata.AnnData,
        ligand: Union[str, list],
        receptor: Union[str, list],
        ligand_clusters: Union[list, str, int],
        receptor_clusters: Union[list, str, int],
        cluster_key: str = 'cell_type',
        ligand_region: Tuple[Any, Any] = None,
        receptor_region: Tuple[Any, Any] = None,
        ligand_color: Union[Colormap, str] = None,
        receptor_color: Union[Colormap, str] = None,
        row_region: Tuple[Any, Any] = None,
        col_region: Tuple[Any, Any] = None,
        agg_method: Literal['mean', 'min', 'gmean'] = 'gmean',
        title: Optional[str] = None,
        figsize: Tuple[Any, Any] = None,
        dpi: int = 100,
        ax: Optional[Axes] = None,
        size: Optional[float] = None,
        obsm_spatial_label: str = 'spatial',
        show: bool = True,
        save: Optional[str] = None,
        cbar_kwargs: Optional[dict] = {},
        **kwargs
) -> Optional[Axes]:
    """
    To map the distribution of ligands and receptors expression of different cell type in a situ map of tissue.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    figsize : Tuple[float, float], optional
        (Width, height) of the figure.
    dpi : float, optional
        The resolution of the figure.
    title : str, optional
        The title of shown figure.
    ax : Axes
        A matplotlib axes object.
    show : bool
        Show the plot, do not return axis.
    save : Union[str, PathLike], optional
        The path where the image is stored.
    ligand : Union[str, list]
        Ligand name or a list containing ligand subunits
    receptor : Union[str, list]
        Receptor name or a list containing receptor subunits
    ligand_clusters : Union[list, str, int]
        The cell type clusters of the ligand shown, cannot be duplicated with the clusters of the receptor,
        otherwise overlap would occur.
    receptor_clusters : Union[list, str, int]
        The cell type clusters of the receptor shown, cannot be duplicated with the clusters of the ligand,
        otherwise overlap would occur.
    cluster_key : str
        The label of cluster in adata.obs.
    ligand_region : Tuple[Any, Any]
        The range of values of the ligand. (ligand_min, ligand_max)
    receptor_region : Tuple[Any, Any]
        The range of values of the receptor. (receptor_min, receptor_max)
    ligand_color : Union[Colormap, str]
        The color of the ligand.
    receptor_color : Union[Colormap, str]
        The color of the receptor.
    row_region : Tuple[Any, Any]
        The row coordinate threshold of the region in the situ map is displayed, and the expression of the whole
        situ map is displayed by default. (row_low, row_high)
    col_region : Tuple[Any, Any]
        The row coordinate threshold of the region in the situ map is displayed, and the expression of the whole
        situ map is displayed by default. (row_low, row_high)
    agg_method : str
        Integrated subunit to calculate ligand or receptor.
        'min': ligand or receptor = min(sub1, sub2, sub3, ...)
        'mean': ligand or receptor = mean(sub1, sub2, sub3, ...)
        'gmean': geometrical mean, ligand or receptor = np.gmean(sub1, sub2, sub3, ...)
    size : Optional[float]
        Size of the points in figure.
    obsm_spatial_label : str
        The key of spatial coordinates in adata.obsm.
    cbar_kwargs : Optional[dict]
        Other parameters in plt.colorbar().
    kwargs : Any, optional
        Other params of ax.scatter()

    Returns
    -------
        If show==False, return Axes
    """

    def subunit_merging(exp, agg_method):
        if agg_method == 'min':
            update_ = np.min(exp, axis=1)
        elif agg_method == 'mean':
            update_ = np.mean(exp, axis=1)
        elif agg_method == 'gmean':
            update_ = gmean(exp, axis=1)
        else:

            ValueError("{} is not a valid agg_method".format(agg_method))

        return update_

    def in_region(data, up, down, method):
        up_index = np.where(data > up, True, False)
        down_index = np.where(data < down, True, False)
        if method == 'drop':
            index_all = up_index | down_index
            data = np.array([-1 if index_all[i] else x for i, x in enumerate(data)])
            return data
        else:
            data[up_index] = up
            data[down_index] = down
            return data

    adata_ligand = copy.deepcopy(adata)
    adata_receptor = copy.deepcopy(adata)
    clusters = adata.obs[cluster_key].tolist()

    if not isinstance(ligand_clusters, list):
        ligand_clusters = [ligand_clusters]

    if not isinstance(receptor_clusters, list):
        receptor_clusters = [receptor_clusters]

    if np.any(np.isin(ligand_clusters, receptor_clusters)):
        logging.error(
            'The ligand clusters is duplicated with the receptor clusters. Duplicate categories\' clusters will be overwritten.'
            'Please check it.'
        )

    index_ligand = np.isin(clusters, ligand_clusters)
    adata_ligand = adata_ligand[index_ligand, :]
    index_receptor = np.isin(clusters, receptor_clusters)
    adata_receptor = adata_receptor[index_receptor, :]

    ligand_x, ligand_y = adata_ligand.obsm[obsm_spatial_label].T
    receptor_x, receptor_y = adata_receptor.obsm[obsm_spatial_label].T
    if isinstance(ligand, list):
        ligand_exp = adata_ligand[:, ligand].X.toarray()
        ligand_exp = subunit_merging(ligand_exp, agg_method)
    else:
        ligand_exp = adata_ligand[:, ligand].X.toarray().T[0]
    if isinstance(receptor, list):
        receptor_exp = adata_receptor[:, receptor].X.toarray()
        receptor_exp = subunit_merging(receptor_exp, agg_method)
    else:
        receptor_exp = adata_receptor[:, receptor].X.toarray().T[0]

    if col_region is not None:
        ligand_x = in_region(ligand_x, up=col_region[1], down=col_region[0], method='drop')
        receptor_x = in_region(receptor_x, up=col_region[1], down=col_region[0], method='drop')
    if row_region is not None:
        ligand_y = in_region(ligand_y, up=row_region[1], down=row_region[0], method='drop')
        receptor_y = in_region(receptor_y, up=row_region[1], down=row_region[0], method='drop')

    if ligand_region is not None:
        ligand_exp = in_region(ligand_exp, up=ligand_region[1], down=ligand_region[0], method='no drop')
    if receptor_region is not None:
        receptor_exp = in_region(receptor_exp, up=receptor_region[1], down=receptor_region[0], method='no drop')

    index_ = []
    for i in range(len(ligand_x)):
        if np.isin(-1, [ligand_x[i], ligand_y[i]]):
            index_.append(False)
        else:
            index_.append(True)

    ligand_x = ligand_x[index_]
    ligand_y = ligand_y[index_]
    ligand_exp = ligand_exp[index_]

    index_ = []
    for i in range(len(receptor_x)):
        if np.isin(-1, [receptor_x[i], receptor_y[i]]):
            index_.append(False)
        else:
            index_.append(True)

    receptor_x = receptor_x[index_]
    receptor_y = receptor_y[index_]
    receptor_exp = receptor_exp[index_]

    if size is None:
        size = 120000 / adata.shape[0]

    if ax is None:
        if figsize is None:
            figsize = (10, 8)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    if ligand_color is None:
        # ligand_color = ListedColormap(sns.color_palette("YlOrBr", as_cmap=True))
        ligand_color = sns.color_palette("YlOrBr", as_cmap=True)
    if receptor_color is None:
        # receptor_color = ListedColormap(sns.color_palette("Blues", as_cmap=True))
        from matplotlib.colors import LinearSegmentedColormap
        receptor_color = sns.color_palette("Blues", as_cmap=True)
        receptor_color = LinearSegmentedColormap.from_list(
            "custom_blues", receptor_color(np.linspace(0.3, 1.0, 256))
        )

    ax_0 = ax.scatter(
        ligand_x,
        ligand_y,
        s=size,
        c=ligand_exp,
        cmap=ligand_color,
        **kwargs
    )

    ax_1 = ax.scatter(
        receptor_x,
        receptor_y,
        s=size,
        c=receptor_exp,
        cmap=receptor_color,
        **kwargs
    )

    ax.set_xlabel(obsm_spatial_label+'1')
    ax.set_ylabel(obsm_spatial_label+'2')

    ax.set_yticks([])
    ax.set_xticks([])

    cbar1 = plt.colorbar(ax_0, ax=ax, orientation='vertical', pad=0.02, **cbar_kwargs)
    cbar2 = plt.colorbar(ax_1, ax=ax, orientation='vertical', pad=0.02, **cbar_kwargs)

    cbar1.ax.set_title('ligand', fontsize=12)
    cbar2.ax.set_title('receptor', fontsize=12)

    cbar1.ax.set_position([0.8, 0.55, 0.06, 0.35])  # 上方的颜色棒
    cbar2.ax.set_position([0.8, 0.1, 0.06, 0.35])  # 下方的颜色棒

    ax.autoscale_view()

    if title is not None:
        title = f'{ligand_clusters}:{ligand},\n {receptor_clusters}:{receptor}'

    fig.suptitle(title, fontsize=20)

    if save is not None:
        plt.savefig(save)

    if show:
        plt.show()
    else:
        return ax