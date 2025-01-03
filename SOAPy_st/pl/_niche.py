import copy
import anndata
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from os import PathLike
from types import MappingProxyType
from typing import Optional, Union, Any, Literal, Tuple, Mapping, Sequence
from ._heatmap import _heatmap_with_dendrogram_and_bar
from ..utils import _get_info_from_sample, _scale
from ._color import _get_palette

__all__ = [
    'show_celltype_sample_heatmap',
    'show_niche_environment',
    'show_celltype_niche_heatmap',
    'show_niche_sample_heatmap'
]


def show_celltype_sample_heatmap(
        adata: anndata.AnnData,
        norm_axis: Literal['celltype', 'sample', None] = None,
        norm_method: Literal['normalization', 'z_score', 'proportion'] = 'normalization',
        celltype_bar: bool = True,
        sample_bar: bool = True,
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 100,
        cmap: LinearSegmentedColormap = None,
        show: bool = True,
        save: Union[str, PathLike] = None,
        celltype_bar_kwargs: dict = {},
        sample_bar_kwargs: dict = {},
        **kwargs
) -> Optional[Tuple[Figure, dict]]:
    """
    Heatmap show the cell composition of each sample. Bar plot can be added to show the overall cell types distribution
    and the total number of cells in each sample, respectively.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    norm_axis : Literal['celltype', 'sample']
        The direction of standardization.
        'celltype': Standardization was performed for each cell type.
        'sample': Standardization was performed for each sample.
        None: No standardization.
    celltype_bar
        Whether to show a bar plot of the cell number of each cell type.
    sample_bar
        Whether to show a bar plot of the cell number of each sample.
    celltype_bar_kwargs : Any
        Other parameters in sns.countplot() of cell type's bar plot.
    sample_bar_kwargs : Any
        Other parameters in sns.countplot() of sample's bar plot.
    norm_method : Literal['normalization', 'z_score', 'proportion']
        Methods of standardization
        'normalization': x^ = (x - min(x)) / (max(x) - min(x))
        'z_score': x^ = (x - mean(x)) / std(x)
        'proportion': x^ = x / sum(x)
    figsize : Tuple[float, float], optional
        (Width, height) of the figure.
    dpi : float, optional
        The resolution of the figure.
    title : str, optional
        The title of shown figure.
    show : bool
        Show the plot, do not return axis.
    save : Union[str, PathLike], optional
        The path where the image is stored.
    cmap
        Color map to use for continous variables.
    kwargs : Any
        Other params of sns.heatmap()

    Returns
    -------
        If show==False, return Tuple[Figure, list[Axes]]

    """
    if celltype_bar_kwargs is None:
        celltype_bar_kwargs = {}
    if sample_bar_kwargs is None:
        sample_bar_kwargs = {}

    if title is None:
        title = 'Celltype-Sample'

    norm_axis = 0 if norm_axis == 'celltype' else 1

    data = _get_info_from_sample(adata=adata, sample_id=None, key='niche')
    sample = data['sample'].tolist()
    celltype = data['celltype'].tolist()
    sample_uni = np.unique(sample)
    celltype_uni = np.unique(celltype)

    dict_clu = {clu: i for i, clu in enumerate(celltype_uni)}
    dict_sam = {clu: i for i, clu in enumerate(sample_uni)}
    mat_sample_cluster = np.zeros((len(sample_uni), len(celltype_uni)))

    for index in range(len(sample)):
        mat_sample_cluster[dict_sam[sample[index]], dict_clu[celltype[index]]] += 1

    fig, axes = _heatmap_with_dendrogram_and_bar(
        mat_sample_cluster,
        data_count=data,
        x_label='celltype',
        y_label='sample',
        x_map=dict_clu,
        y_map=dict_sam,
        x_dendrogram=False,
        norm_axis=norm_axis,
        method=norm_method,
        x_bar=celltype_bar,
        y_bar=sample_bar,
        cmap=cmap,
        title=title,
        figsize=figsize,
        dpi=dpi,
        xbar_kwags=celltype_bar_kwargs,
        ybar_kwags=sample_bar_kwargs,
        **kwargs
)

    if save:
        fig.savefig(save)
    if show:
        fig.show()
    return fig, axes


def show_celltype_niche_heatmap(
        adata: anndata.AnnData,
        norm_method: Literal['normalization', 'z_score', 'proportion'] = 'normalization',
        niche_bar: bool =True,
        niche_bar_kwargs: Mapping[str, Any] = MappingProxyType({}),
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 100,
        cmap: LinearSegmentedColormap = None,
        show: bool = True,
        save: Union[str, PathLike, None] = None,
        **kwargs,
) -> Optional[Tuple[Figure, dict]]:
    """
    The heatmap shows the cell composition of each niche,
    and the bar plot graph shows the number of each niche in all samples.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    niche_bar : bool
        Whether to show a bar plot of the cell number of each niche.
    niche_bar_kwargs : Any
        Other parameters in sns.countplot() of cell type's bar plot.
    norm_method : Literal['normalization', 'z_score', 'proportion']
        Methods of standardization
        'normalization': x^ = (x - min(x)) / (max(x) - min(x))
        'z_score': x^ = (x - mean(x)) / std(x)
        'proportion': x^ = x / sum(x)
    figsize : Tuple[float, float], optional
        (Width, height) of the figure.
    dpi : float, optional
        The resolution of the figure.
    title : str, optional
        The title of shown figure.
    show : bool
        Show the plot, do not return axis.
    save : Union[str, PathLike], optional
        The path where the image is stored.
    cmap
        Color map to use for continous variables.
    kwargs : Any
        Other params of sns.heatmap()

    Returns
    -------
        If show==False, return Tuple[Figure, list[Axes]]

    """
    if niche_bar_kwargs is None:
        niche_bar_kwargs = {}

    if title is None:
        title = 'Celltype-C_niche'

    data = _get_info_from_sample(adata=adata, sample_id=None, key='niche')
    data = data.fillna(0)
    niche = data.columns[:-4]
    drop_col = data.columns[-4:]
    niche_num = len(np.unique(data['C_niche'].tolist()))

    mat_niche = []
    for i in range(niche_num):
        data_use = data[data['C_niche'] == i]
        for j in drop_col:
            data_use = data_use.drop(j, axis=1)
        mat_use = data_use.values
        mat_niche.append(np.array(np.sum(mat_use, axis=0))/mat_use.shape[0])
    mat_niche = np.array(mat_niche)
    pd_data = pd.DataFrame(mat_niche, index=range(niche_num), columns=niche)
    dict_clu = {clu: i for i, clu in enumerate(pd_data.columns)}
    dict_niche = {clu: i for i, clu in enumerate(pd_data.index)}
    fig, axes = _heatmap_with_dendrogram_and_bar(
        pd_data.values,
        data_count=data,
        x_label='celltype',
        y_label='C_niche',
        x_map=dict_clu,
        y_map=dict_niche,
        x_bar=False,
        y_bar=niche_bar,
        x_dendrogram=False,
        method=norm_method,
        norm_axis=1,
        cmap=cmap,
        title=title,
        figsize=figsize,
        dpi=dpi,
        ybar_kwags=niche_bar_kwargs,
        **kwargs
)

    if save:
        fig.savefig(save)
    if show:
        fig.show()
    return fig, axes


def show_niche_sample_heatmap(
        adata: anndata.AnnData,
        norm_axis: Literal['niche', 'sample', None] = None,
        norm_method: Literal['normalization', 'z_score', 'proportion'] = 'normalization',
        niche_bar: bool = True,
        sample_bar: bool = True,
        niche_bar_kwargs: Any = None,
        sample_bar_kwargs: Any = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 100,
        cmap: LinearSegmentedColormap = None,
        show: bool = True,
        save: Union[str, PathLike, None] = None,
        **kwargs
) -> Optional[Tuple[Figure, dict]]:
    """
    The heatmap shows the niche composition in each sample, and the bar plot shows the total number of each niche and
    the total number of cells in each sample, respectively.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    norm_axis : Literal['celltype', 'sample']
        The direction of standardization.
        'celltype': Standardization was performed for each cell type.
        'sample': Standardization was performed for each sample.
        None: No standardization.
    niche_bar : bool
        Whether to show a bar plot of the cell number of each niche.
    sample_bar : bool
        Whether to show a bar plot of the cell number of each sample.
    niche_bar_kwargs
        Other parameters in sns.countplot() of cell niche's bar plot.
    sample_bar_kwargs
        Other parameters in sns.countplot() of sample's bar plot.
    norm_method : Literal['normalization', 'z_score', 'proportion']
        Methods of standardization
        'normalization': x^ = (x - min(x)) / (max(x) - min(x))
        'z_score': x^ = (x - mean(x)) / std(x)
        'proportion': x^ = x / sum(x)
    figsize : Tuple[float, float], optional
        (Width, height) of the figure.
    dpi : float, optional
        The resolution of the figure.
    title : str, optional
        The title of shown figure.
    show : bool
        Show the plot, do not return axis.
    save : Union[str, PathLike], optional
        The path where the image is stored.
    cmap
        Color map to use for continous variables.
    kwargs : Any
        Other params of sns.heatmap()

    Returns
    -------
        If show==False, return Tuple[Figure, list[Axes]]

    """
    if niche_bar_kwargs is None:
        niche_bar_kwargs = {}
    if sample_bar_kwargs is None:
        sample_bar_kwargs = {}

    if title is None:
        title = 'C_niche-Sample'

    data = _get_info_from_sample(adata=adata, sample_id=None, key='niche')
    norm_axis = 0 if norm_axis == 'niche' else 1

    sample = data['sample'].tolist()
    niche = data['C_niche'].tolist()
    sample_uni = np.unique(sample)
    niche_uni = np.unique(niche)

    dict_niche = {clu: i for i, clu in enumerate(np.unique(niche))}
    dict_sam = {clu: i for i, clu in enumerate(np.unique(sample))}

    mat_sample_niche = np.zeros((len(sample_uni), len(niche_uni)))

    for index in range(len(sample)):
        mat_sample_niche[dict_sam[sample[index]], dict_niche[niche[index]]] += 1

    fig, axes = _heatmap_with_dendrogram_and_bar(
        mat_sample_niche,
        data_count=data,
        x_label='C_niche',
        y_label='sample',
        x_map=dict_niche,
        y_map=dict_sam,
        norm_axis=norm_axis,
        method=norm_method,
        x_bar=niche_bar,
        y_bar=sample_bar,
        cmap=cmap,
        title=title,
        figsize=figsize,
        dpi=dpi,
        xbar_kwags=niche_bar_kwargs,
        ybar_kwags=sample_bar_kwargs,
        **kwargs
)

    if save:
        fig.savefig(save)
    if show:
        fig.show()
    return fig, axes


def show_niche_environment(
        adata: anndata.AnnData,
        niche: Union[int, str],
        celltype_key: str = 'clusters',
        niche_key: str = 'C_niche',
        sample_key: str = 'sample',
        sample_id: Union[str, int] = None,
        scale: Union[str, float] = 'hires',
        palette: Union[Sequence[str], ListedColormap] = None,
        title: Optional[str] = None,
        ax: Optional[Axes] = None,
        sig_niche_line_color: str = '#000000',
        other_niche_line_color: str = '#999999',
        spatial_in_obsm: str = 'spatial',
        show: bool = True,
        save: Union[str, PathLike, None] = None,
        **kwargs
) -> Optional[Axes]:
    """
    Show the environment composition of a niche.
    The cell boundaries belonging to this niche will be highlighted (the color is regulated by the sig_niche_line_color
    parameter), and the cell boundaries' color of other cells are regulated by the other_niche_line_color parameter.
    Cells labeled with cell type color indicate participation in this niche calculation.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    niche

    celltype_key : str
        The label of cell type in adata.obs.
    niche_key : str
        The label of niche in adata.obs.
    sample_key : str
        The label of sample in adata.obs.
    sample : Union[str, int]
        Sample number to be shown.
    scale : Union[str, float]
        Scale used in subsequent analyses. If it's Visium data it can also be HE image labels (hires or lower).
        Most of the time you don't need to change this.
    color

    title : str
        The title of this figure.
    sig_niche_line_color : str
        The color of the outline of cells in selected niche.
    other_niche_line_color : str
        The color of the outline of cells in other niches.
    spatial_in_obsm : str
        Keyword of coordinate information in obsm.
    show : bool
        Show the plot, do not return axis.
    save : Union[str, PathLike]
        The path where the image is stored.
    kwargs : Any
        Other params of voronoi_plot_2d()

    Returns
    -------
        If show==False, return Axes.

    """
    from scipy.spatial import Voronoi, voronoi_plot_2d
    from collections import OrderedDict

    adata = adata.copy()
    if sample_id is not None:
        try:
            adata = adata[adata.obs[sample_key] == sample_id, :].copy()
        except ValueError:
            raise ValueError(f'The {sample_key} was not found in adata.obs.columns')

    indices = _get_info_from_sample(adata, sample_id=sample_id, key='indices')
    adata.obs['attention_label'] = None

    for index in range(adata.obs.shape[0]):
        edges = indices[index]
        index_of_niche = adata.obs.index[index]
        if adata.obs.loc[index_of_niche, niche_key] == niche:
            adata.obs.loc[index_of_niche, 'attention_label'] = 'center'
            for nei in edges:
                index_of_neigh = adata.obs.index[nei]
                if index_of_neigh != index_of_niche and adata.obs.loc[index_of_neigh, 'attention_label'] != 'center':
                    adata.obs.loc[index_of_neigh, 'attention_label'] = 'neighbor'

    if type(scale) != float:
        scale = _scale(adata, None, scale)

    points = adata.obsm[spatial_in_obsm]*scale
    cluster_list = adata.obs[celltype_key]

    cluster_unique = cluster_list.unique()
    cluster_unique = sorted(cluster_unique)
    attention = adata.obs['attention_label'].tolist()

    if not isinstance(palette, dict):
        palette = _get_palette(list(cluster_unique), sort_order=True, palette=palette)
    else:
        palette = palette

    points_color = [palette[index] for index in cluster_list]

    vor = Voronoi(points)

    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0.1, 0.1, 0.5, 0.75])

    voronoi_plot_2d(vor,
                    show_vertices=False,
                    point_size=0,
                    line_colors=other_niche_line_color,
                    ax=ax,
                    **kwargs
                    )

    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            if attention[r] == 'center':
                x = [p[0] for p in polygon]
                y = [p[1] for p in polygon]
                x.append(polygon[0][0])
                y.append(polygon[0][1])
                plt.plot(x, y, color=sig_niche_line_color, linewidth=1.0)
                plt.fill(*zip(*polygon), color=points_color[r], label=cluster_list[r])
            elif attention[r] == 'neighbor':
                plt.fill(*zip(*polygon), color=points_color[r], label=cluster_list[r])
            else:
                pass
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    new_label = {}
    for key in sorted(by_label):
        new_label[key] = by_label[key]
    plt.legend(new_label.values(), new_label.keys(), bbox_to_anchor=(1.02, 0.8), loc='upper left')
    plt.title(title)

    if save:
        plt.savefig(save)
    if show:
        plt.show()
    else:
        return ax