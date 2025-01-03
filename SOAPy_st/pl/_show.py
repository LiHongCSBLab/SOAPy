import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from os import PathLike
from types import MappingProxyType
from typing import Optional, Union, Any, Tuple, Sequence, Mapping
import copy
from ._color import _get_palette
from ..utils import _scale, _get_info_from_sample
from matplotlib.axes import Axes

__all__ = [
    'show_moran_scatterplot',
    'show_network',
    'show_voronoi'
]


def show_moran_scatterplot(
        adata: anndata.AnnData,
        palette: Union[Sequence[str], ListedColormap] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[float] = None,
        title: Optional[str] = None,
        ax: Optional[Axes] = None,
        show: bool = True,
        save: Union[str, PathLike] = None,
        **kwargs
) -> Optional[Axes]:
    """
    Show a Moran scatter plot. Highlight the hotspots of the domain_from_local_moran.

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
    palette : Union[Sequence[str], Cycler], optional
        (hotspot_color, other_color).
        Displays the Moran index color for hot and non-hot areas, inputing a list of two RGB colors.
    **kwargs
        Other params of ax.scatter()

    Returns
    -------
        If show==False, return Axes.
    """
    if palette is None:
        palette = ['#dd7e6b', '#bababa']
    if isinstance(palette, ListedColormap):
        palette = palette.colors[0:2]

    if ax is None:
        if figsize is None:
            figsize = (7, 7)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Attribute')
    ax.set_ylabel('Spatial Lag')
    ax.set_title('Moran Scatterplot')

    local_moran = _get_info_from_sample(adata=adata, sample_id=None, key='local_moran')
    local_moran_df = local_moran['LocalMoran']

    for index, spot in local_moran_df.iterrows():
        x = spot['Attribute']
        y = spot['Spatial Lag']
        label = spot['Label']
        if label == 'Hotspot':
            c = palette[0]
        else:
            c = palette[1]
        ax.scatter(x, y,
                   label=label,
                   c=c,
                   **kwargs
                   )

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    attribute_filter = local_moran['attribute_filter']
    spatial_lag_filter = local_moran['spatial_lag_filter']

    ax.vlines(attribute_filter, min(local_moran_df['Spatial Lag']), max(local_moran_df['Spatial Lag']), linestyle='--')
    ax.hlines(spatial_lag_filter, min(local_moran_df['Attribute']), max(local_moran_df['Attribute']), linestyle='--')

    if title is None:
        title = 'Moran Scatterplot'
    plt.title(title, fontsize=20)

    if save:
        plt.savefig(save)

    if show:
        plt.show()
    else:
        return ax


def show_voronoi(
        adata: anndata.AnnData,
        cluster_key: str,
        sample_key: Optional[str] = 'sample',
        sample_id: Union[int, str] = None,
        palette: Union[Sequence[str], ListedColormap, dict] = None,
        title: Optional[str] = None,
        spatial_in_obsm: str = 'spatial',
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[float] = None,
        ax: Optional[Axes] = None,
        show: bool = True,
        save: Union[str, PathLike] = None,
        legend_kwargs: dict = {},
        **kwargs: Any
) -> Optional[Axes]:
    """
    Draw a Voronoi plot of the cell distribution.

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
    sample_key : str, optional
        Batch's key in adata.obs.
    sample_id : Union[int, str], optional
        The sample number which to be shown.
    cluster_key : str
        Keys for annotations in adata.obs.
    palette : Union[Sequence[str], Cycler, dict]
        Colors to use for plotting annotation groups.
    spatial_in_obsm : str
        Keyword of coordinate information in obsm.
    legend_kwargs : dict, optional
        Other params of plt.legend().
    kwargs : Any, optional
        Other params of scipy.spatial.voronoi_plot_2d().

    Returns
    -------
        If show==False, return Axes.
    """
    import matplotlib.pyplot as plt
    from scipy.spatial import Voronoi, voronoi_plot_2d
    from collections import OrderedDict

    if sample_id is not None:
        adata = adata[adata.obs[sample_key] == sample_id, :].copy()
    else:
        adata = copy.deepcopy(adata)

    if title is None:
        title = str(cluster_key)

    points = adata.obsm[spatial_in_obsm]
    cluster_list = adata.obs[cluster_key]

    cluster_unique = cluster_list.unique()

    if not isinstance(palette, dict):
        palette = _get_palette(list(cluster_unique), sort_order=True, palette=palette)
    else:
        palette = palette

    points_color = [palette[index] for index in cluster_list]

    vor = Voronoi(points)

    if figsize is None:
        row = np.max(points[:, 1])
        col = np.max(points[:, 0])
        figsize = (8, 8 * row / col)

    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0.1, 0.2, 0.6, 0.6])

    voronoi_plot_2d(vor,
                    show_vertices=False,
                    point_size=0,
                    ax=ax,
                    **kwargs)

    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=points_color[r], label=cluster_list[r])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    new_label = {}
    for key in sorted(by_label):
        new_label[key] = by_label[key]

    if 'bbox_to_anchor' not in legend_kwargs.keys():
        legend_kwargs['bbox_to_anchor'] = (1.02, 0.8)
    if 'loc' not in legend_kwargs.keys():
        legend_kwargs['loc'] = 'upper left'
    if 'title' not in legend_kwargs.keys():
        legend_kwargs['title'] = cluster_key

    ax.legend(new_label.values(), new_label.keys(), **legend_kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(title, fontsize=20)
    plt.tight_layout()

    if save:
        plt.savefig(save)

    if show:
        plt.show()
    else:
        return ax


def show_network(
        adata: anndata.AnnData,
        sample_key: Optional[str] = 'sample',
        sample_id: Union[int, str] = None,
        cluster_key: str = 'clusters',
        edge_color: str = 'b',
        palette: Union[Sequence[str], ListedColormap, dict] = None,
        title: Optional[str] = 'Spatial network',
        scale: Union[str, float] = 'hires',
        spatial_in_obsm: str = 'spatial',
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 100,
        ax: Optional[Axes] = None,
        spot_size: float = 5,
        show: bool = True,
        save: Union[str, PathLike] = None,
        legend_kwargs: dict = {},
        **kwargs: Any
) -> Optional[Axes]:
    """
    Show the spatial network.

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
    sample_key : str, optional
        Batch's key in adata.obs.
    sample_id : Union[int, str], optional
        The sample number which to be shown.
    cluster_key : str
        Keys for annotations in adata.obs. It can not be None.
    edge_color : str
        The color of edges in the network.
    palette : Union[Sequence[str], Cycler, dict], optional
        Colors to use for plotting annotation groups.
    scale : Union[str, float]
        The scaling factor for distance scaling. If it's Visium data it can also be HE image labels (hires or lower).
        Most of the time you don't need to change this.
    spatial_in_obsm : str
        Keyword of coordinate information in obsm.
    spot_size : float
        The size of the spot
    legend_kwargs : dict, optional
        Other params of plt.legend().
    kwargs : Any, optional
        Other params of scipy.spatial.voronoi_plot_2d().

    Returns
    -------
        If show==False, return Axes.
    """
    edges = _get_info_from_sample(adata=adata, sample_id=sample_id, key='edges')
    if sample_id is not None:
        bdata = adata[adata.obs[sample_key] == sample_id, :].copy()
    else:
        bdata = copy.deepcopy(adata)
    obs = bdata.obs
    if type(scale) != float:
        scale = _scale(bdata, None, scale)

    df_pixel = bdata.obsm[spatial_in_obsm]
    df_pixel = pd.DataFrame(df_pixel, index=bdata.obs_names) * scale
    obs['x'] = df_pixel[0].tolist()
    obs['y'] = df_pixel[1].tolist()

    if figsize is None:
        row = max(obs['y'])
        col = max(obs['x'])
        figsize = (8, 8 * row / col)

    if ax is None:
        fig = plt.figure(figsize=figsize,
                         dpi=dpi,
                         )
        ax = fig.add_axes([0.1, 0.2, 0.6, 0.6])
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    for index, row in edges.iterrows():
        index_i = obs.iloc[int(getattr(row, 'point_1')), :]
        index_j = obs.iloc[int(getattr(row, 'point_2')), :]
        ax.plot([index_i['x'], index_j['x']],
                [index_i['y'], index_j['y']],
                c=edge_color,
                zorder=2,
                **kwargs
                )

    cluster_unique = obs[cluster_key].unique()
    cluster_unique = sorted(cluster_unique)

    if not isinstance(palette, dict):
        palette = _get_palette(list(cluster_unique), sort_order=True, palette=palette)
    else:
        palette = palette

    for index, clu in enumerate(cluster_unique):
        sub_obs = obs[obs[cluster_key] == clu]
        ax.scatter(sub_obs['x'],
                   sub_obs['y'],
                   label=clu,
                   c=palette[clu],
                   s=spot_size,
                   zorder=3)

    if 'bbox_to_anchor' not in legend_kwargs.keys():
        legend_kwargs['bbox_to_anchor'] = (1.02, 0.8)
    if 'loc' not in legend_kwargs.keys():
        legend_kwargs['loc'] = 'upper left'
    if 'title' not in legend_kwargs.keys():
        legend_kwargs['title'] = cluster_key

    ax.legend(**legend_kwargs)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label=title, fontsize=20)
    plt.tight_layout()

    if save:
        plt.savefig(save)

    if show:
        plt.show()
    else:
        return ax