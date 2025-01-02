import copy
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from typing import Any, Tuple, Optional, Union, Sequence, Literal
from matplotlib.colors import ListedColormap
from os import PathLike
from matplotlib.axes import Axes
from ..utils import _get_info_from_sample
from ._color import _get_palette, color_list_50

__all__ = [
    'show_tendency',
    'show_curves_cluster',
    'show_box_plot'
]


def show_tendency(
        adata: ad.AnnData,
        gene_name: Union[str, list],
        method: str = 'poly',
        one_axes: bool = True,
        palette: Union[Sequence[str], ListedColormap, dict] = None,
        norm: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 100,
        grid: bool = False,
        show: bool = True,
        save: Union[str, PathLike, None] = None,
        legend_kwargs: dict = {},
        **kwargs: Any
) -> Union[Axes, Sequence[Axes], None]:
    """
    Plot the tendency curve of the genes.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    gene_name : Union[list, str, None], optional
        The gene names for the regression model need to be calculated.
    method : Literal['poly', 'loess'], optional
        Polynomial regression(poly) or Loess regression(loess).
    one_axes : bool
        Whether to plot all curves on the same axis.
    palette : Union[Sequence[str], Cycler, dict], optional
        Colors to use for plotting annotation groups.
    norm : bool
        Normalize the values of the curves.
    figsize : Tuple[Any, Any]
        The size of format figure.(width, height).
    dpi : int
        Dots per inch values for the output.
    grid : bool
        Whether to show the grid or not.
    show : bool
        Show this plot.
    save : Union[str, PathLike]
        The path where the image is stored.
    legend_kwargs : dict, optional
        Other params of plt.legend(). Only used in 'one_axes == True'.
    kwargs
        Other params of ax.plot().

    Returns
    -------
        If show==False, return Axes or a list of Axes.
    """
    adata = copy.deepcopy(adata)

    if type(gene_name) == str:
        gene_name = [gene_name]

    if method == 'poly':
        param = _get_info_from_sample(adata, sample_id=None, key='poly')
        dic_crd = param['dic_crd_poly']

    elif method == 'loess':
        param = _get_info_from_sample(adata, sample_id=None, key='loess')
        dic_crd = param['dic_crd_loess']
    else:
        raise ValueError

    x = dic_crd['Xest']
    dic_crd.pop('Xest')
    axes = []####

    if not isinstance(palette, dict):
        palette = _get_palette(gene_name, sort_order=False, palette=palette)

    if norm:
        for gene in gene_name:
            y = dic_crd[gene]
            ygrid = []
            ygrid = (np.array(ygrid) - min(y)) / (max(y) - min(y))
            dic_crd[gene] = ygrid

    if one_axes:
        if figsize is None:
            figsize = (8, 6)

        fig = plt.figure(figsize=figsize,
                         dpi=dpi,
                         )

        ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])
        for i, gene in enumerate(gene_name):
            color = palette[gene]

            ax.plot(x, dic_crd[gene],
                    color=color,
                    label=gene,
                    **kwargs
                    )

            if 'bbox_to_anchor' not in legend_kwargs.keys():
                legend_kwargs['bbox_to_anchor'] = (1.02, 0.8)
            if 'loc' not in legend_kwargs.keys():
                legend_kwargs['loc'] = 'upper left'
            if 'title' not in legend_kwargs.keys():
                legend_kwargs['title'] = 'gene'

            ax.legend(**legend_kwargs)

        axes = ax

        if grid:
            ax.grid()
        ax.set(xlabel='distance',
               ylabel='expression',
               title=method
               )

    else:
        length = len(gene_name)

        if figsize is None:
            figsize = (8 * length, 6)

        fig = plt.figure(figsize=figsize,
                         dpi=dpi,
                         )

        gap = 0
        for i, gene in enumerate(gene_name):
            if i == 0:
                ax = fig.add_axes([0.05, 0.15, 0.8 / length, 0.7])
                gap += 0.05 + 0.8 / length
            else:
                ax = fig.add_axes([gap + 0.1/length, 0.15, 0.8 / length, 0.7])
                gap += 0.9/length

            color = palette[gene]

            ax.plot(x, dic_crd[gene],
                    color=color,
                    **kwargs
                    )

            ax.set(xlabel='distance',
                   ylabel='expression',
                   title=gene)
            if grid:
                ax.grid()
            axes.append(ax)
    plt.tight_layout()

    if save:
        fig.savefig(save)
    if show:
        fig.show()
    else:
        return axes


def show_curves_cluster(
        adata: ad.AnnData,
        method: Literal['poly', 'loess'] = 'poly',
        show_standard_error: bool = True,
        SE_alpha: Optional[float] = None,
        palette: Union[Sequence[str], ListedColormap] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 100,
        ax: Optional[Axes] = None,
        grid: bool = False,
        show: bool = True,
        save: Union[str, PathLike, None] = None,
        legend_kwargs: dict = {},
        **kwargs: Any
) -> Optional[Axes]:
    """
    Plot the tendency curve of the gene clusters.

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
    method : Literal['poly', 'loess'], optional
        Polynomial regression(poly) or Loess regression(loess).
    show_standard_error : bool
        Plot the standard error margin of the clustering curve.
    SE_alpha : bool
        Transparency between the drawn standard errors
    palette : Union[Sequence[str], ListedColormap], optional
        Colors to use for plotting annotation groups.
    title : str, optional
        The title of figure
    grid : bool
        Whether to add grid or not.
    legend_kwargs : dict, optional
        Other params of plt.legend(). Only used in 'one_axes == True'.
    kwargs : Any
        Other params of ax.plot()

    Returns
    -------
        If show==False, return Axes.
    """
    adata = copy.deepcopy(adata)

    gene_params = _get_info_from_sample(adata, sample_id=None, key=method)
    dic_crd = gene_params['dic_crd_' + method]
    param = gene_params['df_param_' + method]
    cluster_params = _get_info_from_sample(adata, sample_id=None, key='gene_cluster')
    k = cluster_params['k']
    gene_cluster = cluster_params['gene_cluster']

    if figsize is None:
        figsize = (10, 8)

    if title is None:
        title = 'Gene cluster'

    if palette is None:
        palette = color_list_50
    if isinstance(palette, ListedColormap):
        palette = palette.colors

    if ax is None:
        fig = plt.figure(figsize=figsize,
                         dpi=dpi,
                         )

        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])

    for i in range(k):

        data_new = gene_cluster[gene_cluster['cluster'] == i]
        gene_name = data_new.index

        x = dic_crd['Xest']
        x_num = len(x)
        matrix_y = []
        if method == 'poly':
            xgrid = np.linspace(min(x), max(x), num=x_num)
            for gene in gene_name:
                y = dic_crd[gene]
                params = param.loc[gene, 'param']
                ygrid = []
                for x0 in xgrid:
                    ygrid.append(sum([param * x0 ** i for i, param in enumerate(params)]))
                ygrid = (np.array(ygrid) - min(y)) / (max(y) - min(y))
                matrix_y.append(ygrid)
        else:
            samples = np.random.choice(len(x), x_num, replace=True)
            samples = np.sort(samples)
            xgrid = x[samples]
            for gene in gene_name:
                y = dic_crd[gene]
                ygrid = y[samples]
                ygrid = (np.array(ygrid) - min(y)) / (max(y) - min(y))
                matrix_y.append(ygrid)

        matrix_y = np.array(matrix_y)
        meta_y = np.mean(matrix_y, 0)
        #meta_y = (meta_y - meta_y.min()) / (meta_y.max() - meta_y.min())

        ax.plot(xgrid, meta_y, c=palette[i], label='cluster ' + str(i), **kwargs)

        if show_standard_error and matrix_y.shape[0] > 1:
            if SE_alpha is None:
                SE_alpha = 0.3
            low_CI_bound, high_CI_bound = st.t.interval(0.95, matrix_y.shape[0] - 1,
                                                        loc=meta_y,
                                                        scale=st.sem(matrix_y))

            ax.fill_between(xgrid, low_CI_bound, high_CI_bound, alpha=SE_alpha, color=palette[i])

    ax.set(title=title, xlabel='distance', ylabel='norm expression')

    if 'bbox_to_anchor' not in legend_kwargs.keys():
        legend_kwargs['bbox_to_anchor'] = (1.02, 0.8)
    if 'loc' not in legend_kwargs.keys():
        legend_kwargs['loc'] = 'upper left'
    if 'title' not in legend_kwargs.keys():
        legend_kwargs['title'] = 'gene cluster'
    ax.legend(**legend_kwargs)

    plt.tight_layout()

    if grid:
        ax.grid()
    if save:
        plt.savefig(save)
    if show:
        plt.show()
    else:
        return ax


def show_box_plot(
        adata: ad.AnnData,
        cluster_key: str,
        score_key: str,
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 100,
        ax: Optional[Axes] = None,
        show: bool = True,
        save: Union[str, PathLike, None] = None,
        **kwargs: Any,
) -> Optional[Axes]:
    """
    Boxplot of gene expression for each cluster is shown

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
    cluster_key
        cluster keyword in adata.obs.index.
    score_key
        Gene name in adata.var_names.
    show : bool
        Show the plot, do not return axis.
    save : Union[str, PathLike]
        The path where the image is stored.
    kwargs
        Other params of ax.plot().

    Returns
    -------

    """
    import pandas as pd
    import seaborn as sns

    if title is None:
        title = 'ANOVA'

    cluster = adata.obs[cluster_key]
    if score_key in adata.obs.columns:
        score = np.array(adata.obs[score_key])
    elif score_key in adata.var_names:
        index = adata.var_names.tolist().index(score_key)
        score = adata.X.toarray()[:, index]
    else:
        raise ValueError

    df = pd.DataFrame({'cluster': cluster, 'score': score})

    if ax is None:
        if figsize is None:
            figsize = (8, 6)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)

    ax = sns.boxplot(x='cluster', y='score', ax=ax, data=df, **kwargs)

    ax.set(title=title, xlabel=cluster_key, ylabel=score_key)
    if save:
        plt.savefig(save)
    if show:
        plt.show()
    else:
        return ax