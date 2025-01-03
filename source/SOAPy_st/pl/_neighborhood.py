import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
from typing import Union, Optional, Tuple, Literal, Any
from os import PathLike
from ..utils import _get_info_from_sample
import matplotlib
import matplotlib.patches as patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = ['show_neighborhood_analysis', 'show_infiltration_analysis']


def show_neighborhood_analysis(
        adata: ad.AnnData,
        method: Literal['excluded', 'include'] = 'excluded',
        sample_id: Union[int, str] = None,
        figsize: Optional[Tuple[float, float]] = (8, 8),
        dpi: int = 100,
        title: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: Any = None,
        show: bool = True,
        save: Union[str, PathLike, None] = None,
        ax: Optional[Axes] = None,
        **kwargs: Any
) -> Optional[Axes]:
    """
    Use heatmap to display the results of neighbor analysis.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    method : str, optional
        'included': Z-scores of edges between two cell types were counted directly after randomization.
        'excluded': After randomization, remove self-connected edges between cells of the same type and calculate the z-score of edges between two cell types.
    sample_id : Union[int, str], optional
        The sample number which to be shown.
    figsize : Tuple[float, float], optional
        (Width, height) of the figure.
    dpi : float, optional
        The resolution of the figure.
    title : str, optional
        The title of shown figure.
    vmin : float, optional
        Lower end of scale bar.
    vmax : float, optional
        Upper end of scale bar.
    cmap : Any, optional
        Color map to use for continuous variables.
    show : bool, optional
        Show the plot, do not return axis.
    save : Union[str, PathLike], optional
        The path where the image is stored.
    ax : Axes, optional
        A matplotlib axes object.
    kwargs : Any
        Other params of sns.heatmap().

    Returns
    -------
    Optional[Axes]
        The matplotlib axes object if `show` is False, otherwise None.
    """
    import seaborn as sns

    if title is None:
        title = 'Neighborhood analysis'

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True, dpi=dpi, figsize=figsize)

    zscore_df = _get_info_from_sample(adata, sample_id=sample_id, key=method + '_score')
    label = zscore_df.index
    zscore = zscore_df.values

    sns.heatmap(zscore,
                cmap=cmap,
                xticklabels=label,
                yticklabels=label,
                fmt='d',
                cbar=True,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
                **kwargs)

    ax.set_title(title, fontsize=20)

    if save:
        plt.savefig(save)
    if show:
        plt.show()
    else:
        return ax


def show_infiltration_analysis(
        adata: ad.AnnData,
        parenchyma: str,
        nonparenchyma: str,
        min_nonparenchyma: Optional[int] = None,
        sample: Union[list, None] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 100,
        color_num: str = 'red',
        color_score: str = 'blue',
        grid: bool  = True,
        show: bool = True,
        save: Union[str, PathLike, None] = None,
) -> Optional[Tuple[Axes, Axes]]:
    """
    Use barplot to display infiltration scores and the number of non-parenchymal cells.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    parenchyma : str
        The cluster name of parenchyma cell.
    nonparenchyma : str
        The cluster name of non-parenchyma cell.
    min_nonparenchyma

    sample : Union[list, None]
        Samples that need to be presented with infiltration scores.
        By default, all samples for which the infiltration score was calculated are displayed.
    figsize : Tuple[Any, Any]
        The size of format figure.(width, height).
    dpi : int
        Dots per inch values for the output.
    color_num : str
        The color of cells number of non-parenchyma.
    color_score : str
        The color of infiltration score.
    grid : bool
        Whether to add grid or not.
    show : bool
        Show the plot, do not return axis.
    save : Union[str, PathLike]
        The path where the image is stored.

    Returns
    -------
        If show==False, return a tuple of two Axes. ax_left is the barplot Axes of nonparenchyma number,
            and ax_right is the barplot Axes of infiltration score.

    References
    -------
        Nicolas P. Rougier. Scientific Visualization: Python + Matplotlib. Nicolas P. Rougier. , 2021, 978-2-
            9579901-0-8. ffhal-03427242
    """
    data = pd.DataFrame(columns=['sample', 'infiltration score', 'non-number'])

    if sample is None:
        sample = _get_info_from_sample(adata, sample_id=None, key='infiltration_sample')
    for sample_id in sample:
        infiltration = _get_info_from_sample(adata, sample_id=sample_id, key='infiltration_analysis')
        number = _get_info_from_sample(adata, sample_id=sample_id, key='cluster_number')

        data.loc[len(data.index)] = [sample_id, infiltration.loc[parenchyma, nonparenchyma], number[nonparenchyma]]

    max_score = np.floor(max(data['infiltration score']) * 1.2) + 1
    max_number = np.floor(max(data['non-number']) * 1.2) + 1

    data1 = copy.deepcopy(data)
    data1 = data1[data1['non-number'] >= min_nonparenchyma]
    # data1 = data1.sort_values(by='min score', ascending=False)

    right = np.array(data1['infiltration score'].tolist())
    left = np.array(data1['non-number'].tolist())
    sample = data1['sample'].tolist()

    matplotlib.rc("axes", facecolor="white")
    matplotlib.rc("figure.subplot", wspace=0.65)
    matplotlib.rc("grid", color="white")
    matplotlib.rc("grid", linewidth=1)

    fig = plt.figure(figsize=figsize, facecolor="white", dpi=dpi)

    ax_left = plt.subplot(121)

    ax_left.spines["left"].set_color("none")
    ax_left.spines["right"].set_zorder(10)
    ax_left.spines["bottom"].set_color("none")
    ax_left.xaxis.set_ticks_position("top")
    ax_left.yaxis.set_ticks_position("right")
    ax_left.spines["top"].set_position(("data", len(sample) + 0.25))
    ax_left.spines["top"].set_color("w")

    plt.xlim(max_number, 0)
    plt.ylim(0, len(sample))

    plt.xticks([max_number, max_number//2, 0], [str(max_number), str(max_number//2), nonparenchyma])
    ax_left.get_xticklabels()[-1].set_weight("bold")
    ax_left.get_xticklines()[-1].set_markeredgewidth(0)
    for label in ax_left.get_xticklabels():
        label.set_fontsize(10)
    plt.yticks([])

    # Plot data
    for i in range(len(left)):
        H, h = 0.8, 0.55
        value = left[i]
        p = patches.Rectangle(
            (0, i + (1 - H) / 2.0),
            value,
            H,
            fill=True,
            transform=ax_left.transData,
            lw=0,
            facecolor=color_num,
            alpha=0.6,
        )
        ax_left.add_patch(p)

    if grid:
        ax_left.grid()

    ax_right = plt.subplot(122, sharey=ax_left)

    ax_right.spines["right"].set_color("none")
    ax_right.spines["left"].set_zorder(10)
    ax_right.spines["bottom"].set_color("none")
    ax_right.xaxis.set_ticks_position("top")
    ax_right.yaxis.set_ticks_position("left")
    ax_right.spines["top"].set_position(("data", len(sample) + 0.25))
    ax_right.spines["top"].set_color("w")

    plt.xlim(0, 6)
    plt.ylim(0, len(sample))

    plt.xticks(
        [0, max_score//2, max_score],
        ["Infiltration", str(max_score//2), str(max_score)],
    )
    ax_right.get_xticklabels()[0].set_weight("bold")
    for label in ax_right.get_xticklabels():
        label.set_fontsize(10)
    ax_right.get_xticklines()[1].set_markeredgewidth(0)
    plt.yticks([])

    for i in range(len(right)):
        H, h = 0.8, 0.55
        value = right[i]
        p = patches.Rectangle(
            (0, i + (1 - H) / 2.0),
            value,
            H,
            fill=True,
            transform=ax_right.transData,
            lw=0,
            facecolor=color_score,
            alpha=0.6,
        )
        ax_right.add_patch(p)

    if grid:
        ax_right.grid()

    for i in range(len(sample)):
        x1, y1 = ax_left.transData.transform_point((0, i + 0.5))
        x2, y2 = ax_right.transData.transform_point((0, i + 0.5))
        x, y = fig.transFigure.inverted().transform_point(((x1 + x2) / 2, y1))
        sample_id = sample[i]
        plt.text(
            x,
            y,
            sample_id,
            transform=fig.transFigure,
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="center",
        )
    # fig.tight_layout()

    axes = (ax_left, ax_right)

    if save:
        fig.savefig(save)
    if show:
        fig.show()
    else:
        return axes