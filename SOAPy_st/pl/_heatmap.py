import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Optional, Union, Any, Mapping, Literal, Tuple
from types import MappingProxyType
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap


def get_s(pvalue, s) -> Tuple[int, str]:
    """
    Adjusts scatter point size and type based on p-value.
    """
    if pvalue > 0.5:
        new_s = s
        type = '> 0.5'
    elif 0.5 > pvalue > 0.1:
        new_s = s*2
        type = '> 0.1'
    elif 0.1 > pvalue > 0.05:
        new_s = s * 3
        type = '> 0.05'
    else:
        new_s = s * 4
        type = '< 0.05'
    return new_s, type


def get_c(score, vmin, vmax):
    """
    Adjusts the score based on given thresholds.

    Parameters
    ----------
        score : float
            Score.
        vmin : float, optional
            Minimum threshold value.
        vmin : float, optional
            Maximum threshold value.

    Returns
    -------
        float: Adjusted score.
    """
    if vmax is not None and score > vmax:
        score = vmax
    if vmin is not None and score < vmin:
        score = vmin
    return score


def _dotplot(
        scores: pd.DataFrame,
        pvalue: pd.DataFrame,
        s_cutoff: float = 0.05,
        p_cutoff: float = 3.0,
        figsize = (6.4, 5.5),
        vmax: float = None,
        vmin: float = None,
        ax=None,
        cmap=None,
        sig_color='#000000',
        s: int = 10,
        label_fontsize=10,
        **kwargs
) -> Optional[Axes]:
    """
    Main underlying helper function for generating heatmaps.

    Parameters
    ----------
    scores : pd.DataFrame
        Score DataFrame.
    pvalue : pd.DataFrame
        P-value DataFrame.
    s_cutoff : float, optional
        Cutoff value for scatter point size.
    p_cutoff : float, optional
        Cutoff value for p-values.
    figsize : Tuple[float, float], optional
        Figure size.
    vmax : float, optional
        Maximum threshold value.
    vmin : float, optional
        Minimum threshold value.
    ax : Axes, optional
        Matplotlib Axes object.
    cmap : str, optional
        Color map.
    sig_color : str, optional
        Color for significant points.
    s : int, optional
        Initial scatter point size.
    label_fontsize : int, optional
        Font size for labels.
    kwargs: Any, optional
        Other params of ax.scatter()

    Returns
    -------
    Axes
        Generated Axes object.
    """
    if type(cmap) == type(None):
        cmap = "Spectral_r"

    if type(ax) == type(None):
        fig, ax = plt.subplots(figsize=figsize)

    n_rows = scores.shape[0] * scores.shape[1]
    flat_df = pd.DataFrame(index=list(range(n_rows)), columns=['x', 'y', 'value', 'p', 'type', 'sig'])
    i = 0
    for index_ct, ct in enumerate(scores.columns):
        for index_lr, lr in enumerate(scores.index):
            new_p, label = get_s(pvalue.values[index_lr, index_ct], s)
            score = get_c(scores.values[index_lr, index_ct], vmin=vmin, vmax=vmax)
            if pvalue.values[index_lr, index_ct] < p_cutoff and scores.values[index_lr, index_ct] > s_cutoff:
                flat_df.iloc[i, :] = [ct, lr, score, new_p, label, sig_color]
            else:
                flat_df.iloc[i, :] = [ct, lr, score, new_p, label, '#FFFFFF']
            i += 1

    x = flat_df['x']
    y = flat_df['y']

    x_labels = list(x.values)
    y_labels = list(y.values)
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    out = ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        c=flat_df['value'].tolist(),
        s=flat_df['p'].tolist(),
        edgecolor=flat_df['sig'].tolist(),
        cmap=cmap,
        marker="o",
        **kwargs
    )

    out.set_array(flat_df['value'].values.astype(int))
    out.set_clim(min(flat_df['value']), max(flat_df['value']))
    cbar = plt.colorbar(out)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("lr score", rotation=270)

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in np.unique(x_labels)])
    ax.set_xticklabels(np.unique(x_labels), rotation=45, horizontalalignment="right", fontsize=label_fontsize)
    ax.set_yticks([y_to_num[v] for v in np.unique(y_labels)])
    ax.set_yticklabels(np.unique(y_labels), fontsize=label_fontsize)

    # x_label = ax.get_xticklabels()
    # [x_label_temp.set_fontname('Times New Roman') for x_label_temp in x_label]
    # y_label = ax.get_yticklabels()
    # [y_label_temp.set_fontname('Times New Roman') for y_label_temp in y_label]

    return ax


def _norm_mat(data, axis, method):
    if axis == 1:
        data = data.T
    if method == 'z_score':
        for i in range(data.shape[1]):
            data[:, i] = (data[:, i] - data[:, i].mean()) / data[:, i].std()
    elif method == 'normalization':
        for i in range(data.shape[1]):
            data[:, i] = (data[:, i] - min(data[:, i])) / \
                                     (max(data[:, i]) - min(data[:, i]))
    elif method == 'proportion':
        for i in range(data.shape[1]):
            data[:, i] = (data[:, i]) / sum(data[:, i])
    if axis == 1:
        data = data.T
    return data


def _consensus_list(mat, ax, metric='euclidean', method='average', axis=0):
    """
    Generate dendrogram.

    Parameters
    ----------
    mat : np.ndarray
        Input matrix.
    ax : Axes
        Matplotlib Axes object.
    metric : str, optional
        Distance metric.
    method : str
        Hierarchical clustering method.
    axis : int, optional
        Clustering axis.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Reordered indices and linkage matrix.
    """
    from scipy.spatial import distance
    from scipy.cluster import hierarchy

    if axis != 0 and axis !=1:
        ValueError('The axis must be either 0 or 1')
    if axis == 1:
        mat = mat.T
    pairwise_dists = distance.pdist(mat, metric=metric)
    linkage = hierarchy.linkage(pairwise_dists, method=method)
    if axis == 0:
        dendrogram = hierarchy.dendrogram(linkage, color_threshold=-np.inf, ax=ax, orientation='left')
    else:
        dendrogram = hierarchy.dendrogram(linkage, color_threshold=-np.inf, ax=ax)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    reordered_ind = dendrogram['leaves']
    return reordered_ind, linkage


def _heatmap_with_dendrogram_and_bar(
        data: np.ndarray,
        data_count: np.ndarray,
        x_label: str,
        y_label: str,
        x_map: dict = None,
        y_map: dict = None,
        x_dendrogram: bool = True,
        y_dendrogram: bool = True,
        x_bar: bool = True,
        y_bar: bool = True,
        method: Optional[Literal['z_score', 'normalization', 'proportion']] = 'normalization',
        norm_axis: Optional[Literal[0, 1]] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 100,
        cmap: LinearSegmentedColormap = None,
        xbar_kwags: Mapping[str, Any] = MappingProxyType({}),
        ybar_kwags: Mapping[str, Any] = MappingProxyType({}),
        **kwargs
) -> Tuple[Figure, dict]:
    """
    Display heatmap with two counting barplots.

    Parameters
    ----------
    data : np.ndarray
        Data.
    data_count : np.ndarray
        Count data.
    x_label : str
        x-axis label.
    y_label : str
        y-axis label.
    x_map : Optional[dict], optional
        x-axis mapping.
    y_map : Optional[dict], optional
        y-axis mapping.
    x_dendrogram : bool, optional
        Whether to show x-axis dendrogram.
    y_dendrogram : bool, optional
        Whether to show y-axis dendrogram.
    x_bar : bool, optional
        Whether to show x-axis count barplot.
    y_bar : bool, optional
        Whether to show y-axis count barplot.
    method : Optional[Literal['z_score', 'normalization', 'proportion']], optional
        Normalization method.
    norm_axis : Optional[Literal[0, 1]], optional
        Normalization axis.
    title : Optional[str], optional
        Figure title.
    figsize : Optional[Tuple[float, float]], optional
        Figure size.
    dpi : int, optional
        Image resolution.
    cmap : Optional[LinearSegmentedColormap], optional
        Color map for continuous variables.
    xbar_kwags : Mapping[str, Any], optional
        Parameters for x-axis count barplot.
    ybar_kwags : Mapping[str, Any], optional
        Parameters for y-axis count barplot.
    kwargs
        Other parameters.

    Returns
    -------
    Tuple[Figure, List[Axes]]
        Generated Figure object and list of Axes.
    """
    import seaborn as sns

    norm_data = copy.deepcopy(data)
    axes = {}

    if norm_axis is not None:
        norm_data = _norm_mat(norm_data, norm_axis, method)

    if figsize is None:
        figsize = (12, 8)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.suptitle(title, fontsize=20, x=0.5, y=0.93)

    gs = GridSpec(40, 40)

    ax_heatmap = fig.add_subplot(gs[4:29, 4:29])
    ax_cbar = fig.add_subplot(gs[0:3, 2:3])
    axes['ax_heatmap'] = ax_heatmap
    axes['ax_cbar'] = ax_cbar

    if x_dendrogram:
        ax_dendrogram_x = fig.add_subplot(gs[1:4, 4:29])
        new_xtick, linkage_x = _consensus_list(norm_data, ax_dendrogram_x,
                                               metric='euclidean', method='average', axis=1)
        axes['ax_dendrogram_x'] = ax_dendrogram_x
    else:
        new_xtick = range(norm_data.shape[1])
    if y_dendrogram:
        ax_dendrogram_y = fig.add_subplot(gs[4:29, 1:4])
        new_ytick, linkage_y = _consensus_list(norm_data, ax_dendrogram_y,
                                               metric='euclidean', method='average', axis=0)
        axes['ax_dendrogram_y'] = ax_dendrogram_y
    else:
        new_ytick = range(norm_data.shape[0])

    sns.heatmap(norm_data[np.ix_(new_ytick, new_xtick)],
                ax=ax_heatmap, cmap=cmap, cbar_ax=ax_cbar,
                **kwargs)

    len_x = len(new_xtick)
    len_y = len(new_ytick)
    if x_map is not None:
        dic_new = dict(zip(x_map.values(), x_map.keys()))
        new_xtick = [dic_new[i] for i in new_xtick]
    if y_map is not None:
        dic_new = dict(zip(y_map.values(), y_map.keys()))
        new_ytick = [dic_new[i] for i in new_ytick]

    if x_bar:
        ax_heatmap.set_xticks([])
        xbar_kwags = dict(xbar_kwags)
        bottom_hist = fig.add_subplot(gs[30:38, 4:29])
        sns.countplot(x=x_label, data=data_count, ax=bottom_hist, order=new_xtick, **xbar_kwags)
        # bottom_hist.set_xticks(np.arange(len_x), new_xtick)
        bottom_hist.xaxis.set_ticks_position('top')
        bottom_hist.invert_yaxis()
        bottom_hist.spines['right'].set_visible(False)
        bottom_hist.spines['bottom'].set_visible(False)
        bottom_hist.spines['left'].set_visible(False)
        bottom_hist.set_xlabel(x_label)
    else:
        ax_heatmap.set_xticks(np.arange(len_x), new_xtick)

    if y_bar:
        ax_heatmap.set_yticks([])
        ybar_kwags = dict(ybar_kwags)
        right_hist = fig.add_subplot(gs[4:29, 30:38])
        sns.countplot(y=y_label, data=data_count, ax=right_hist, orient='h', order=new_ytick, **ybar_kwags)
        # right_hist.set_yticks(np.arange(len_y), new_ytick)
        right_hist.set_ylabel(y_label, rotation=270)
        right_hist.yaxis.set_label_position("right")
        right_hist.spines['top'].set_visible(False)
        right_hist.spines['right'].set_visible(False)
        right_hist.spines['bottom'].set_visible(False)
    else:
        ax_heatmap.set_yticks(np.arange(len_y), new_ytick)

    return fig, axes