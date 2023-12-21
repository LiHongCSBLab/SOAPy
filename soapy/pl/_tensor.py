import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, Union
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from ..tl._tensor import TensorDecomposition

__all__ = [
    'show_factor_matrix_in_CP_tensor',
    'show_factor_matrix_in_tucker_tensor',
    'show_proportion_in_CP',
    'show_proportion_in_tucker'
]


def show_factor_matrix_in_CP_tensor(
        tensor: TensorDecomposition,
        factor_name: str,
        **kwargs
):
    """
    Visualizes a factor in the CP decomposition tensor.

    Parameters
    ----------
    tensor : TensorDecomposition
        The TensorDecomposition object containing the CP decomposition.
    factor_name : str
        The name of the factor to visualize.
    **kwargs : dict
        Additional keyword arguments for seaborn.clustermap.

    Returns
    -------
    None
    """

    tensor_object = tensor
    rank = tensor_object.CP['rank']
    factor = tensor_object.CP['factors'][factor_name]
    dict_label = tensor_object.dict_sup[factor_name]
    _show_factor(factor, dict_label, rank, factor_name, **kwargs)


def show_factor_matrix_in_tucker_tensor(
        tensor: TensorDecomposition,
        factor_name: str,
        **kwargs
):
    """
    Visualizes a factor in the Tucker decomposition tensor.

    Parameters
    ----------
    tensor : TensorDecomposition
        The TensorDecomposition object containing the Tucker decomposition.
    factor_name : str
        The name of the factor to visualize.
    **kwargs : dict
        Additional keyword arguments for seaborn.clustermap.

    Returns
    -------
    None
    """

    tensor_object = tensor
    factor_rank = tensor_object.tucker['rank'][factor_name]
    factor = tensor_object.tucker['factors'][factor_name]
    dict_label = tensor_object.dict_sup[factor_name]
    _show_factor(factor, dict_label, factor_rank, factor_name, **kwargs)


def show_proportion_in_CP(
        tensor: TensorDecomposition,
        module: int,
        figsize: Optional[tuple] = None,
        factor_name: Union[list, str, None] = None,
        max_key: Union[None, int, list] = None,
        order: bool = True,
        show: bool = True,
        save: Optional[str] = None,
        **kwargs
):
    """
    Visualizes the proportion in a CP decomposition.

    Parameters
    ----------
    tensor : TensorDecomposition
        The TensorDecomposition object containing the CP decomposition.
    module : int
        The module to visualize.
    figsize : tuple, optional
        Figure size.
    factor_name : Union[list, str, None], optional
        The name of the factor to visualize.
    max_key : Union[None, int, list], optional
        Maximum number of keys to show.
    order : bool, optional
        Whether to order the keys.
    show : bool, optional
        Whether to display the figure.
    save : str, optional
        Filepath to save the figure.
    **kwargs : dict
        Additional keyword arguments for seaborn.barplot.

    Returns
    -------
    None or list of Axes
        If show is False, returns the list of Axes.
    """

    tensor_object = tensor
    factors = tensor_object.CP['factors']
    dict_sup = tensor_object.dict_sup
    if factor_name is None:
        factor_name = tensor_object.factor_name

    if not isinstance(factor_name, list):
        factor_name = [factor_name]
    if not isinstance(max_key, list):
        max_key = [max_key]*len(factor_name)
    if figsize is None:
        figsize = (6 * len(factor_name), 6)
    fig, axes = plt.subplots(1, len(factors), figsize=figsize)
    for index, name in enumerate(factor_name):
        _show_proportion(factor=factors[name],
                         module=module,
                         dict_label=dict_sup[name],
                         factor_name=name,
                         ax=axes[index],
                         max_key=max_key[index],
                         order=order,
                         **kwargs)
    fig.suptitle('module '+str(module))
    fig.tight_layout()
    if save:
        fig.savefig(save)
    if show:
        fig.show()
    else:
        return axes


def show_proportion_in_tucker(
        tensor: TensorDecomposition,
        module: list,
        figsize: Optional[tuple] = None,
        factor_name: Union[list, str, None] = None,
        max_key: Union[None, int, list] = None,
        order: bool = True,
        show: bool = True,
        save: Optional[str] = None,
        **kwargs
):
    """
    Visualizes the proportion in a Tucker decomposition.

    Parameters
    ----------
    tensor : TensorDecomposition
        The TensorDecomposition object containing the Tucker decomposition.
    module : list
        List of modules to visualize.
    figsize : tuple, optional
        Figure size.
    factor_name : Union[list, str, None], optional
        The name of the factor to visualize.
    max_key : Union[None, int, list], optional
        Maximum number of keys to show.
    order : bool, optional
        Whether to order the keys.
    show : bool, optional
        Whether to display the figure.
    save : str, optional
        Filepath to save the figure.
    **kwargs : dict
        Additional keyword arguments for seaborn.barplot.

    Returns
    -------
    None or list of Axes
        If show is False, returns the list of Axes.
    """
    tensor_object = tensor
    factors = tensor_object.CP['factors']
    dict_sup = tensor_object.dict_sup
    if factor_name is None:
        factor_name = tensor_object.factor_name

    if not isinstance(factor_name, list):
        factor_name = [factor_name]
    if not isinstance(max_key, list):
        max_key = [max_key] * len(factor_name)
    if figsize is None:
        figsize = (6 * len(factor_name), 6)
    fig, axes = plt.subplots(1, len(factors), figsize=figsize)
    for index, name in enumerate(factor_name):
        _show_proportion(factors[name],
                         module[index],
                         dict_sup[name], name, ax=axes[index], max_key=max_key[index],
                         order=order, **kwargs)

    plt.suptitle('module '+str(module))
    plt.show()
    if save:
        fig.savefig(save)
    if show:
        fig.show()
    else:
        return axes


def _show_factor(factor: np.ndarray,
                 dict_label: dict,
                 rank: int,
                 factor_name: str,
                 **kwargs
                 ):
    """
    Display a factor using seaborn.clustermap.

    Parameters
    ----------
    factor : np.ndarray
        The factor to display.
    dict_label : dict
        Dictionary of labels.
    rank : int
        Rank of the factor.
    factor_name : str
        Name of the factor.
    **kwargs : dict
        Additional keyword arguments for seaborn.clustermap.

    Returns
    -------
    None
    """

    index_name = list(dict_label.keys())
    col_name = ['factor ' + str(i) for i in range(rank)]
    tensor_df = pd.DataFrame(factor, index=index_name, columns=col_name)
    sns.clustermap(tensor_df, col_cluster=False, **kwargs)
    plt.xticks(rotation=45)
    plt.suptitle(factor_name)
    plt.show()


def _show_proportion(factor: np.ndarray,
                     module,
                     dict_label: dict,
                     factor_name,
                     ax,
                     max_key: Optional[int] = None,
                     order: bool = True,
                     **kwargs
                     ):
    """
    Display the proportion using seaborn.barplot.

    Parameters
    ----------
    factor : np.ndarray
        The factor to display.
    module : int
        The module to visualize.
    dict_label : dict
        Dictionary of labels.
    factor_name : str
        Name of the factor.
    ax : matplotlib.axes._subplots.AxesSubplot
        Matplotlib axes for plotting.
    max_key : Optional[int], optional
        Maximum number of keys to show.
    order : bool, optional
        Whether to order the keys.
    **kwargs : dict
        Additional keyword arguments for seaborn.barplot.

    Returns
    -------
    None
    """

    x = np.array(list(dict_label.keys()))
    y = np.array(factor[:, module])
    if order:
        index = np.argsort(y)[::-1]
        x = x[index]
        y = y[index]
    if max_key is not None:
        if len(x) > max_key:
            x = x[: max_key]
            y = y[: max_key]
    sns.barplot(x=x, y=y, ax=ax, **kwargs)
    ax.set_xticklabels(x, rotation=20, fontsize=10)
    ax.set_title(factor_name)