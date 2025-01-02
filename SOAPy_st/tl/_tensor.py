import pandas as pd
import numpy as np
import tensorly as tl
import anndata as ad
from anndata import AnnData
from typing import Optional, Union, Literal, Tuple
from ..utils import _check_adata_type
# from utils import insert_to_uns

__all__ = ['TensorDecomposition']


class TensorDecomposition:
    """
    Function:

    The data of anndata type is assembled tensor and tensor decomposition is carried out

    It supports the construction of high-dimensional tensors in count form using attributes in obs and the construction
    of high-dimensional tensors in expression value form using attributes in obs matching genes

    The constructed tensor can use Z-score, 0-1 regularization or proportional form for data balancing

    Tensor decomposition provides two forms of CP decomposition and tucker decomposition, both of which can choose the
    non-negative form of decomposition to assemble tensors of anndata type and conduct tensor decomposition
    """
    def __init__(self):
        """
        Initialize the tensorDecomposition object.
        """

        self.tensor = None
        self.dict_sup = None
        self.factor_name = None
        self.CP = None
        self.tucker = None

    def input_tensor(
            self,
            tensor: np.ndarray,
            factor_name: list,
            ticks_list: list
    ) -> None:
        """
        Set the input tensor, factor names, and supporting dictionary.

        Parameters
        ----------
        tensor : np.ndarray
            The input tensor for decomposition.
        factor_name : list
            A list of factor names.
        ticks_list : list
            A list of tick values corresponding to factor names.

        Returns
        -------
        None
        """
        self.tensor = tensor
        self.factor_name = factor_name
        self.dict_sup = {}
        for index, ticks in enumerate(ticks_list):
            dict_sub = {clu: i for i, clu in enumerate(ticks)}
            self.dict_sup[factor_name[index]] = dict_sub

    @property
    def get_tensor(self) -> np.ndarray:
        """
        Get the current tensor.

        Returns
        -------
        np.ndarray
            The current tensor.
        """
        return self.tensor

    def tensor_with_obs(
            self,
            adata: AnnData,
            obs_factor: list,
    ) -> None:
        """
        Construct a tensor using observation factors from an AnnData object.

        Parameters
        ----------
        adata : AnnData
            An AnnData object containing spatial omics data and spatial information.
        obs_factor : list
            A list of observation factors to use for tensor construction.

        Returns
        -------
        None
        """

        def add_array_value(arr, value, *args):
            arr[args] += value
            return arr

        self.factor_name = obs_factor
        self.dict_sup = {}
        dimensionality = []
        for label in obs_factor:
            dict_sub = {clu: i for i, clu in enumerate(adata.obs[label].unique())}
            self.dict_sup[label] = dict_sub
            dimensionality.append(len(dict_sub))
        tensor = np.zeros(dimensionality)

        for row in adata.obs.itertuples():
            index = []
            for label in obs_factor:
                index.append(self.dict_sup[label][getattr(row, label)])
            add_array_value(tensor, 1, *index)
        self.tensor = tl.tensor(tensor)

    def tensor_with_gene(
            self,
            adata: AnnData,
            obs_factor: list,
            gene_name: Optional[list] = None,
            method: Literal['mean', 'max', 'sum'] = 'mean',
    ) -> None:
        """
        Construct a tensor using observation factors and gene expression values from an AnnData object.

        Parameters
        ----------
        adata : AnnData
            An AnnData object containing spatial omics data and spatial information.
        obs_factor : list
            A list of observation factors to use for tensor construction.
        gene_name : list or None, optional
            A list of gene names to include in the tensor. If None, all genes in adata will be used.
        method : Literal['mean', 'max', 'sum']
            The method of combining expression levels,
            'mean': the mean expression levels of all cells with consistent obs_factor;
            'max': the maximum value of obs_factor expression of all cells consistent with obs_factor;
            'sum': the expression of all cells with consistent obs_factor was summed.

        Returns
        -------
        None
        """
        self.dict_sup = {}
        if gene_name is None:
            gene_name = list(adata.var_names)
        gene_dict = {gene: index for index, gene in enumerate(adata.var_names)}
        dimensionality = []

        # Create mappings for obs_factor and gene
        for label in obs_factor:
            dict_sub = {clu: i for i, clu in enumerate(adata.obs[label].unique())}
            self.dict_sup[label] = dict_sub
            dimensionality.append(len(dict_sub))
        dict_gene = {gene: i for i, gene in enumerate(gene_name)}
        self.dict_sup['gene'] = dict_gene
        dimensionality.append(len(dict_gene))

        # Initialize tensors
        tensor = np.zeros(dimensionality, dtype=np.float32)
        counts = np.zeros(dimensionality, dtype=np.int32)

        # Vectorized computation
        obs_indices = [
            adata.obs[label].map(self.dict_sup[label]).values for label in obs_factor
        ]
        obs_indices = np.array(obs_indices).T  # Shape: (n_cells, len(obs_factor))
        gene_indices = np.array([dict_gene[gene] for gene in gene_name])

        # Prepare index arrays
        obs_indices_expanded = np.repeat(obs_indices, len(gene_name), axis=0)
        gene_indices_expanded = np.tile(gene_indices, obs_indices.shape[0])

        # Flattened index computation for fast access
        flattened_indices = np.ravel_multi_index(
            (
                *obs_indices_expanded.T,
                gene_indices_expanded,
            ),
            tensor.shape,
        )

        # Update tensor and counts
        expression_values = adata.X[:, [gene_dict[gene] for gene in gene_name]].flatten()
        np.add.at(tensor, np.unravel_index(flattened_indices, tensor.shape), expression_values)
        np.add.at(counts, np.unravel_index(flattened_indices, counts.shape), 1)

        # Adjust for 'mean' method
        if method == 'mean':
            counts[counts == 0] = 1  # Prevent division by zero
            tensor /= counts

        self.factor_name = obs_factor + ['gene']
        self.tensor = tl.tensor(tensor)

    def normalization(
            self,
            factor_name: str,
            method: Literal['proportion', 'normalization', 'proportion', 'max'] = 'proportion'
    ) -> None:
        """
        Normalize the tensor along a specified factor.

        Parameters
        ----------
        factor_name : str
            The name of the factor along which normalization should be applied.
        method : Literal['z_score', 'normalization', 'proportion']
            The normalization method.

        Returns
        -------
        None
        """
        tensor = self.tensor
        shape = [index for index, name in enumerate(self.factor_name)]
        axis = self.factor_name.index(factor_name)
        shape[0], shape[axis] = shape[axis], shape[0]
        tensor = np.transpose(tensor, shape)
        dict_index = self.dict_sup[factor_name]
        if method == 'z_score':
            for key, val in dict_index.items():
                axis_mu = np.mean(tensor[val, :, :])
                axis_std = np.std(tensor[val, :, :])
                tensor[val, :, :] = (tensor[val, :, :] - axis_mu) / axis_std
        elif method == 'normalization':
            for key, val in dict_index.items():
                axis_max = np.max(tensor[val, :, :])
                axis_min = np.min(tensor[val, :, :])
                tensor[val, :, :] = (tensor[val, :, :] - axis_min) / (axis_max - axis_min)
        elif method == 'proportion':
            for key, val in dict_index.items():
                cell_number = np.sum(tensor[val, :, :])
                tensor[val, :, :] /= cell_number
        elif method == 'max':
            for key, val in dict_index.items():
                axis_max = np.max(tensor[val, :, :])
                tensor[val, :, :] = tensor[val, :, :] / axis_max
        tensor = np.transpose(tensor, shape)
        self.tensor = tensor

    def highly_variable(
            self,
            factor_name: str,
            top_num: int,
    ):
        tensor = self.tensor
        shape = [index for index, name in enumerate(self.factor_name)]
        axis = self.factor_name.index(factor_name)
        shape[0], shape[axis] = shape[axis], shape[0]
        tensor = np.transpose(tensor, shape)
        dict_index = self.dict_sup[factor_name]

        std = []
        for key, val in dict_index.items():
            std.append(np.std(tensor[val, :, :]))

        rank_std = np.argsort(-np.array(std))[0: top_num]
        tensor = tensor[rank_std, :, :]
        key_ = list(dict_index.keys())
        key_ = [key_[index] for index in rank_std]

        dict_index = {clu: i for i, clu in enumerate(key_)}
        self.dict_sup[factor_name] = dict_index

        tensor = np.transpose(tensor, shape)
        self.tensor = tensor


    def CP_decomposition(
            self,
            rank: int,
            non_negative: bool = False,
            use_hals: bool = False,
            **kwargs
    ):
        """
        Perform CP decomposition on the tensor.

        Parameters
        ----------
        rank : int
            The rank of the decomposition.
        non_negative : bool, optional
            Whether to use non-negative decomposition.
        **kwargs : Any
            additional keyword arguments for the decomposition algorithm.

        Returns
        -------
        tuple
            A tuple containing weights, factors, and normalized root mean squared error (NRE).
        """
        tensor = self.tensor
        self.CP = {}
        if non_negative:
            if use_hals:
                from tensorly.decomposition import non_negative_parafac_hals
                weights, factors = non_negative_parafac_hals(tensor=tensor,
                                                             rank=rank,
                                                             normalize_factors=True,
                                                             **kwargs)
            else:
                from tensorly.decomposition import non_negative_parafac
                weights, factors = non_negative_parafac(tensor=tensor,
                                                        rank=rank,
                                                        normalize_factors=True,
                                                        **kwargs)
        else:
            from tensorly.decomposition import parafac_power_iteration
            weights, factors = parafac_power_iteration(tensor=tensor,
                                                       rank=rank,
                                                       **kwargs)

        from tensorly.cp_tensor import cp_to_tensor
        # print(weights)
        # print(factors)
        tensor_hat = cp_to_tensor((weights, factors))
        nre = _nre_similar(tensor, tensor_hat)
        self.CP['rank'] = rank
        self.CP['weights'] = weights
        self.CP['factors'] = {}
        for index, name in enumerate(self.factor_name):
            self.CP['factors'][name] = factors[index]
        self.CP['nre'] = nre

        return weights, factors, nre

    def tucker_decomposition(
            self,
            rank: Union[int, tuple, list],
            non_negative: bool = False,
            **kwargs
    ):
        """
        Perform Tucker decomposition on the tensor.

        Parameters
        ----------
        rank : int, tuple, list
            The rank of the decomposition for each mode.
        non_negative : bool, optional
            Whether to use non-negative decomposition.
        **kwargs :
            additional keyword arguments for the tensorly.decomposition algorithm.

        Returns
        -------
        tuple
            A tuple containing core tensor, factors, and normalized root mean squared error (NRE).
        """
        tensor = self.tensor
        self.tucker = {}
        if non_negative:
            from tensorly.decomposition import non_negative_tucker
            core, factors = non_negative_tucker(tensor=tensor, rank=rank, **kwargs)
        else:
            from tensorly.decomposition import tucker
            core, factors = tucker(tensor=tensor, rank=rank, **kwargs)

        from tensorly.tucker_tensor import tucker_to_tensor
        tensor_hat = tucker_to_tensor((core, factors))
        nre = _nre_similar(tensor, tensor_hat)
        self.tucker['rank'] = {}
        if isinstance(rank, int):
            rank = [rank]*len(self.factor_name)
        for index, name in enumerate(self.factor_name):
            self.tucker['rank'][name] = rank[index]
        self.tucker['weights'] = core
        self.tucker['factors'] = {}
        for index, name in enumerate(self.factor_name):
            self.tucker['factors'][name] = factors[index]
        self.tucker['nre'] = nre

        return core, factors, nre


def _nre_similar(tensor1, tensor2) -> float:
    """
    normalized reconstruction error
    """
    # normalized reconstruction error
    from tensorly import norm
    NRE = norm(tensor1 - tensor2)/norm(tensor1)
    return NRE

