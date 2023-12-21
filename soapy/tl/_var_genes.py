import logging
import anndata
import numpy as np
import pandas as pd
from typing import Union, Literal
from ..utils import _check_adata_type

__all__ = ["cal_spatialDE", "cal_sparkX", "cal_spark"]


def _get_gene_list(
        gene_name,
        all_genes,
) -> list:

    if gene_name is None:
        gene_name = all_genes
    if isinstance(gene_name, str):
        gene_name = [gene_name]

    is_in_list = np.isin(gene_name, all_genes)
    if not all(is_in_list):
        gene_not_in_list = gene_name[is_in_list]
        logging.error(f'{gene_not_in_list} not in adata.var_names.', exc_info=True)
    return gene_name


def cal_spatialDE(
        adata: anndata.AnnData,
        gene_name: Union[list, str, None] = None,
        spatial_in_obsm: str = 'spatial',
) -> pd.DataFrame:
    """
    Calculate SpatialDE analysis on spatial omics data.
    Doi: 10.1038/nmeth.4636
    Github: https://github.com/Teichlab/SpatialDE

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    gene_name : Union[list, str, None], optional
        List of genes of interest or a single gene name to focus the analysis on.
        If None, the analysis will consider all genes in the dataset.
    spatial_in_obsm : str
        The key of spatial coordinates in adata.obsm

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the results of the SpatialDE analysis.

    """
    import SpatialDE

    adata = _check_adata_type(adata, spatial_in_obsm, True)

    gene_name = _get_gene_list(gene_name=gene_name, all_genes=adata.var_names.tolist())

    exp_mat = pd.DataFrame(adata.X.todense(), columns=gene_name, index=adata.obs_names)
    coord = adata.obsm[spatial_in_obsm]
    results = SpatialDE.run(coord, exp_mat)

    return results


def cal_sparkX(
        adata: anndata.AnnData,
        gene_name: Union[list, str, None] = None,
        num_core: int = 5,
        spatial_in_obsm: str = 'spatial',
) -> pd.DataFrame:
    """
    Calculate SPARK-X analysis on spatial omics data.
    Doi: 10.1186/s13059-021-02404-0
    Github: https://github.com/xzhoulab/SPARK

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    gene_name : Union[list, str, None], optional
        List of genes of interest or a single gene name to focus the analysis on.
        If None, the analysis will consider all genes in the dataset.
    num_core : int
        The number of CPU core.
    spatial_in_obsm : str
        The key of spatial coordinates in adata.obsm

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the results of the SPARK-X analysis.

    """

    # load required packages
    from rpy2.robjects import r
    from rpy2.robjects import pandas2ri
    import rpy2.robjects as ro
    pandas2ri.activate()

    adata = _check_adata_type(adata, spatial_in_obsm, True)

    gene_name = _get_gene_list(gene_name=gene_name, all_genes=adata.var_names.tolist())
    count_mat = pd.DataFrame(adata.X.todense(), columns=gene_name, index=adata.obs_names).transpose()
    coord_mat = adata.obsm[spatial_in_obsm]

    RunSPARKX = r(f"""
        function(count_mat,coord_mat){{
        library(SPARK)
        count_mat=as.matrix(count_mat)
        count_mat=Matrix::Matrix(count_mat,sparse=TRUE)
        sparkX_result=sparkx(count_mat,as.matrix(coord_mat),numCores={num_core},option="mixture")
        return(list(data.frame(sparkX_result[["res_mtest"]],check.names=FALSE),rownames(count_mat)))
    }}
    """)

    test_result = RunSPARKX(count_mat, coord_mat)
    test_result[0].index = test_result[1]
    test_result = test_result[0]
    test_result = ro.conversion.rpy2py(test_result)

    return test_result


def cal_spark(
        adata: anndata.AnnData,
        gene_name: Union[list, str, None] = None,
        method: Literal['gaussian', 'poisson'] = 'gaussian',
        num_core: int = 5,
        spatial_in_obsm: str = 'spatial',
) -> pd.DataFrame:
    """
    Perform SPARK analysis on spatial omics data.
    Doi: 10.1038/s41592-019-0701-7
    Github: https://github.com/xzhoulab/SPARK

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    gene_name : Union[list, str, None], optional
        List of genes of interest or a single gene name to focus the analysis on.
        If None, the analysis will consider all genes in the dataset.
    method : Literal['gaussian', 'poisson'], optional
        The statistical method to be used for the analysis. Must be one of 'gaussian' or 'poisson'.
    num_core : int
        The number of CPU core.
    spatial_in_obsm : str
        The key of spatial coordinates in adata.obsm

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the results of the SPARK analysis.
    """

    # load required packages
    from rpy2.robjects import r
    from rpy2.robjects import pandas2ri
    import rpy2.robjects as ro
    pandas2ri.activate()

    adata = _check_adata_type(adata, spatial_in_obsm, True)

    gene_name = _get_gene_list(gene_name=gene_name, all_genes=adata.var_names.tolist())

    count_mat = pd.DataFrame(adata.X.todense(), columns=gene_name, index=adata.obs_names).transpose()
    coord_mat = adata.obsm[spatial_in_obsm]

    def generate_spark_r_function(method):
        r_code = f"""
            function(count_mat, coord_mat) {{
                library(SPARK)
                rownames(coord_mat) = colnames(count_mat)
                spark_obj = CreateSPARKObject(counts = count_mat, location = coord_mat, percentage = 0, min_total_counts = 0)
                spark_obj@lib_size = apply(spark_obj@counts, 2, sum)
                spark_obj = spark.vc(spark_obj, covariates = NULL, fit.model = '{method}', num_core = '{num_core}', verbose = F)
                spark_obj = spark.test(spark_obj, check_positive = T, verbose = F)
                return(spark_obj@res_mtest)
            }}
        """
        return r_code

    test_result = r(generate_spark_r_function(method=method))(count_mat, coord_mat)
    test_result = ro.conversion.rpy2py(test_result)

    return test_result