from ._mask import get_mask_from_domain
from ._domain import domain_from_unsupervised, global_moran, cal_aucell, domain_from_local_moran
from ._var_genes import cal_spatialDE, cal_sparkX, cal_spark
from ._tendency import adata_from_mask, wilcoxon_test, spearman_correlation, ANOVA, spatial_tendency, gene_cluster
from ._interaction import neighborhood_analysis, infiltration_analysis, get_c_niche
from ._ccc import cell_level_communications, cell_type_level_communication, lr_pairs
from ._tensor import TensorDecomposition
