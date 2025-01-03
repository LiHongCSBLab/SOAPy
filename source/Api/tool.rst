.. module:: SOAPy_st
.. automodule:: SOAPy_st
   :noindex:

tl
============================

Spatial domain
-------------------

.. module:: SOAPy_st.tl
.. currentmodule:: SOAPy_st

.. autosummary::
   :toctree: .

    tl.domain_from_unsupervised
    tl.cal_aucell
    tl.domain_from_local_moran
    tl.global_moran

Mask
-------------------

.. module:: SOAPy_st.tl
.. currentmodule:: SOAPy_st

.. autosummary::
   :toctree: .

    tl.adata_from_mask
    tl.get_mask_from_domain

Spatial tendency
-------------------

.. module:: SOAPy_st.tl
.. currentmodule:: SOAPy_st

.. autosummary::
   :toctree: .

    tl.ANOVA
    tl.wilcoxon_test
    tl.spearman_correlation
    tl.spatial_tendency
    tl.gene_cluster
    tl.cal_spatialDE
    tl.cal_sparkX
    tl.cal_spark

Cell type proximity
-------------------

.. module:: SOAPy_st.tl
.. currentmodule:: SOAPy_st

.. autosummary::
   :toctree: .

   tl.neighborhood_analysis
   tl.infiltration_analysis

Niche composition
-----------------

.. module:: SOAPy_st.tl
.. currentmodule:: SOAPy_st

.. autosummary::
   :toctree: .

    tl.get_c_niche

Spatial communication
---------------------

.. module:: SOAPy_st.tl
.. currentmodule:: SOAPy_st

.. autosummary::
   :toctree: .

    tl.lr_pairs
    tl.cell_level_communications
    tl.cell_type_level_communication

Tensor decomposition
---------------------

.. module:: SOAPy_st.tl
.. currentmodule:: SOAPy_st

.. autosummary::
   :toctree: .

    tl.TensorDecomposition
