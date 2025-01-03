"""
Spatial Omics includes barcode-based spatial transcriptomics, image-based spatial transcriptomics ,spatial proteomics and
spatial metabolomics. Barcode-based data is easily converted to Anndata format, but image-based data need Pre-processing to
convert to Anndata. We developed a model named all2adata to convert all spatial omics data to Anndata format.
"""

import os
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import sparse
from os import PathLike
from typing import Optional, Union, Literal
from pathlib import Path
import scipy.io as sio

__all__ = ["read_csv2adata", "read_mult_image2adata", "read_visium2adata", "read_dsp2adata", "st_Seurat2Anndata"]

###############################Spatial Omics to Anndata##############################


def read_csv2adata(
        express: Union[PathLike, str, pd.DataFrame],
        meta: Union[PathLike, str, pd.DataFrame] = None,
        spatial: Union[np.ndarray, list] = None,
        express_kwargs: dict = {},
        meta_kwargs: dict = {},
) -> anndata.AnnData:
    """
    To read Barcode_based spatial omics data.
    Slide seqV2, Stereo seq and other table data

    Parameters
    ----------
    express : Union[PathLike, str, pd.DataFrame]
        Count matrix file or its path.
    location : Union[PathLike, str, pd.DataFrame]
        Coordinates matrix file or its path.
    scale : float
        The scale of the coordinates, usually for visual matching with different resolutions of HE or other images,
        is not scaled by default.
    express_kwargs : dict
        Other params of pd.read_csv(express)
    location_kwargs : dict
        Other params of pd.read_csv(location)

    Returns
    -------
        anndata.Anndata object
    """

    if type(express) is not pd.DataFrame:
        express = pd.read_csv(express, **express_kwargs)

    if type(meta) is not pd.DataFrame and meta is not None:
        meta = pd.read_csv(meta, **meta_kwargs)

    adata = sc.AnnData(
        express.values,
        obs=meta,
        var=pd.DataFrame(index=express.columns),
        obsm={},
    )
    # sc.pp.calculate_qc_metrics(adata, inplace=True)

    if isinstance(spatial, np.ndarray):
        adata.obsm['spatial'] = spatial
    elif isinstance(spatial, list):
        if len(spatial) != 2:
            raise ValueError('The data dimension must be equal to 2')
        i, j = spatial
        assert (i in list(meta.columns) and j in list(meta.columns)), 'The coordinate information should be included in the meta'
        adata.obsm['spatial'] = meta.loc[:, [i, j]].values

    adata.var_names_make_unique()
    adata.X = sparse.csr_matrix(adata.X)

    return adata


def read_mult_image2adata(
        image: Union[np.ndarray, str, PathLike],
        mask: Union[np.ndarray, str, PathLike],
        channel_names: list = None,
        remove_channels: list = None,
        max_quantile: float = 0.98,
        scale: float = 1.0,
) -> anndata.AnnData:
    """
    To read Image_based spatial omics data.
    Slide seqV2, Stereo seq and other image data

    Parameters
    ----------
    image : Union[np.ndarray, str, PathLike]
        multiplexed image (.tiff file); with dimension (z,x,y), where z represents channel number, x and y represent length and width of sample respectively
        you can use np.swapaxes() for dimension. E.g. convert (x,y,z) to (z,x,y):
        img_new=np.swapaxes(np.swapaxes(img,0,2),1,2) # (x,y,z)→(z,y,x)→(z,x,y).
    mask : Union[np.ndarray, str, PathLike]
        image of mask, whose size should be equal to img.
    channel_names : list
        list of channel names (e.g. protein names), whose length should be equal to channel number of img.
    remove_channels : list
        List of the index of channels to be removed.
    max_quantile : float
        highest quantile clipped as highest value.
    scale ; float
        The scale of the coordinates, usually for visual matching with different resolutions of HE or other images,
        is not scaled by default.

    Returns
    -------
        anndata.Anndata object
    """

    import tifffile

    if isinstance(image, np.ndarray):
        img = image.astype(np.uint8)
    else:
        img = tifffile.imread(image)

    if type(mask) is np.ndarray:
        mask = mask.astype(np.uint8)
    else:
        mask = plt.imread(mask, 0)

    mask[mask != 0] = 1
    retval, mask_new, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
    mask = mask_new
    exp_mat = np.zeros((mask.max(), img.shape[0]))

    for g in range(img.shape[0]):
        img_g = img[g, :, :]
        if img_g.sum() != 0:
            img_g[img_g >= np.quantile(img_g[img_g != 0], max_quantile)] = np.quantile(img_g[img_g != 0],
                                                                                       max_quantile)
        for i in range(1, mask.max() + 1):
            l = mask == i
            exp_mat[i - 1, g] = np.mean(img_g[l])

    if remove_channels:
        for g in remove_channels:
            exp_mat[:, g] = 0

    adata = anndata.AnnData(exp_mat)

    if channel_names is None:
        channel_names = [index for index in img.shape[0]]

    if channel_names:
        adata.var_names = channel_names
    adata.obs_names = range(1, len(adata.obs_names) + 1)

    adata.uns['img'] = img
    adata.uns['mask'] = mask
    adata.uns['var_for_analysis'] = [channel_names[i] for i in range(len(channel_names)) if
                                     remove_channels.count(i) == 0]

    adata.obsm['spatial'] = centroids[1:, :]

    adata.var_names_make_unique()

    adata.uns["spatial"] = {}
    adata.uns["spatial"]['scale'] = scale
    adata.uns['SOAPy'] = {}

    return adata


def read_visium2adata(
        path: Union[str, PathLike],
        count_file: str = "filtered_feature_bc_matrix.h5",
        filtered_feature_bc_matrix='filtered_feature_bc_matrix',
) -> anndata.AnnData:
    """
    Read spatial transcriptomics data of 10X Visium.
        If an h5 file exists, the read_visium() method in scanpy is called by default,
        otherwise it is read through the filtered_feature_bc_matrix file.

    filtered_feature_bc_matrix file need include:
        features.tsv.gz;
        barcodes.tsv.gz;
        matrix.mtx.gz.

    Parameters
    ----------
    path :
        Path of Visium root directory folder.
    count_file :
        Which file in the passed directory to use as the count file.
        Typically either filtered_feature_bc_matrix.h5 or raw_feature_bc_matrix.h5.
    filtered_feature_bc_matrix :
        The folder containing the counts information in the root directory.


    Returns
    -------
        anndata.Anndata object
    """
    # sc.read_visium()
    path = Path(path)
    if os.path.exists(path / count_file):
        adata = sc.read_visium(path, count_file=count_file)
    else:
        genes = pd.read_csv(path / filtered_feature_bc_matrix / 'features.tsv.gz', header=None, sep='\t')
        barcodes = pd.read_csv(path / filtered_feature_bc_matrix / 'barcodes.tsv.gz', header=None, sep='\t')
        mtx = sio.mmread(path / filtered_feature_bc_matrix / 'matrix.mtx.gz').T

        genes = pd.DataFrame(index=genes[0].tolist())
        barcodes = pd.DataFrame(index=barcodes[0].tolist())

        crd = pd.read_csv(path / 'spatial/tissue_positions_list.csv', header=None, index_col=0)
        crd = crd.iloc[:, [3, 4]]
        index = np.isin(crd.index.tolist(), barcodes.index.tolist())
        crd = crd.loc[index, :]

        spatial = crd.values
        adata = anndata.AnnData(mtx, obs=barcodes, var=genes, obsm={})

        adata.obsm['spatial'] = spatial

    return adata


def read_dsp2adata(
        information_file: Union[str, PathLike],
        slide_name: Union[str, list] = None,
        xml_file: Optional[dict] = None,
        polygon_key: str = 'Polygon',
        point_attribute: str = 'Points',
) -> anndata.AnnData:
    """
    Read spatial transcriptomics data of NanoString GeoMx DSP.

    The Export3_BiologicalProbeQC.xlsx or Export4_NormalizationQ3.xlsx file in DSP is used to build adata data.
    If the information of each AOI(ROI) sampling point needs to be placed,
    the dictionary is used to pass in the xml file of each slide.

    Parameters
    ----------
    information_file : Union[str, PathLike]
        The path of Export3_BiologicalProbeQC.xlsx or Export4_NormalizationQ3.xlsx file.
    slide_name : Union[str, list], optional
        The name or names list of used slides. By default, all slides are read.
    xml_file : dict, optional
        If sampling points inside each AOI are needed, pass in the xml file using the dictionary.
        e.g.: {'mu_dev_E13_011': './mu_dev_E13_011.ome.xml'}
    polygon_key : str
        Polygon Tag in .xml file
    point_attribute : str
        Point attribute in polygon Tag.

    Returns
    -------
        anndata.Anndata object
    """
    import xml.dom.minidom

    if isinstance(slide_name, str):
        slide_name = [slide_name]

    roi_inf = pd.read_excel(
        information_file,
        sheet_name='SegmentProperties',
        index_col='SegmentDisplayName',
        header=0,
    )
    if slide_name is None:
        obs = roi_inf
    else:
        obs = roi_inf.loc[[True if i in slide_name else False for i in roi_inf['ScanLabel'].tolist()], :]

    roi_express = pd.read_excel(
        information_file,
        sheet_name='TargetCountMatrix',
        index_col=0,
        header=0,
    ).T

    express = roi_express.loc[obs.index, :]
    var = pd.DataFrame(index=express.columns)

    adata = anndata.AnnData(express.values, obs=obs, var=var, uns={}, obsm={})
    adata.obsm['spatial'] = obs.loc[:, ['ROICoordinateX', 'ROICoordinateY']]

    if xml_file is not None:
        slide_list = list(obs['SlideName'].unique())
        points_roi = pd.DataFrame(columns=['slide', 'roi', 'x', 'y'])

        for slide, path_xml in xml_file.items():
            if slide not in slide_list:
                raise ValueError

            dom = xml.dom.minidom.parse(path_xml)
            root = dom.documentElement
            Polygons = root.getElementsByTagName(polygon_key)
            roi_list = obs[obs['SlideName'] == slide]['ROILabel'].astype(int).tolist()

            for index, Polygon in enumerate(Polygons):
                if (index + 1) not in roi_list:
                    continue
                points = Polygon.getAttribute(point_attribute)
                points = points.split()
                for point in points:
                    x, y = point.split(',')
                    x = float(x)
                    y = float(y)
                    points_roi.loc[len(points_roi.index)] = [slide, index + 1, x, y]

        adata.uns['point'] = points_roi

    return adata

# based rpy2
# Python library preparation
from typing import List
from os import PathLike


# function to transform 10X Visium data
def st_Seurat2Anndata(
        seurat_file: PathLike,
        exp_mat_slot: List = ['RNA', 'counts'],
        res_type: str = 'lowres',
) -> anndata.AnnData:
    """
    Transform the Seurat .rds spatial omics data into Anndata.

    Parameters
    ----------
    seurat_file : PathLike
        Path of seurat_obj.rds file.
    exp_mat_slot : list
        list with assay and slot name of expression matrix in seurat object to export to adata.X.
    res_type : str
        resolution type of image in Seurat object (default: 'lowres' in Seurat).
    Returns
    -------
        anndata.Anndata object

    """

    # R console preparation
    from rpy2.robjects import r
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri

    # assert r_home!=None 'R Console is not assigned'

    # Automatically transfer R-data.frame to Python-DataFrame
    pandas2ri.activate()

    # R fucntion preparation
    readRDS = robjects.r['readRDS']
    to_array = robjects.r['as.array']
    to_data_frame = robjects.r['as.data.frame']
    transpose = robjects.r['t']

    # required R package loading

    Seurat = importr('Seurat')
    SeuratObject = importr('SeuratObject')

    # get simple data as single cell
    GetMetaData = r("""
    function(seurat_obj){
    library(Seurat)
    metadata=seurat_obj@meta.data
    metadata$Idents=Idents(seurat_obj)
    return(metadata)
    }
    """)

    GetMetaFeature = r("""
    function(seurat_obj,assay='RNA'){
    library(Seurat)
    library(dplyr)
    gene_feature=GetAssay(seurat_obj,assay=assay)@meta.features
    gene_feature$'genome'=1
    return(gene_feature)
    }
    """)

    # GetAssayData = r("""
    # function(seurat_obj,assay,slot){
    # library(Seurat)
    # exp_mat=GetAssayData(seurat_obj,assay=assay,slot=slot)
    # return(exp_mat %>% as.data.frame())
    # }
    # """)

    # get spatial data specific specific for ST
    GetSpatial = r("""
    function(seurat_obj)
    return(seurat_obj@images)
    """)

    GetCoordinates_Visium = r("""
    function(image_obj){
    library(Seurat)
    coord_mat=image_obj@coordinates
    return(coord_mat[,c('imagerow','imagecol')])
    }
    """)

    GetCoordinates_SlideSeq = r("""
    function(image_obj){
    library(Seurat)
    coord_mat=image_obj@coordinates
    return(coord_mat[,c('x','y')])
    }
    """)

    GetCellCoordinates = r("""
    function(image_obj){
    library(Seurat)
    return(GetTissueCoordinates(image_obj[["centroids"]])[,c('x','y')])
    }
    """)

    GetSegmentationCoordinates = r("""
    function(image_obj){
    library(Seurat)
    return(GetTissueCoordinates(image_obj[["segmentation"]]))
    }
    """)

    GetMoleculeCoordinates = r("""
    function(image_obj,assay,slot){
    library(Seurat)
    library(dplyr)
    x=rownames(GetAssayData(seurat_obj,assay=assay,slot=slot))
    return(GetTissueCoordinates(image_obj[["molecules"]]) %>% filter(molecule %in% x))
    }
    """)

    GetArrayCoordinates = r("""
    function(image_obj){
    library(Seurat)
    coord_mat=image_obj@coordinates[,c('col','row')]
    colnames(coord_mat)=c('array_row','array_col')
    return(coord_mat)
    }
    """)

    GetImage = r("""
    function(image_obj){
    library(Seurat)
    return(image_obj@image)
    }
    """)

    GetScaleFactor = r("""
    function(image_obj){
    library(Seurat)
    return(image_obj@scale.factors)
    }
    """)

    GetImageType = r("""
    function(image_obj){
    library(Seurat)
    return(class(image_obj))
    }
    """)

    import pandas as pd
    import numpy as np

    seurat_obj = readRDS(seurat_file)

    assay = exp_mat_slot[0]
    slot = exp_mat_slot[1]
    count_mat = SeuratObject.GetAssayData(seurat_obj, assay=assay, slot=slot)

    count_mat = to_array(count_mat)
    count_mat = count_mat.transpose()

    # get gene names
    metafeature = robjects.conversion.rpy2py(GetMetaFeature(seurat_obj, assay))

    # metadata
    metadata = robjects.conversion.rpy2py(GetMetaData(seurat_obj))

    # create anndata
    adata = anndata.AnnData(X=count_mat, var=metafeature, obs=metadata)
    coord_mat_ls = []

    images = GetSpatial(seurat_obj)
    ImageType = GetImageType(images[0])[0]

    # 10X Visium
    if ImageType == 'VisiumV1':
        array_coord_mat_ls = []
        spatial_dir = {}
        for i in range(len(images)):
            image_obj = images[i]
            # sample name
            sample_name = images.names[i]
            # coordinate matrix
            coord_mat = GetCoordinates_Visium(image_obj)
            coord_mat = np.array(coord_mat)
            coord_mat = coord_mat.T
            coord_mat_ls.append(coord_mat)
            array_coord_mat = robjects.conversion.rpy2py(GetArrayCoordinates(image_obj))
            array_coord_mat_ls.append(array_coord_mat)
            # image array
            image_array = SeuratObject.GetImage(image_obj, mode='raw')
            # scale factor
            scalefactors = GetScaleFactor(image_obj)
            scalefactors = {scalefactors.names[i]: scalefactors[i][0] for i in range(len(scalefactors))}
            scalefactors = {'fiducial_diameter_fullres': scalefactors['fiducial'],
                            'tissue_hires_scalef': scalefactors['hires'],
                            'tissue_lowres_scalef': scalefactors['lowres']}
            scalefactors['spot_diameter_fullres'] = 0.6196 * scalefactors['fiducial_diameter_fullres']
            spatial_dir[sample_name] = {'images': {res_type: image_array},
                                        'scalefactors': scalefactors}
        # add spatial data
        adata.uns['spatial'] = spatial_dir
        if len(array_coord_mat_ls) == 1:
            array_coord_mat = array_coord_mat_ls[0]
        else:
            array_coord_mat = pd.concat(array_coord_mat_ls, axis=0)
        adata.obs = pd.concat((adata.obs, array_coord_mat), axis=1)

    # Slide-Seq
    if ImageType == 'SlideSeq':
        for i in range(len(images)):
            image_obj = images[i]
            # sample name
            sample_name = images.names[i]
            # coordinate matrix
            coord_mat = GetCoordinates_SlideSeq(image_obj)
            coord_mat = np.array(coord_mat)
            coord_mat = coord_mat.T
            coord_mat_ls.append(coord_mat)

    # STARmap
    # if ImageType=='STARmap':

    # MERSCOPe, Xenium, Nanostring CosMx, CODEX, etc.
    if ImageType == 'FOV':
        for i in range(len(images)):
            image_obj = images[i]
            # sample name
            sample_name = images.names[i]
            # coordinate matrix
            coord_mat = GetCellCoordinates(image_obj)
            coord_mat = np.array(coord_mat)
            coord_mat = coord_mat.T
            coord_mat_ls.append(coord_mat)
            # image_data=GetMoleculeCoordinates(image_obj,assay,slot)
            # segmentation_data=GetSegmentationCoordinates=(image_obj)

    # add coordination matrix
    if len(coord_mat_ls) == 1:
        coord_mat = coord_mat_ls[0]
    else:
        coord_mat = np.concatenate(coord_mat_ls, axis=0)
    coord_mat = coord_mat[:, [1, 0]]
    adata.obsm['spatial'] = coord_mat

    return adata


# function to transform scRNA-Seq data
def sc_Seurat2Anndata(
        seurat_file,
        exp_mat_slot=['RNA', 'counts']
):
    # R console preparation
    from rpy2.robjects import r
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    # assert r_home!=None 'R Console is not assigned'

    # Automatically transfer R-data.frame to Python-DataFrame
    pandas2ri.activate()

    # R fucntion preparation
    readRDS = robjects.r['readRDS']
    to_array = robjects.r['as.array']
    to_data_frame = robjects.r['as.data.frame']
    transpose = robjects.r['t']

    # required R package loading

    Seurat = importr('Seurat')
    SeuratObject = importr('SeuratObject')

    # get simple data as single cell
    GetMetaData = r("""
    function(seurat_obj){
    library(Seurat)
    metadata=seurat_obj@meta.data
    return(metadata)
    }
    """)

    GetMetaFeature = r("""
    function(seurat_obj,assay='RNA'){
    library(Seurat)
    library(dplyr)
    gene_feature=GetAssay(seurat_obj,assay=assay)@meta.features
    gene_feature$'genome'=1
    return(gene_feature)
    }
    """)

    GetAssayData = r("""
    function(seurat_obj,assay,slot){
    library(Seurat)
    exp_mat=GetAssayData(seurat_obj,assay=assay,slot=slot)
    return(exp_mat %>% as.data.frame())
    }
    """)

    seurat_obj = readRDS(seurat_file)

    # raw count matrix
    assay = exp_mat_slot[0]
    slot = exp_mat_slot[1]
    count_mat = SeuratObject.GetAssayData(seurat_obj, assay=assay, slot=slot)
    count_mat = to_array(count_mat)
    count_mat = count_mat.transpose()

    # get gene names
    metafeature = GetMetaFeature(seurat_obj, assay)

    # metadata
    metadata = GetMetaData(seurat_obj)

    # create anndata
    adata = anndata.AnnData(X=count_mat, var=metafeature, obs=metadata)

    return adata