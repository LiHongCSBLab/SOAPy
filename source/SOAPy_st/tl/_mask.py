import copy
import numpy as np
import scanpy as sc
import cv2 as cv
from typing import Optional, Union
from .utils import _assert_variable_in_list
from ..utils import _check_adata_type, _add_info_from_sample, _scale


class Spatialimg(object):
    """
    Description:
    This class's function is that clustering points can be automatically divided into spatial regions
    """

    def __init__(self,
                 adata: sc.AnnData,
                 clusters: Union[list, str, int],
                 KSize: int,
                 cluster_key: Optional[str] = 'cluster_domain',
                 k_blur: Optional[int] = 7,
                 scale: Union[float, str] = 'hires',
                 eliminate_hole: bool = False,
                 remove_small_objects: bool = False,
                 minsize: int = 1000,
                 connectivity: int = 1,
                 ):

        assert cluster_key in adata.obs.columns, "Clustering must be specified before mask generation."
        assert KSize % 2 == 1, "KSize has to be odd number"

        self.adata = adata
        self.adata_obs = copy.deepcopy(adata.obs)
        self.cluster_label = cluster_key
        self.scale = scale

        print(adata.obsm['spatial'].max(axis=0))
        self.heigh = adata.obsm['spatial'].max(axis=0)[1] * self.scale
        self.width = adata.obsm['spatial'].max(axis=0)[0] * self.scale

        crd = self.adata.obsm['spatial'] * float(self.scale)
        self.adata_obs['imagerow'] = crd[:, 1].astype('int64').tolist()
        self.adata_obs['imagecol'] = crd[:, 0].astype('int64').tolist()

        if type(clusters) in [int, str]:
            clusters = [clusters]
        self.clusters = clusters
        self.KSize = (KSize, KSize)
        self.k_blur = k_blur
        self.eliminate_hole = eliminate_hole
        self.remove_small_objects = remove_small_objects
        if remove_small_objects:
            self.minsize = minsize
            self.connectivity = connectivity

    @property
    def Spot_to_Mask(self,
                     ) -> np.ndarray:

        """
        Get mask from spatial domain spot in ST

        Returns
        -------
        numpy.ndarray, mask of the selected domain
        """
        self.img_mask = self.__Preprocessing()
        self.img_mask = self.__Deliate_and_erode()
        if self.eliminate_hole:
            self.img_mask = self.__Eliminate_hole()
        if self.remove_small_objects:
            self.img_mask = self.__Remove_small_objects()

        return self.img_mask

    def __Preprocessing(self):

        """
        Preprocessing function

        :return:
        None
        """
        import math

        for cluster in self.clusters:
            if cluster not in self.adata_obs[self.cluster_label].unique():
                raise ValueError('Input clusters must belong to existing clusters. Checking your input value')

        self.adata_obs.loc[:, 'selected'] = False
        for barcode in self.adata_obs.index:
            if self.adata_obs.loc[barcode, self.cluster_label] in self.clusters:
                self.adata_obs.loc[barcode, 'selected'] = True

        self.adata_obs = self.adata_obs[self.adata_obs['selected']]
        self.mask_data = np.zeros((int(self.heigh * 1.1), int(self.width * 1.1)))

        for index, row in self.adata_obs.iterrows():
            self.mask_data[
                math.floor(row['imagerow']), math.floor(
                    row['imagecol'])] = 255
        kernel = np.ones(self.KSize, dtype=np.uint8)
        imgDilate = cv.dilate(self.mask_data, kernel=kernel).astype('uint8')
        self.mask_data = imgDilate
        return imgDilate

    def __Deliate_and_erode(self,
                            ) -> np.ndarray:

        """
        Function of dilation and erosion

        Returns
        -------
        numpy.ndarray, mask
        """
        kernel = np.ones(self.KSize, dtype=np.uint8)
        imgErode = cv.erode(self.mask_data, kernel=kernel)
        imgErode = imgErode.astype('uint8')
        # imgErode = imgDilate.astype('uint8')

        return imgErode

    def __Eliminate_hole(self,
                         ) -> np.ndarray:

        """
        Fynction that eliminate holes

        Returns
        -------
        numpy.ndarray, mask
        """
        mask = self.img_mask.astype('uint8')
        imgBin = cv.bitwise_not(mask)
        marker = np.zeros_like(imgBin, dtype=np.uint8)
        marker[0, :] = 255
        marker[-1, :] = 255
        marker[:, 0] = 255
        marker[:, -1] = 255

        element = cv.getStructuringElement(shape=cv.MORPH_CROSS, ksize=(5, 5))
        while True:
            marker_pre = marker
            dilation = cv.dilate(marker, kernel=element)
            dilation = dilation.astype('uint8')
            marker = np.min((dilation, imgBin), axis=0)
            if (marker_pre == marker).all():
                break
        imgRebuild = cv.bitwise_not(marker)
        print(imgRebuild)

        return imgRebuild

    def __Remove_small_objects(self,
                               ) -> np.ndarray:

        """
        Function that remove small connected components

        Returns
        -------
        numpy.ndarray, mask
        """

        from skimage import morphology
        imgBlur = cv.blur(self.img_mask, (self.k_blur, self.k_blur))
        ret, imgBlur = cv.threshold(imgBlur, 100, 255, cv.THRESH_BINARY)
        imgBlur = np.array(imgBlur, dtype=bool)
        imgRemove = morphology.remove_small_objects(imgBlur, min_size=self.connectivity, connectivity=self.connectivity)
        imgRemove = imgRemove.astype('uint8') * 255
        return imgRemove


def get_mask_from_domain(
        adata: sc.AnnData,
        clusters: Union[list, str, int],
        KSize: int,
        cluster_key: Optional[str] = 'domain',
        k_blur: Optional[int] = 7,
        scale: Union[str, float] = 'hires',
        eliminate_hole: bool = False,
        remove_small_objects: bool = False,
        minsize: int = 1000,
        connectivity: int = 1,
        inplace: bool = True,
) -> np.ndarray:
    """
    A mask image is generated according to the selected category, and the mask can be processed using morphological methods
    such as dilation and erosion, removal of holes, and removal of small connected components

    The size of mask is based on adata.obsm['spatial']

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    clusters
        Making mask in this clusters
    KSize
        The size of convolution kernel in function dilate
    cluster_key
        Label to which the param 'clusters' belongs
    k_blur
        The size of convolution kernel in function cv2.blur()
    scale
        The spatial scale used by the operation
    eliminate_hole
        If True, use cv.bitwise_not() to eliminate hole
    remove_small_objects
         If True, use morphology.remove_small_objects() to remove small domains
    minsize
        The shortest perimeter of removed domains. Used during
        labelling if `RemoveSmallObjects` is True.
    connectivity
        The connectivity defining the neighborhood of a pixel. Used during
        labelling if `RemoveSmallObjects` is True.
        1 is  four neighborhood, 2 is eight neighborhood.
    inplace : bool, optional
        If True, Modify directly in the original adata.

    Returns
    -------
    numpy.ndarray, mask of the selected domain
    """
    adata = _check_adata_type(adata, 'spatial', inplace)

    if type(scale) != float:
        scale = _scale(adata, None, scale)

    New_Mask = Spatialimg(adata=adata,
                          clusters=clusters,
                          KSize=KSize,
                          cluster_key=cluster_key,
                          k_blur=k_blur,
                          scale=scale,
                          eliminate_hole=eliminate_hole,
                          remove_small_objects=remove_small_objects,
                          minsize=minsize,
                          connectivity=connectivity
                          )

    Mask = New_Mask.Spot_to_Mask
    _add_info_from_sample(adata, sample_id=None, keys='mask', add=Mask)

    return Mask
