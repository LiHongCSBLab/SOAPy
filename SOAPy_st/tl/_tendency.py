import copy
from typing import Optional, Union, Literal
import logging as logg
import anndata
from tqdm import tqdm
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import cv2 as cv
import statsmodels.api as sm
from skmisc.loess import loess
from ..utils import _scale, _add_info_from_sample, _get_info_from_sample, _check_adata_type
import warnings

__all__ = ["wilcoxon_test", "spearman_correlation", "ANOVA", "spatial_tendency", "gene_cluster", "adata_from_mask"]


def polynomial_regression(
        dis_and_express: pd.DataFrame,
        gene_name: str,
        frac: int,
        sd: Optional[int] = None,
        drop_zero: bool = False,
):
    """
    polynomial regression function
    """
    mat_only_gene_express = dis_and_express.loc[:, ['distance', gene_name]].values

    if sd != None:
        y = mat_only_gene_express[:, 1]
        mu = np.mean(y)
        std = np.std(y)
        mat_only_gene_express[(mat_only_gene_express[:, 1] < mu - sd * std), 1] = mu - sd * std
        mat_only_gene_express[(mat_only_gene_express[:, 1] > mu + sd * std), 1] = mu + sd * std

    if drop_zero:
        mat_only_gene_express = mat_only_gene_express[~(mat_only_gene_express[:, 1] == 0)]

    x = mat_only_gene_express[:, 0]
    y = mat_only_gene_express[:, 1]
    ind = x.argsort()
    x = x[ind]
    y = y[ind]
    X = x

    if frac > 1:
        for index in range(frac - 1):
            X = np.column_stack((X, x ** (index + 2)))
    X = sm.add_constant(X)

    model = sm.OLS(y.astype(float), X)
    res = model.fit()
    y_fitted = res.fittedvalues

    y_min = y_fitted.min()
    y_max = y_fitted.max()
    ran = y_max - y_min

    aic = res.aic
    bic = res.bic

    corr = np.corrcoef(x=x, y=y)[0, 1]

    return res.f_pvalue, x, y_fitted, res.params, ran, corr, len(y), aic, bic


def loess_(
        dis_and_express: pd.DataFrame,
        gene_name: str,
        frac_0_1: float,
        sd: Optional[int] = None,
        drop_zero: bool = False,
):
    """
    loess function
    """

    mat_only_gene_express = dis_and_express.loc[:, ['distance', gene_name]].values

    if sd is not None:
        y = mat_only_gene_express[:, 1]
        mu = np.mean(y)
        std = np.std(y)
        mat_only_gene_express[(mat_only_gene_express[:, 1] < mu - sd * std), 1] = mu - sd * std
        mat_only_gene_express[(mat_only_gene_express[:, 1] > mu + sd * std), 1] = mu + sd * std

    if drop_zero:
        mat_only_gene_express = mat_only_gene_express[~(mat_only_gene_express[:, 1] == 0)]

    x = mat_only_gene_express[:, 0]
    y = mat_only_gene_express[:, 1]

    # x = np.array(x)
    # y = np.array(y)
    ind = x.argsort()
    y = y[ind]
    x = x[ind]

    l = loess(x, y, span=frac_0_1)
    l.fit()
    pred = l.predict(x, stderror=True)
    yest = pred.values

    yBar = np.mean(y)
    SSE = 0
    SST = 0
    for i in range(0, len(x)):
        SSE += (yest[i] - y[i]) ** 2
        SST += (y[i] - yBar) ** 2

    r_sq = round(1 - SSE / SST, 3)
    corr = np.corrcoef(x=x, y=y)[0, 1]

    y_min = yest.min()
    y_max = yest.max()
    ran = y_max - y_min

    return x, yest, r_sq, corr, ran, len(y)


class SpatialTendency(object):

    def __init__(self,
                 adata: sc.AnnData,
                 gene_name: Union[str, list],
                 radius: Optional[float] = None,
                 scale: Union[float, str] = None,
                 clusters: Union[str, int, list] = 'all',
                 cluster_key: Optional[str] = None,
                 spatial_in_obsm: str = 'spatial',
                 ) -> None:
        """
        initialization
        """

        if clusters != 'all':
            if type(clusters) != list:
                clusters = [clusters]
            self.tendency_selected = [(i in clusters) for i in adata.obs[cluster_key]]

        if type(adata.X) is not np.ndarray:
            df = pd.DataFrame(adata.X.toarray())
        else:
            df = pd.DataFrame(adata.X)

        df.index = adata.obs.index
        df.columns = adata.var.index
        df_pixel = adata.obsm[spatial_in_obsm]
        # df_pixel = df_pixel.iloc[:, [0, 1]]

        if type(scale) != float:
            scale = _scale(adata, None, scale)
        df_pixel = pd.DataFrame(df_pixel, index=adata.obs_names) * scale

        adata.obs['imagerow'] = df_pixel[1].tolist()
        adata.obs['imagecol'] = df_pixel[0].tolist()

        self.adata = adata
        self.scale = str(scale)

        self.express = df
        if clusters != 'all':
            self.express = self.express.loc[self.tendency_selected, :]
        self.coordinate = adata.obs.loc[:, ['imagerow', 'imagecol']]
        if clusters != 'all':
            self.coordinate = self.coordinate.loc[self.tendency_selected, :]
        if type(gene_name) != list:
            gene_name = [gene_name]
        self.gene_name = gene_name
        if radius is None:
            radius = np.inf
        self.radius = radius
        self.inSpots = []

        if radius <= 0:
            raise RuntimeError("Radius should greater than 0")
        self.__structureCheck()

    def __structureCheck(self, ) -> None:
        """
        checking for mismatched data

        """

        shape_express = self.express.shape
        shape_coordinate = self.coordinate.shape

        if len(shape_express) > 2:
            raise RuntimeError("Express should be 2-dimensionality")
        if len(shape_coordinate) > 2:
            raise RuntimeError("Coordinate should be 2-dimensionality")

    def __distance_spot_and_contour(self,
                                    contour: np.ndarray,
                                    location: Optional[str],
                                    ) -> pd.DataFrame:
        """
        The distance from each spot to the mask boundary is calculated, positive outside the mask and negative
        inside the mask
        """

        spot_x = []
        spot_y = []
        dists = []
        exps = []

        for gene_name in self.gene_name:
            exps.append([])
        threshold = self.__find_threshold(contour, self.radius)
        crds = self.coordinate

        for index in crds.index:
            point = (crds.loc[index, 'imagecol'], crds.loc[index, 'imagerow'])
            if point in self.inSpots:
                continue
            if not self.__fliter(point, threshold):
                continue

            point = (float(point[0]), float(point[1]))
            dist = cv.pointPolygonTest(contour, point, True)
            dist = -dist

            if location == 'out':
                if dist < 0 or abs(dist) > self.radius:  ##########
                    if dist < 0:  ###############
                        self.inSpots.append(point)  ##################
                    continue
            if location == 'in':
                if dist > 0 or abs(dist) > self.radius:  ##########
                    continue
            if location == 'all':
                if abs(dist) > self.radius:
                    continue
            for index_gene, gene_name in enumerate(self.gene_name):
                exps[index_gene].append(self.express.at[index, gene_name])
            spot_x.append(point[0])
            spot_y.append(point[1])
            dists.append(dist)

        contour_dis_and_express = {'point_x': spot_x, 'point_y': spot_y, 'distance': dists}

        for index, gene_name in enumerate(self.gene_name):
            contour_dis_and_express[gene_name] = exps[index]

        contour_dis_and_express = pd.DataFrame(contour_dis_and_express)

        return contour_dis_and_express

    def __find_threshold(self,
                         contour: np.ndarray,
                         radius: float,
                         ) -> list:

        left_point = tuple(contour[contour[:, :, 0].argmin()][0])
        right_point = tuple(contour[contour[:, :, 0].argmax()][0])
        top_point = tuple(contour[contour[:, :, 1].argmin()][0])
        bottom_point = tuple(contour[contour[:, :, 1].argmax()][0])

        return [left_point[0] - radius, right_point[0] + radius, top_point[1] - radius, bottom_point[1] + radius]

    def __fliter(self,
                 point: tuple,
                 threshold: list,
                 ) -> bool:
        """
        Pre-filter spots, keeping points that are likely to be within distance
        """
        if point[0] < threshold[0] or point[0] > threshold[1]:
            return False
        if point[1] < threshold[2] or point[1] > threshold[3]:
            return False

        return True

    def get_new_adata(self,
                      mask: np.ndarray,
                      location: Optional[str],
                      radius,
                      ):
        contours, hierarchy = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        self.adata.obs['in_mask'] = None

        for contour in contours:
            threshold = self.__find_threshold(contour, self.radius)

            for index in self.adata.obs_names.tolist():
                point = (self.adata.obs.loc[index, 'imagecol'], self.adata.obs.loc[index, 'imagerow'])
                if not self.__fliter(point, threshold):
                    continue

                point = (float(point[0]), float(point[1]))
                marker = cv.pointPolygonTest(contour, point, False)
                marker = -marker
                if self.adata.obs.loc[index, 'in_mask'] is None:
                    self.adata.obs.loc[index, 'in_mask'] = marker
                else:
                    if marker < 0:
                        self.adata.obs.loc[index, 'in_mask'] = marker
                    else:
                        self.adata.obs.loc[index, 'in_mask'] = np.min((self.adata.obs.loc[index, 'in_mask'], marker))

        if radius is None:
            radius = np.inf
        if location == 'in':
            left = -radius
            right = 0
        elif location == 'out':
            left = 0
            right = radius
        else:
            left = -radius
            right = radius

        bdata = self.adata[np.isin(left <= self.adata.obs['in_mask'], True, False), :].copy()
        bdata = bdata[np.isin(bdata.obs['in_mask'] <= right, True, False), :].copy()

        return bdata

    def get_dist_and_express(self,
                             mask: np.ndarray,
                             location: Optional[str] = 'all',
                             ) -> pd.DataFrame:

        contours, hierarchy = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        index = 0
        dis_and_express = []
        x_all = []
        y_all = []
        dis_all = []
        express_all = []

        for index_gene, gene_name in enumerate(self.gene_name):
            express_all.append([])

        # Get the data for each spot
        for contour in contours:
            contour_dis_and_express = self.__distance_spot_and_contour(contour, location)
            dis_and_express.append([contour, dis_and_express])
            x_all = x_all + contour_dis_and_express['point_x'].tolist()
            y_all = y_all + contour_dis_and_express['point_y'].tolist()
            dis_all = dis_all + contour_dis_and_express['distance'].tolist()
            for index_gene, gene_name in enumerate(self.gene_name):
                express_all[index_gene] = express_all[index_gene] + contour_dis_and_express[gene_name].tolist()
            index += 1

        # Integrate data from all spots
        dis_and_express_all = {'x': x_all, 'y': y_all, 'distance': dis_all}
        for index_gene, gene_name in enumerate(self.gene_name):
            dis_and_express_all[gene_name] = express_all[index_gene]

        dis_and_express_all = pd.DataFrame(dis_and_express_all)
        dis_and_express_all = self.__remove_duplicates(dis_and_express_all)

        for index_point in dis_and_express_all.index:
            if (dis_and_express_all.loc[index_point, 'x'], dis_and_express_all.loc[index_point, 'y']) in self.inSpots:
                dis_and_express_all.drop(labels=index_point, inplace=True)

        _add_info_from_sample(self.adata, sample_id=None, keys='distance_and_expression', add=dis_and_express_all)
        _add_info_from_sample(self.adata, sample_id=None, keys='contours', add=contours)

        return dis_and_express_all

    def loess(self,
              mask: np.ndarray,
              frac_0_1: float = 0.6666,
              sd: Optional[int] = None,
              location: Optional[str] = 'all',
              drop_zero: bool = False,
              ) -> anndata.AnnData:
        """
        loess regression of expression and distance
        """

        warnings.filterwarnings('ignore')
        dis_and_express = self.get_dist_and_express(mask, location)

        dic_crd = {}
        list_name = []
        list_r_sq_lowess = []
        list_corr_gene_distance = []
        list_range_lowess = []
        list_num_of_spots = []

        mark = 0
        for gene_name in tqdm(self.gene_name):
            xest, yest, r_sq, corr, ran, spot_num = loess_(dis_and_express,
                                                           gene_name,
                                                           frac_0_1,
                                                           sd=sd,
                                                           drop_zero=drop_zero
                                                           )
            if mark == 0:
                dic_crd['Xest'] = xest
                mark = 1
            dic_crd[gene_name] = yest
            list_name.append(gene_name)
            list_r_sq_lowess.append(r_sq)
            list_corr_gene_distance.append(corr)
            list_range_lowess.append(ran)
            list_num_of_spots.append(spot_num)

        dict_gene = {
            'R_square': list_r_sq_lowess,
            'correlation': list_corr_gene_distance,
            'range': list_range_lowess,
            'Spots number': list_num_of_spots
        }

        df_param_loess = pd.DataFrame(dict_gene, index=list_name)

        add_data = {'dic_crd_loess': dic_crd, 'df_param_loess': df_param_loess, 'loess_frac': frac_0_1}
        _add_info_from_sample(self.adata, sample_id=None, keys='loess', add=add_data)

        return self.adata

    def polynomialRegression(self,
                             mask: np.ndarray,
                             frac: int = 4,
                             sd: Optional[int] = None,
                             location: Optional[str] = 'out',
                             drop_zero: bool = False,
                             ) -> anndata.AnnData:
        """
        Polynomial regression of expression and distance
        """

        warnings.filterwarnings('ignore')
        dis_and_express = self.get_dist_and_express(mask, location)

        dic_crd = {}
        list_name = []
        list_p_value_Poly = []
        list_param_Poly = []
        list_range_Poly = []
        list_corr_gene_distance = []
        list_num_of_spots = []
        list_aic = []
        list_bic = []

        mark = 0
        for gene_name in tqdm(self.gene_name):
            f_pvalue, xest, yest, param, ran, corr, spot_num, aic, bic = polynomial_regression(
                dis_and_express=dis_and_express,
                gene_name=gene_name,
                frac=frac,
                sd=sd,
                drop_zero=drop_zero
                )
            if mark == 0:
                dic_crd['Xest'] = xest
                mark = 1
            dic_crd[gene_name] = yest
            list_name.append(gene_name)
            list_p_value_Poly.append(f_pvalue)
            list_param_Poly.append(param)
            list_range_Poly.append(ran)
            list_corr_gene_distance.append(corr)
            list_num_of_spots.append(spot_num)
            list_aic.append(aic)
            list_bic.append(bic)

        dict_PR = {
            'p_value': list_p_value_Poly,
            'param': list_param_Poly,
            'range': list_range_Poly,
            'correlation': list_corr_gene_distance,
            'Spots number': list_num_of_spots,
            'AIC': list_aic,
            'BIC': list_bic
        }

        df_param_poly = pd.DataFrame(dict_PR, index=list_name)

        add_data = {'dic_crd_poly': dic_crd, 'df_param_poly': df_param_poly, 'poly_frac': frac}
        _add_info_from_sample(self.adata, sample_id=None, keys='poly', add=add_data)

        return self.adata

    def __remove_duplicates(self,
                            dis_and_express: pd.DataFrame,
                            ) -> pd.DataFrame:
        """
        removing duplicate spots
        """
        dis_and_express['distance_abs'] = abs(dis_and_express['distance'])
        dis_and_express = dis_and_express.sort_values(by='distance_abs')
        dis_and_express = dis_and_express.drop_duplicates(subset=['x', 'y'], keep='first')
        for index in dis_and_express.index:
            if (dis_and_express.at[index, 'x'], dis_and_express.at[index, 'y']) in self.inSpots:
                dis_and_express.drop(index=index)
        dis_and_express.reset_index(drop=True)
        return dis_and_express

    def MannWhitney(self,
                    mask: np.ndarray,
                    location: str = 'out',
                    alternative: str = 'two-sided',
                    ran: Optional[float] = None,
                    cut: Optional[float] = 0,
                    alpha: float = 0.05,
                    drop_zero: bool = False
                    ) -> pd.DataFrame:

        from scipy import stats
        from statsmodels.stats.multitest import fdrcorrection

        self.__structureCheck()
        df_dist_express = self.get_dist_and_express(mask, location=location)
        distance = df_dist_express['distance'].tolist()
        near = [True if i > cut else False for i in distance]
        far = [True if i <= cut else False for i in distance]

        df_dist_express.drop(labels=['x', 'y', 'distance', 'distance_abs'], axis=1)
        df_Wilcoxon_Mann_Whitney = pd.DataFrame(columns=['gene', 'stat', 'P value',
                                                         'effective spot near', 'effective spot far'])

        for gene in self.gene_name:
            list_gene_exp = df_dist_express[gene].values
            if ran is not None:
                if max(list_gene_exp) - min(list_gene_exp) < ran:
                    continue
            list_exp_near = list_gene_exp[np.array(near)]
            list_exp_far = list_gene_exp[np.array(far)]
            if drop_zero:
                list_exp_near = list(filter(lambda x: x != 0, list_exp_near))
                list_exp_far = list(filter(lambda x: x != 0, list_exp_far))
                effective_spot_near = len(list_exp_near)
                effective_spot_far = len(list_exp_far)
                if len(list_exp_near) == 0:
                    list_exp_near.append(0)
                if len(list_exp_far) == 0:
                    list_exp_far.append(0)
            else:
                effective_spot_near = len(list_exp_near)
                effective_spot_far = len(list_exp_far)
            stat, pvalue = stats.mannwhitneyu(list_exp_near, list_exp_far, alternative=alternative)
            df_Wilcoxon_Mann_Whitney.loc[len(df_Wilcoxon_Mann_Whitney.index)] = [gene, stat, pvalue,
                                                                                 effective_spot_near,
                                                                                 effective_spot_far]
        rej, p_adjust = fdrcorrection(df_Wilcoxon_Mann_Whitney['P value'], alpha=alpha)
        df_Wilcoxon_Mann_Whitney['P fdr'] = p_adjust
        df_Wilcoxon_Mann_Whitney['P rej'] = rej

        return df_Wilcoxon_Mann_Whitney

    def spearman(self,
                 mask: np.ndarray,
                 num: Optional[float] = None,
                 ran: Optional[float] = None,
                 drop_zero: bool = False
                 ) -> pd.DataFrame:

        from scipy import stats
        from statsmodels.stats.multitest import fdrcorrection

        gap = self.radius / num

        self.__structureCheck()
        df_dist_express = self.get_dist_and_express(mask)

        distance = df_dist_express['distance'].tolist()

        df_dist_express.drop(labels=['x', 'y', 'distance', 'distance_abs'], axis=1)
        df_Spearman = pd.DataFrame(columns=['gene', 'coef', 'P value'])

        dis_level = gap
        level = 1
        for index in range(len(distance)):
            if distance[index] < dis_level:
                distance[index] = level
            else:
                while (distance[index] > dis_level):
                    dis_level += gap
                    level += 1
                distance[index] = level

        for gene in self.gene_name:
            list_gene_exp = df_dist_express[gene].tolist()
            if ran is not None:
                if max(list_gene_exp) - min(list_gene_exp) < ran:
                    continue
            distance_gene = copy.deepcopy(distance)
            if drop_zero:
                length = len(distance)
                drop = 0
                for index in range(length):
                    if list_gene_exp[index - drop] == 0:
                        list_gene_exp.pop(index - drop)
                        distance_gene.pop(index - drop)
                        drop += 1

            coef, pvalue = stats.spearmanr(distance_gene, list_gene_exp)
            df_Spearman.loc[len(df_Spearman.index)] = [gene, coef, pvalue]
        rej, p_adjust = fdrcorrection(df_Spearman['P value'], alpha=0.05)
        df_Spearman['P fdr'] = p_adjust
        df_Spearman['P rej'] = rej
        _add_info_from_sample(self.adata, sample_id=None, keys='spearman', add=df_Spearman)

        return df_Spearman


#####################################################

class GeneCluster(object):
    def __init__(self,
                 dic_crd: dict,
                 method: str,
                 params: pd.DataFrame,
                 norm: bool = True,
                 pvalue: Optional[float] = None,
                 fdr: bool = False,
                 range_min: Optional[float] = None,
                 correlation: Optional[float] = None,
                 ):
        """
        initialization
        """
        self.dic_crd = dic_crd
        self.method = method
        self.df_params = params
        self.norm = norm
        self.pvalue = pvalue
        self.fdr = fdr
        self.range_min = range_min
        self.correlation = correlation
        self.gene_express = pd.DataFrame(dic_crd)
        self.gene_express = self.gene_express.drop_duplicates(subset=['Xest'])
        num_orginal = self.df_params.shape[0]
        self.__select_gene()
        num_filter = self.df_params.shape[0]
        logg.info(f'The original {num_orginal} gene curves, after filtering the remaining {num_filter}')

    def clusterTheGeneTendency(self, k: int, sdbw: bool = True, num=50):
        """
        Determining Regression types
        """

        if self.method == 'lowess':
            self.__cluster_loess(k, sdbw)
        elif self.method == 'poly':
            self.__cluster_Poly(num, k, sdbw)

    def __select_gene(self):
        """
        Filter genes based on user-specified metrics
        """
        if self.pvalue is not None:
            from statsmodels.stats.multitest import fdrcorrection
            if self.fdr:
                rej, p_adjust = fdrcorrection(self.df_params['p_value'], alpha=self.pvalue)
                self.df_params['p_adjust'] = p_adjust
                self.df_params['p_rej'] = rej
                self.df_params = self.df_params[self.df_params['p_rej'] == True]
            else:
                self.df_params = self.df_params[self.df_params['p_value'] < self.pvalue]

        if self.range_min is not None:
            self.df_params = self.df_params[abs(self.df_params['range']) > self.range_min]

        if self.correlation is not None:
            self.df_params = self.df_params[abs(self.df_params['correlation']) > self.correlation]
        assert self.df_params.shape[0] >= 2, 'The number of cluster must >= 2'

    def __cluster_Poly(self, num, k, sdbw):
        """
        Cluster polynomial regression curves
        """

        from .utils import _best_k
        list_gene = []
        x = self.gene_express['Xest']

        for gene_name in self.df_params.index:
            y = self.gene_express[gene_name]
            params = self.df_params.loc[gene_name, 'param']
            xgrid = np.linspace(x.min(), x.max(), num)
            ygrid = []
            for x0 in xgrid:
                ygrid.append(sum([param * x0 ** i for i, param in enumerate(params)]))
            if self.norm:
                ygrid = (np.array(ygrid) - y.min()) / (y.max() - y.min())
            list_gene.append(ygrid)

        array_gene = np.array(list_gene)

        res, km = _best_k(array_gene, k, sdbw)

        self.df_params['cluster'] = res
        self.k = km

    def __cluster_loess(self, k, sdbw):
        """
        Cluster loess regression curves
        """

        from .utils import _best_k
        list_gene = []

        for gene_name in self.df_params.index:
            y = self.gene_express[gene_name]
            if self.norm:
                ygrid = (np.array(y) - y.min()) / (y.max() - y.min())
            else:
                ygrid = y
            list_gene.append(ygrid)

        array_gene = np.array(list_gene)

        res, km = _best_k(array_gene, k, sdbw)
        self.df_params['cluster'] = res
        self.k = km


def adata_from_mask(
        adata: anndata.AnnData,
        mask: np.ndarray,
        location: Literal['all', 'in', 'out'] = 'in',
        spatial_in_obsm: str = 'spatial',
        radius: float = None,
) -> anndata.AnnData:
    """
    Generate a new adata object through the selected mask region.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    mask : numpy.ndarray
        A binarized image data of ROI.
    location : str, optional
        'in': The selected spots inside the contours.
        'out': The selected spots outside the contours.
        'all': both of 'in' and 'out'.
    spatial_in_obsm : str, optional
        The key of spatial coordinates in adata.obsm
    radius : float
        The range of the point whose included in the calculation.

    Returns
    -------
    anndata.Anndata object
    """
    adata = _check_adata_type(adata, spatial_in_obsm, inplace=True)

    new_tendency = SpatialTendency(
        adata=adata,
        radius=None,
        gene_name='None',
        scale=1.0,
        spatial_in_obsm=spatial_in_obsm,
    )

    adata = new_tendency.get_new_adata(
        mask=mask,
        location=location,
        radius=radius,
    )

    return adata


def wilcoxon_test(
        adata: anndata.AnnData,
        mask: np.ndarray,
        radius: float,
        gene_name: Union[str, list, None] = None,
        clusters: Union[str, int, list] = 'all',
        cluster_label: Optional[str] = None,
        location: Literal['all', 'in', 'out'] = 'all',
        scale: Union[float, str] = 'hires',
        spatial_in_obsm: str = 'spatial',
        alternative: str = 'two-sided',
        ran: Optional[float] = None,
        cut: Optional[float] = 0,
        alpha: float = 0.05,
        drop_zero: bool = False,
        inplace: bool = True,
) -> pd.DataFrame:
    """
    Wilcoxon tests were performed using the distance from each spot(cell) to the mask boundary and the
    expression of a gene(protein) at each spot.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    mask : numpy.ndarray
        A binarized image data of ROI.
    radius : float
        Maximum distance considered, negative distance by absolute value.
    gene_name : Union[list, str, None], optional
        The gene names for the regression model need to be calculated.
    clusters : Union[str, int, list], optional
        The cluster of the spot being counted, the default is all clusters.
    cluster_label : str, optional
        The label of cluster in adata.obs.
    location : Literal['all', 'in', 'out']
        'in': The selected spots inside the contours.
        'out': The selected spots outside the contours.
        'all': both of 'in' and 'out'.
    scale : Union[str, float], optional
        Scale used in subsequent analyses. If it's Visium data it can also be HE image labels (hires or lower).
        Most of the time you don't need to change this
    spatial_in_obsm : str, optional
        The key of spatial coordinates in adata.obsm
    alternative: {‘two-sided’, ‘less’, ‘greater’}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        For more details, see scipy.stats.mannwhitneyu()
    ran : float, optional
        Minimum range, genes are ignored when their range is smaller than this value.
    cut : float, optional
        Distance threshold for grouping. The default value is 0, which represents the two groups inside and outside the
        contour.
    alpha : float, optional
        Family-wise error rate of P-fdr. Defaults to 0.05.
    drop_zero : bool, optional
        Whether to remove all spots with 0 expression.
    inplace : bool, optional
        Whether to change the original adata.

    Returns
    -------
    pandas.DataFrame
        The wilcoxon-test result is stored, including the p-value('P value'), the fdr p-value('P fdr'),
        and the label of whether the p-value is significant or not('P rej').
    """
    adata = _check_adata_type(adata, spatial_in_obsm, inplace)

    if gene_name is None:
        gene_name = adata.var_names.tolist()

    new_tendency = SpatialTendency(adata=adata,
                                   gene_name=gene_name,
                                   radius=radius,
                                   scale=scale,
                                   clusters=clusters,
                                   cluster_key=cluster_label,
                                   spatial_in_obsm=spatial_in_obsm,
                                   )

    df_wilcoxon = new_tendency.MannWhitney(mask=mask,
                                           location=location,
                                           alternative=alternative,
                                           ran=ran,
                                           cut=cut,
                                           alpha=alpha,
                                           drop_zero=drop_zero,
                                           )

    return df_wilcoxon


def spearman_correlation(
        adata: anndata.AnnData,
        mask: np.ndarray,
        radius: float,
        gene_name: Union[str, list] = 'all',
        clusters: Union[str, int, list] = 'all',
        cluster_key: Optional[str] = None,
        scale: Union[float, str] = 'hires',
        spatial_in_obsm: str = 'spatial',
        num: Optional[float] = None,
        ran: Optional[float] = None,
        drop_zero: bool = False,
        inplace: bool = True,
) -> pd.DataFrame:
    """
    Spearman test was performed on the expression levels of genes in each class after the samples were classified
    equally by distance

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    mask : numpy.ndarray
        A binarized image data of ROI
    radius : float
        Maximum distance considered, negative distance by absolute value
    gene_name : Union[list, str, None], optional
        The gene names for the regression model need to be calculated.
    clusters : Union[str, int, list], optional
        The cluster of the spot being counted, the default is all clusters.
    cluster_label : str, optional
        The label of cluster in adata.obs.
    scale : Union[str, float], optional
        Scale used in subsequent analyses. If it's Visium data it can also be HE image labels (hires or lower).
        Most of the time you don't need to change this
    spatial_in_obsm : str, optional
        The key of spatial coordinates in adata.obsm
    num : int, optional
        The number of gap
    ran : float, optional
        Minimum range, genes are ignored when their range is smaller than this value.
    drop_zero : bool, optional
        Whether to remove all spots with 0 expression.
    inplace : bool, optional
        Whether to change the original adata.

    Returns
    -------
    pandas.DataFrame
        The spearman result is stored, including the p-value('P value'), the fdr p-value('P fdr'),
        and the label of whether the p-value is significant or not('P rej').
    """
    adata = _check_adata_type(adata, spatial_in_obsm, inplace)

    if gene_name == 'all':
        gene_name = adata.var_names.tolist()

    New_tendency = SpatialTendency(adata=adata,
                                   gene_name=gene_name,
                                   radius=radius,
                                   scale=scale,
                                   clusters=clusters,
                                   cluster_key=cluster_key,
                                   spatial_in_obsm=spatial_in_obsm,
                                   )

    df_spearman = New_tendency.spearman(mask=mask,
                                        num=num,
                                        ran=ran,
                                        drop_zero=drop_zero,
                                        )

    return df_spearman


def ANOVA(
        adata: ad.AnnData,
        score_label: str,
        cluster_label: str = 'cell_type',
):
    """
    Anova of genes or other metrics

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    score_label : str
        Selected genes or other indicators in obs.columns.
    cluster_label : str, optional
        The label of cluster in adata.obs.

    Returns
    -------
    The p-value of ANOVA
    """
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    cluster = adata.obs[cluster_label]
    if score_label in adata.obs.columns:
        score = np.array(adata.obs[score_label])
    elif score_label in adata.var_names:
        index = adata.var_names.tolist().index(score_label)
        score = adata.X.toarray()[:, index]
    else:
        raise ValueError

    df = pd.DataFrame({'cluster': cluster, 'score': score})

    fml = 'score' + '~C(' + 'cluster' + ')'

    model = ols(fml, data=df).fit()
    anova_table_1 = anova_lm(model, typ=2).reset_index()
    p1 = anova_table_1.loc[0, 'PR(>F)']
    _add_info_from_sample(adata, sample_id=None, keys='ANOVA', add=p1)

    return p1


def spatial_tendency(
        adata: ad.AnnData,
        mask: np.ndarray,
        radius: int,
        method: Literal['poly', 'loess'] = 'poly',
        gene_name: Union[str, list] = 'all',
        clusters: Union[str, int, list] = 'all',
        cluster_key: Optional[str] = None,
        location: Literal['all', 'in', 'out'] = 'all',
        scale: Union[float, str] = 'hires',
        spatial_in_obsm: str = 'spatial',
        frac: Union[int, float] = None,
        sd: Optional[int] = None,
        drop_zero: bool = False,
        inplace: bool = True,
) -> anndata.AnnData:
    """
    We used two regression methods, Loess regression, and Polynomial regression, to study the variation of the
    expression with the min distance from its location to the pixel of boundary.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    mask : numpy.ndarray
        Binarized image data of ROI.
    radius : float
        The range of the point whose included in the calculation.
    method : Literal['poly', 'loess'], optional
        Polynomial regression(poly) or Loess regression(loess).
    gene_name : Union[list, str, None], optional
        The gene names for the regression model need to be calculated.
    clusters : Union[str, int, list], optional
        The cluster of the spot being counted, the default is all clusters.
    cluster_key : Union[str, int, list], optional
        The key of cluster in adata.obs.
    location : str, optional
        'in': The selected spots inside the contours.
        'out': The selected spots outside the contours.
        'all': both of 'in' and 'out'.
    scale : Union[str, float], optional
        scale used in subsequent analyses. If it's Visium data it can also be HE image labels (hires or lower).
        Most of the time you don't need to change this
    spatial_in_obsm : str, optional
        The key of spatial coordinates in adata.obsm
    frac : Union[int, float], optional
        The highest degree of a polynomial regression or lowess regression smoothness.
    sd : int, optional
        The coefficient of the standard deviation of the tail treatment.
    drop_zero : bool, optional
        Whether to remove all spots with 0 expression
    inplace : bool, optional
        Whether to change the original adata.

    Returns
    -------
    - :attr:`anndata.AnnData.uns` ``['SOAPy']['poly']['dic_crd_poly'] or ['SOAPy']['loess']['dic_crd_loess']`` - Store
        the shape of curves
    - :attr:`anndata.AnnData.uns` ``['SOAPy']['poly']['df_param_poly'] or ['SOAPy']['loess']['df_param_loess']`` - Store
        additional params for curves


    """
    adata = _check_adata_type(adata, spatial_in_obsm, inplace)

    if gene_name == 'all':
        gene_name = adata.var_names.tolist()

    New_tendency = SpatialTendency(adata=adata,
                                   gene_name=gene_name,
                                   radius=radius,
                                   scale=scale,
                                   clusters=clusters,
                                   cluster_key=cluster_key,
                                   spatial_in_obsm=spatial_in_obsm,
                                   )
    if method == 'poly':
        if frac is None:
            frac = 4
        adata = New_tendency.polynomialRegression(mask=mask,
                                                  frac=frac,
                                                  sd=sd,
                                                  location=location,
                                                  drop_zero=drop_zero,
                                                  )
    elif method == 'loess':
        if frac is None:
            frac = 0.6
        adata = New_tendency.loess(mask=mask,
                                   frac_0_1=frac,
                                   sd=sd,
                                   location=location,
                                   drop_zero=drop_zero
                                   )
    else:
        logg.error(f'{method} is not in [\'poly\', \'loess\']', exc_info=True)
        raise ValueError()

    return adata


def gene_cluster(
        adata: ad.AnnData,
        k: int,
        norm: bool = True,
        method: Literal['poly', 'loess'] = 'poly',
        pvalue: Optional[float] = None,
        fdr: bool = False,
        range_min: Optional[float] = None,
        correlation: Optional[float] = None,
        best_k_sdbw: bool = True,
        num_spots: int = 50,
        inplace: bool = True,
) -> anndata.AnnData:
    """
    The regression curves of multiple genes are clustered, and the curves can be screened by adjusting a variety of
    indicators

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial omics data and spatial information.
    k : int
        The number of cluster
    norm : bool, optional
        If true, data normalization
    method : Literal['poly', 'loess'], optional
        Polynomial regression(poly) and Lowess regression(loess).
    pvalue : float, optional
        Threshold for the p-value in Polynomial regression.
    fdr : bool, optional
        If True, the p-values were corrected for FDR.
    range_min : float, optional
        Genes whose expression range is less than this value are discarded.
    correlation : float, optional
        Threshold for the correlation
    best_k_sdbw : bool, optional
        Automated cluster number screening using sdbw.
    num_spots : int, optional
        The number of sampling points in curves clustering.
    inplace : bool, optional
        Whether to change the original adata.

    Returns
    -------
    - :attr:`anndata.AnnData.uns` ``['SOAPy']['poly']['df_param_poly'] or ['SOAPy']['loess']['df_param_loess']`` - Store
        additional params for curves

    """
    adata = _check_adata_type(adata, 'spatial', inplace)

    if method == 'poly':
        poly_data = _get_info_from_sample(adata, sample_id=None, key='poly')
        dic_crd = poly_data['dic_crd_poly']
        params = poly_data['df_param_poly']
    elif method == 'loess':
        loess_data = _get_info_from_sample(adata, sample_id=None, key='loess')
        dic_crd = loess_data['dic_crd_loess']
        params = loess_data['df_param_loess']
    else:
        logg.error(f'{method} is not in [\'poly\', \'loess\']', exc_info=True)
        raise ValueError()

    New_gc = GeneCluster(dic_crd=dic_crd,
                         method=method,
                         params=params,
                         norm=norm,
                         pvalue=pvalue,
                         fdr=fdr,
                         range_min=range_min,
                         correlation=correlation)
    New_gc.clusterTheGeneTendency(k=k, sdbw=best_k_sdbw, num=num_spots)

    gene_cluster_params = {'gene_cluster': New_gc.df_params, 'k': New_gc.k, 'method': method}
    _add_info_from_sample(adata, sample_id=None, keys='gene_cluster', add=gene_cluster_params)

    return adata
