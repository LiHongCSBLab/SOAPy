import copy
from scipy import stats
from typing import Optional, Union
from tqdm import tqdm
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import cv2 as cv
import statsmodels.api as sm
import warnings


######################################
#                                    #
#           Spatial Tendency         #
#                                    #
######################################

class SpatialTendency(object):

    def __init__(self,
                 adata: sc.AnnData,
                 gene_Name: Union[str, list],
                 radius: int,
                 scale: Union[float, str] = None,
                 clusters: Union[str, int, list] = 'all',
                 cluster_label: Optional[str] = None,
                 library_id: Optional[str] = None,
                 ) -> None:
        """

        Parameters
        ----------
        adata
        gene_Name
        radius
        scale
        clusters
        cluster_label
        """

        if clusters != 'all':
            if type(clusters) != list:
                clusters = [clusters]
            adata.obs = adata.obs.loc[[(i in clusters) for i in adata.obs[cluster_label]], :]

        if type(adata.X) is not np.ndarray:
            df = pd.DataFrame(adata.X.toarray())
        else:
            df = pd.DataFrame(adata.X)

        df.index = adata.obs.index
        df.columns = adata.var.index
        df_pixel = adata.obsm.to_df()
        df_pixel = df_pixel.loc[:, ['spatial1', 'spatial2']]

        if scale is None or scale == 'hires':
            if library_id is None:
                library_id = list(adata.uns['spatial'].keys())[0]
            self.name = 'hires'
            scale = adata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef']
        elif scale == 'lowres':
            if library_id is None:
                library_id = list(adata.uns['spatial'].keys())[0]
            self.name = 'lowres'
            scale = adata.uns['spatial'][library_id]['scalefactors']['tissue_lowres_scalef']
        else:
            self.name = 'user_defined'

        adata.obs['imagerow'] = df_pixel.loc[adata.obs_names, 'spatial2'] * scale
        adata.obs['imagecol'] = df_pixel.loc[adata.obs_names, 'spatial1'] * scale

        self.adata = adata
        self.scale = str(scale)

        try:
            adata.uns['SOAPy']
        except KeyError:
            adata.uns['SOAPy'] = {}

        try:
            adata.uns['SOAPy'][self.name]
        except KeyError:
            adata.uns['SOAPy'][self.name] = {}

        self.express = df
        self.coordinate = adata.obs
        if type(gene_Name) != list:
            gene_Name = [gene_Name]
        self.gene_Name = gene_Name
        self.radius = radius
        self.inSpots = []

        if radius <= 0:
            raise RuntimeError("Radius must greater than 0")
        self.__structureCheck()

    def __structureCheck(self, ) -> None:
        """
        checking for mismatched data

        """

        shape_express = self.express.shape
        shape_coordinate = self.coordinate.shape

        if len(shape_express) > 2:
            raise RuntimeError("Express must be 2-dimensionality")
        if len(shape_coordinate) > 2:
            raise RuntimeError("Coordinate must be 2-dimensionality")

    def __distance_spot_and_contour(self,
                                    contour: np.ndarray,
                                    location: Optional[str],
                                    ) -> pd.DataFrame:
        """

        Parameters
        ----------
        contour
        location

        Returns
        -------

        """

        assert location in ['in', 'out', 'all'], 'location must in [in ,out, all]'

        point_x = []
        point_y = []
        dists = []
        exps = []

        for gene_name in self.gene_Name:
            exps.append([])
        threshold = self.__find_threshold(contour, self.radius)
        crds = self.coordinate

        for index in crds.index:
            point = (crds.at[index, 'imagecol'], crds.at[index, 'imagerow'])
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
            for index_gene, gene_name in enumerate(self.gene_Name):
                exps[index_gene].append(self.express.at[index, gene_name])
            point_x.append(point[0])
            point_y.append(point[1])
            dists.append(dist)

        contour_dis_and_express = {'point_x': point_x, 'point_y': point_y, 'distance': dists}

        for index, gene_name in enumerate(self.gene_Name):
            contour_dis_and_express[gene_name] = exps[index]

        contour_dis_and_express = pd.DataFrame(contour_dis_and_express)

        return contour_dis_and_express

    def __find_threshold(self,
                         contour: np.ndarray,
                         radius: int,
                         ) -> list:
        """

        Parameters
        ----------
        contour
        radius

        Returns
        -------

        """

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
        预过滤spot，保留可能在距离内的点

        :param point:点坐标（x，y）
        :param threshold:范围轮廓
        :return:是否可能在最小范围内的bool型
        """
        if point[0] < threshold[0] or point[0] > threshold[1]:
            return False
        if point[1] < threshold[2] or point[1] > threshold[3]:
            return False

        return True

    def get_dist_and_express(self,
                             mask: np.ndarray,
                             location: Optional[str] = 'all',
                             ) -> pd.DataFrame:
        """

        Parameters
        ----------
        mask
        location

        Returns
        -------

        """
        contours, hierarchy = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        index = 0
        dis_and_express = []
        x_all = []
        y_all = []
        dis_all = []
        express_all = []

        for index_gene, gene_name in enumerate(self.gene_Name):
            express_all.append([])

        # 每个点的数据
        for contour in contours:
            contour_dis_and_express = self.__distance_spot_and_contour(contour, location)
            dis_and_express.append([contour, dis_and_express])
            x_all = x_all + contour_dis_and_express['point_x'].tolist()
            y_all = y_all + contour_dis_and_express['point_y'].tolist()
            dis_all = dis_all + contour_dis_and_express['distance'].tolist()
            for index_gene, gene_name in enumerate(self.gene_Name):
                express_all[index_gene] = express_all[index_gene] + contour_dis_and_express[gene_name].tolist()

            index += 1

        # 所有点数据
        dis_and_express_all = {'x': x_all, 'y': y_all, 'distance': dis_all}
        for index_gene, gene_name in enumerate(self.gene_Name):
            dis_and_express_all[gene_name] = express_all[index_gene]

        dis_and_express_all = pd.DataFrame(dis_and_express_all)
        dis_and_express_all = self.__remove_duplicates(dis_and_express_all)

        for index_point in dis_and_express_all.index:
            if (dis_and_express_all.loc[index_point, 'x'], dis_and_express_all.loc[index_point, 'y']) in self.inSpots:
                dis_and_express_all.drop(labels=index_point, inplace=True)

        self.adata.uns['SOAPy'][self.name]['dis_and_express_all'] = dis_and_express_all
        self.adata.uns['SOAPy'][self.name]['contours'] = contours

        return dis_and_express_all

    def lowess(self,
               mask: np.ndarray,
               frac_0_1: float = 0.6666,
               sd: Optional[int] = None,
               location: Optional[str] = 'all',
               ) -> tuple:
        """

        Parameters
        ----------
        mask
        frac_0_1
        sd
        location

        Returns
        -------

        """
        warnings.filterwarnings('ignore')
        dis_and_express = self.get_dist_and_express(mask, location)

        dic_crd = {}
        list_name = []
        list_r_sq_lowess = []
        list_corr_gene_distance = []
        list_range_lowess = []

        mark = 0
        for gene_name in tqdm(self.gene_Name):
            xest, yest, r_sq, corr, ran = self.__ST_LOESS(dis_and_express, gene_name, frac_0_1, sd=sd)
            if mark == 0:
                dic_crd['Xest'] = xest
                mark = 1
            dic_crd[gene_name] = yest
            list_name.append(gene_name)
            list_r_sq_lowess.append(r_sq)
            list_corr_gene_distance.append(corr)
            list_range_lowess.append(ran)

        dict_gene = {
            'R_square': list_r_sq_lowess,
            'correlation': list_corr_gene_distance,
            'range': list_range_lowess
        }

        df_param_lowess = pd.DataFrame(dict_gene, index=list_name)

        self.adata.uns['SOAPy'][self.name]['dic_crd_lowess'] = dic_crd
        self.adata.uns['SOAPy'][self.name]['df_param_lowess'] = df_param_lowess

        return dic_crd, df_param_lowess

    def polynomialRegression(self,
                             mask: np.ndarray,
                             frac: int = 4,
                             sd: Optional[int] = None,
                             location: Optional[str] = 'out', ):

        warnings.filterwarnings('ignore')
        dis_and_express = self.get_dist_and_express(mask, location)

        dic_crd = {}
        list_name = []
        list_p_value_Poly = []
        list_param_Poly = []
        list_range_Poly = []
        list_corr_gene_distance = []

        mark = 0
        for gene_name in tqdm(self.gene_Name):
            f_pvalue, xest, yest, param, ran, corr = self.__polynomial_regression(dis_and_express, gene_name, frac,
                                                                                  sd=sd)
            if mark == 0:
                dic_crd['Xest'] = xest
                mark = 1
            dic_crd[gene_name] = yest
            list_name.append(gene_name)
            list_p_value_Poly.append(f_pvalue)
            list_param_Poly.append(param)
            list_range_Poly.append(ran)
            list_corr_gene_distance.append(corr)

        dict_gene = {
            'p_value': list_p_value_Poly,
            'param': list_param_Poly,
            'range': list_range_Poly,
            'correlation': list_corr_gene_distance
        }

        df_param_poly = pd.DataFrame(dict_gene, index=list_name)

        self.adata.uns['SOAPy'][self.name]['dic_crd_poly'] = dic_crd
        self.adata.uns['SOAPy'][self.name]['df_param_poly'] = df_param_poly

        return dic_crd, df_param_poly

    def __polynomial_regression(self,
                                dis_and_express: pd.DataFrame,
                                gene_name: str,
                                frac: int,
                                sd: Optional[int] = None,
                                ):

        x = dis_and_express['distance']
        y = dis_and_express[gene_name]
        x = np.array(x)
        y = np.array(y)
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

        corr = np.corrcoef(x=x, y=y)[0, 1]

        return res.f_pvalue, x, y_fitted, res.params, ran, corr

    def __ST_LOESS(self,
                   dis_and_express: pd.DataFrame,
                   gene_name: str,
                   frac_0_1: float,
                   sd: Optional[int] = None,
                   ):
        """

        Parameters
        ----------
        dis_and_express
        gene_name
        frac_0_1
        sd

        Returns
        -------

        """
        # 进行lowess回归
        lowess = sm.nonparametric.lowess
        x = dis_and_express['distance']
        y = dis_and_express[gene_name]

        if sd != None:
            mu = y.mean()
            std = y.std()
            dis_and_express = dis_and_express.drop(dis_and_express[dis_and_express[gene_name] < mu - sd * std].index)
            dis_and_express = dis_and_express.drop(dis_and_express[dis_and_express[gene_name] > mu + sd * std].index)

            x = dis_and_express['distance']
            y = dis_and_express[gene_name]

        x = np.array(x)
        y = np.array(y)
        ind = x.argsort()
        y = y[ind]
        x = x[ind]

        model = lowess(endog=y, exog=x, frac=frac_0_1)
        yest = model[:, 1]

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

        return x, yest, r_sq, corr, ran

    def __remove_duplicates(self,
                            dis_and_express: pd.DataFrame,
                            ) -> pd.DataFrame:
        """

        Parameters
        ----------
        dis_and_express

        Returns
        -------

        """
        dis_and_express['distance_abs'] = abs(dis_and_express['distance'])
        dis_and_express = dis_and_express.sort_values(by='distance_abs')
        dis_and_express = dis_and_express.drop_duplicates(subset=['x', 'y'], keep='first')
        for index in dis_and_express.index:
            if (dis_and_express.at[index, 'x'], dis_and_express.at[index, 'y']) in self.inSpots:
                dis_and_express.drop(index=index)
        dis_and_express.reset_index(drop=True)
        return dis_and_express

    def wilcoxon(self,
                 mask: np.ndarray,
                 ) -> None:
        """

        Parameters
        ----------
        mask

        Returns
        -------

        """
        self.__structureCheck()
        df_dist_express = self.get_dist_and_express(mask)
        print(df_dist_express)

    def spearman(self,
                 mask: np.ndarray,
                 gap: float
                 ) -> list:
        """

        Parameters
        ----------
        mask
        gap

        Returns
        -------

        """
        self.__structureCheck()
        df = self.get_dist_and_express(mask)
        distance = df['distance'].tolist()
        express = df['express'].tolist()
        dis_level = gap
        level = 1
        for i in range(len(distance)):
            if distance[i] < dis_level:
                distance[i] = level
            else:
                while (distance[i] < dis_level):
                    dis_level += gap
                level += 1
                distance[i] = level
        coef, pvalue = stats.spearmanr(distance, express)
        return [coef, pvalue]


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

        Parameters
        ----------
        dic_crd
        method
        params
        scale
        pvalue
        fdr
        range_min
        correlation
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
        self.__select_gene()

    def clusterTheGeneTendency(self, k: int, sdbw: bool = True):
        """

        Parameters
        ----------
        k

        Returns
        -------

        """
        self.__select_gene()

        if self.method == 'lowess':
            self.__cluster_lowess(k, sdbw)
        elif self.method == 'PR':
            self.__cluster_Poly(k, sdbw)

    def __select_gene(self):

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

    def __cluster_Poly(self, k, sdbw):

        from .utils import _best_k
        list_gene = []
        x = self.gene_express['Xest']

        for gene_name in self.df_params.index:
            y = self.gene_express[gene_name]
            params = self.df_params.loc[gene_name, 'param']
            xgrid = np.linspace(x.min(), x.max())
            ygrid = []
            for x0 in xgrid:
                ygrid.append(sum([param * x0 ** i for i, param in enumerate(params)]))
            if self.norm:
                ygrid = (np.array(ygrid) - y.min()) / (y.max() - y.min())
            list_gene.append(ygrid)

        array_gene = np.array(list_gene)

        res, km = _best_k(array_gene, k, sdbw)

        self.df_params['cluster'] = res
        self.best_k = km

    def __cluster_lowess(self, k, sdbw):
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
        self.best_k = km


def ANOVA(adata: ad.AnnData,
          cluster_label,
          score_label,
          ):
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

    return p1


def spatial_tendency(adata: ad.AnnData,
                     mask: np.ndarray,
                     radius: int,
                     method: str = 'PR',
                     gene_Name: Union[str, list] = 'all',
                     clusters: Union[str, int, list] = 'all',
                     cluster_label: Optional[str] = None,
                     library_id: Optional[str] = None,
                     scale: Union[float, str] = 'hires',
                     frac: Union[int, float] = None,
                     sd: Optional[int] = None,
                     location: Optional[str] = 'all',
                     ) -> tuple[dict, pd.DataFrame]:
    """

    Parameters
    ----------
    adata
    mask
    radius
    method
    gene_Name
    clusters
    cluster_label
    scale
    frac
    sd
    location

    Returns
    -------

    """
    # adata = copy.deepcopy(adata)
    if gene_Name == 'all':
        gene_Name = adata.var_names.tolist()

    New_tendency = SpatialTendency(adata=adata,
                                   gene_Name=gene_Name,
                                   radius=radius,
                                   scale=scale,
                                   clusters=clusters,
                                   cluster_label=cluster_label,
                                   library_id=library_id,
                                   )
    if method == 'PR':
        if frac is None:
            frac = 4
        dic_crd, df_param = New_tendency.polynomialRegression(mask=mask,
                                                              frac=frac,
                                                              sd=sd,
                                                              location=location)
    elif method == 'lowess':
        if frac is None:
            frac = 0.6
        dic_crd, df_param = New_tendency.lowess(mask=mask,
                                                frac_0_1=frac,
                                                sd=sd,
                                                location=location)
    else:
        raise ValueError('\'method\' must in [\'PR\', \'lowess\']')

    return dic_crd, df_param


def gene_cluster(adata: ad.AnnData,
                 k: int,
                 scale: Union[float, str] = 'hires',
                 norm: bool = True,
                 method: str = 'PR',
                 pvalue: Optional[float] = None,
                 fdr: bool = False,
                 range_min: Optional[float] = None,
                 correlation: Optional[float] = None,
                 select_with_sdbw: bool = True,
                 ) -> tuple[int, pd.DataFrame]:
    """

    Parameters
    ----------
    adata
    k
    scale
    norm
    method
    pvalue
    fdr
    range_min
    correlation
    select_with_sdbw

    Returns
    -------

    """
    if method == 'PR':
        dic_crd = adata.uns['SOAPy'][scale]['dic_crd_poly']
        params = adata.uns['SOAPy'][scale]['df_param_poly']
    elif method == 'lowess':
        dic_crd = adata.uns['SOAPy'][scale]['dic_crd_lowess']
        params = adata.uns['SOAPy'][scale]['df_param_lowess']
    else:
        raise NameError

    New_gc = GeneCluster(dic_crd=dic_crd,
                         method=method,
                         params=params,
                         norm=norm,
                         pvalue=pvalue,
                         fdr=fdr,
                         range_min=range_min,
                         correlation=correlation)
    New_gc.clusterTheGeneTendency(k=k + 1, sdbw=select_with_sdbw)
    if method == 'PR':
        adata.uns['SOAPy'][scale]['gene_cluster_poly'] = New_gc.df_params
        adata.uns['SOAPy'][scale]['k_poly'] = New_gc.best_k
    elif method == 'lowess':
        adata.uns['SOAPy'][scale]['gene_cluster_lowess'] = New_gc.df_params
        adata.uns['SOAPy'][scale]['k_lowess'] = New_gc.best_k

    return New_gc.best_k, New_gc.df_params
