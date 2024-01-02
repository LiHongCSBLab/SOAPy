import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scanpy.plotting import palettes


def transparent_back(img: np.ndarray):
    """
    Set the image's white background to transparent
    Parameters
    ----------
    img: image of numpy.ndarray

    Returns
    -------

    """
    from PIL import Image
    img = Image.fromarray(img)
    img = img.convert('RGBA')
    L, H = img.size
    color_0 = img.getpixel((2, 2))
    for h in range(H):
        for l in range(L):
            dot = (l, h)
            color_1 = img.getpixel(dot)
            if color_1 == color_0:
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot, (0, 0, 0, 0))
    return img


def niche_counter(obs,
                  niche_name,
                  cluster_name,
                  niche_label,
                  cluster_label,
                  ) -> list:
    from collections import Counter

    n_cluster = len(cluster_name)
    list_cluster = []
    for i in range(n_cluster):
        list_cluster.append([])

    for index_k in niche_name:
        df_k = obs[obs[niche_label] == index_k]
        dict_k = Counter(df_k[cluster_label].tolist())
        sum = 0
        for val in dict_k.values():
            sum += val
        for key in dict_k.keys():
            dict_k[key] = dict_k[key] / sum
        for k, cluster in enumerate(cluster_name):
            if cluster in dict_k.keys():
                list_cluster[k].append(dict_k[cluster])
            else:
                list_cluster[k].append(0)

    for i in range(n_cluster):
        list_cluster[i] = np.array(list_cluster[i])

    return list_cluster


### function from stlearn

def get_cmap(cmap):
    """Checks inputted cmap string."""
    if cmap == "vega_10_scanpy":
        cmap = palettes.vega_10_scanpy
    elif cmap == "vega_20_scanpy":
        cmap = palettes.vega_20_scanpy
    elif cmap == "default_102":
        cmap = palettes.default_102
    elif cmap == "default_28":
        cmap = palettes.default_28
    elif type(cmap) == str:  # If refers to matplotlib cmap
        cmap_n = plt.get_cmap(cmap).N
        return plt.get_cmap(cmap), cmap_n
    elif type(cmap) == matplotlib.colors.LinearSegmentedColormap:  # already cmap
        cmap_n = cmap.N
        return cmap, cmap_n

    cmap_n = len(cmap)
    cmaps = matplotlib.colors.LinearSegmentedColormap.from_list("", cmap)

    cmap_ = plt.cm.get_cmap(cmaps)

    return cmap_, cmap_n


def check_cmap(cmap):
    """Initialize cmap"""
    scanpy_cmap = ["vega_10_scanpy", "vega_20_scanpy", "default_102", "default_28"]
    stlearn_cmap = ["jana_40", "default"]
    cmap_available = plt.colormaps() + scanpy_cmap + stlearn_cmap
    error_msg = (
            "cmap must be a matplotlib.colors.LinearSegmentedColormap OR"
            "one of these: " + str(cmap_available)
    )
    if type(cmap) == str:
        assert cmap in cmap_available, error_msg
    elif type(cmap) != matplotlib.colors.LinearSegmentedColormap:
        raise Exception(error_msg)

    return cmap


def get_colors(adata, obs_key, cmap=None, label_set=None):
    """
    Retrieves colors if present in adata.uns, if not present then will set
    them as per scanpy & return in order requested.
    """
    # Checking if colors are already set #
    col_key = f"{obs_key}_colors"
    if col_key in adata.uns:
        labels_ordered = adata.obs[obs_key].cat.categories
        colors_ordered = adata.uns[col_key]
    else:  # Colors not already present
        check_cmap(cmap)
        cmap, cmap_n = get_cmap(cmap)

        if not hasattr(adata.obs[obs_key], "cat"):  # Ensure categorical
            adata.obs[obs_key] = adata.obs[obs_key].astype("category")
        labels_ordered = adata.obs[obs_key].cat.categories
        colors_ordered = [
            matplotlib.colors.rgb2hex(cmap(i / (len(labels_ordered) - 1)))
            for i in range(len(labels_ordered))
        ]
        adata.uns[col_key] = colors_ordered

    # Returning the colors of the desired labels in indicated order #
    if label_set is not None:
        colors_ordered = [
            colors_ordered[np.where(labels_ordered == label)[0][0]]
            for label in label_set
        ]

    return colors_ordered