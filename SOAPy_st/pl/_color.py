# define color list in SOAPy
from typing import Sequence, Union
from matplotlib.colors import ListedColormap

# default palette
# 10 color
color_list_10 = [
    "#765005", "#0780cf", "#fa6d1d", "#0e2c82", "#b6b51f",
    "#da1f18", "#701866", "#f47a75", "#009db2", "#024b51"
]

# 50 color
color_list_50 = [
    '#5050FFFF', '#CE3D32FF', '#749B58FF', '#F0E685FF', '#466983FF', '#BA6338FF', '#5DB1DDFF',
    '#802268FF', '#6BD76BFF', '#D595A7FF', '#924822FF', '#837B8DFF', '#C75127FF', '#D58F5CFF',
    '#7A65A5FF', '#E4AF69FF', '#3B1B53FF', '#CDDEB7FF', '#612A79FF', '#AE1F63FF', '#E7C76FFF',
    '#5A655EFF', '#CC9900FF', '#99CC00FF', '#A9A9A9FF', '#CC9900FF', '#99CC00FF', '#00D68FFF',
    '#14FFB1FF', '#00CC99FF', '#0099CCFF', '#0A47FFFF', '#4775FFFF', '#FFC20AFF', '#FFD147FF',
    '#990033FF', '#991A00FF', '#996600FF', '#809900FF', '#339900FF', '#00991AFF', '#009966FF',
    '#008099FF', '#003399FF', '#1A0099FF', '#660099FF', '#990080FF', '#D60047FF', '#FF1463FF',
    '#00D68FFF'
]

# default colorbar
cmap_default = 'parula'


def _get_palette(categorical, sort_order: bool = True, palette: Union[Sequence, ListedColormap] = None) -> dict:
    are_all_str = all(map(lambda x: isinstance(x, str), categorical))
    if not are_all_str:
        categorical = str(categorical)

    if sort_order:
        categorical = sorted(categorical)

    if palette is None:
        if len(categorical) <= 10:
            palette = color_list_10
        else:
            palette = color_list_50

    if isinstance(palette, ListedColormap):
        palette = palette.colors

    palette = palette[0: len(categorical)]
    palette = dict(zip(categorical, palette))
    return palette