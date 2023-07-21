from .utils import save_figures_to_pdf, label_subplots
from .line_plots import annotate_line
from . import hovmoller
from . import visualize
from . import utils

from matplotlib import pyplot as plt
from .geographic import geo_subplot, fill_lon_gap, rcMaps
from . import cmaps
from .cmaps import lighten_color


def _create_colormap_dict():
    from cmocean import cm as cmo
    from matplotlib import cm
    from .cmaps import cmap as cmap_loader
    from munch import Munch as munch

    subset = [
        "BlueYellowRed",
        "BkBlAqGrYeOrReViWh200",
        "BlGrYeOrReVi200",
        "3saw",
    ]

    maps = dict()
    for name in subset:
        cmap = cmap_loader(name)
        cm.register_cmap(name=name, cmap=cmap)
        maps[name] = cmap

    return munch(maps)


def _set_styles():
    import matplotlib as mpl
    import seaborn as sns
    from cycler import cycler

    sns.set_style("ticks")
    sns.set_palette("tab20")

    mpl.rcParams["lines.linewidth"] = 2.5

    mpl.rcParams["axes.prop_cycle"] = cycler(color=mpl.cm.tab10(range(10)))

    mpl.rcParams["axes.grid"] = False
    mpl.rcParams["axes.linewidth"] = 0.5
    mpl.rcParams["axes.edgecolor"] = "k"
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["xtick.major.size"] = 2.5
    mpl.rcParams["ytick.major.size"] = 2.5
    mpl.rcParams["xtick.major.width"] = 0.5
    mpl.rcParams["xtick.minor.size"] = 0
    mpl.rcParams["ytick.minor.size"] = 0

    mpl.rcParams["axes.xmargin"] = 0.01
    mpl.rcParams["axes.ymargin"] = 0.01

    mpl.rcParams["legend.facecolor"] = "none"
    mpl.rcParams["legend.edgecolor"] = "none"

    mpl.rcParams["figure.subplot.hspace"] = 0.07
    mpl.rcParams["figure.subplot.wspace"] = 0.07
    mpl.rcParams["figure.subplot.left"] = 0.05
    mpl.rcParams["figure.subplot.right"] = 0.95
    mpl.rcParams["figure.subplot.bottom"] = 0.075
    mpl.rcParams["figure.subplot.top"] = 0.95

    mpl.rcParams["savefig.dpi"] = 120
    mpl.rcParams["figure.dpi"] = 120
    mpl.rcParams["figure.figsize"] = [9, 3.8]


_set_styles()
cm = _create_colormap_dict()
