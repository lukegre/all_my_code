"""
Contains a function to quickly plot xarray datasets on a map

Loading the script creates a method for xr.DataArrays that can be used as follows:

da.mean('time').map()

Defaults can also be changed by changing values in the rcMaps dictionary.
I haven't figured out how this can be changed in notebooks, but you can just
change these with the **kwargs argument.

"""

import warnings
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from functools import wraps
from matplotlib import MatplotlibDeprecationWarning

# should speed up plotting in cartopy > 0.20
os.environ["PYPROJ_GLOBAL_CONTEXT"] = "ON"

from cartopy import crs  # noqa E402

warnings.filterwarnings("ignore", ".*All-NaN slice encountered.*")
warnings.filterwarnings("ignore", ".*invalid value encountered in less.*")
warnings.filterwarnings("ignore", ".*convolution.*")
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


rcMaps = {
    # these are specific to the map_subplot function
    "proj": crs.PlateCarree(central_longitude=0),
    "land_color": "none",
    "coast_res": "110m",
    "coast_lw": 0.5,
    "robust": True,
    "round": False,
    # you can add any colorbar kwarg here and it will be set as the default
    "colorbar.pad": 0.02,
    "colorbar.fraction": 0.1,
}


def map_subplot(
    pos=111,
    proj=rcMaps["proj"],
    round=rcMaps["round"],
    land_color=rcMaps["land_color"],
    coast_res=rcMaps["coast_res"],
    fig=None,
    dpi=90,
    figsize=None,
    **kwargs
):
    """
    Makes an axes object with a cartopy projection for the current figure

    Parameters
    ----------
    pos: int/list [111]
        Either a 3-digit integer or three separate integers
        describing the position of the subplot. If the three
        integers are *nrows*, *ncols*, and *index* in order, the
        subplot will take the *index* position on a grid with *nrows*
        rows and *ncols* columns. *index* starts at 1 in the upper left
        corner and increases to the right.

        *pos* is a three digit integer, where the first digit is the
        number of rows, the second the number of columns, and the third
        the index of the subplot. i.e. fig.add_subplot(235) is the same as
        fig.add_subplot(2, 3, 5). Note that all integers must be less than
        10 for this form to work.
    proj: crs.Projection()
        the cartopy coord reference system object to create the projection.
        Defaults to crs.PlateCarree(central_longitude=205) if not given
    round: bool [True]
        If the projection is stereographic, round will cut the corners and
        make the plot round
    land_color: str ['w']
        the color of the land patches
    coast_res: str ['110m']
        the resolution at which coastal lines are plotted. Valid options are
        110m, 50m, 10m
    **kwargs:
        passed to fig.add_subplot(**kwargs)

    """
    from cartopy import feature, crs
    import matplotlib.path as mpath

    fig = plt.gcf()

    is_default_width = fig.get_figwidth() == plt.rcParams["figure.figsize"][0]
    is_default_height = fig.get_figheight() == plt.rcParams["figure.figsize"][1]
    if is_default_width and is_default_height:
        n_row = pos // 100
        n_col = (pos - (n_row * 100)) // 10
        width = n_col * 8
        height = n_row * 3.5
        fig.set_size_inches(width, height)

    ax = fig.add_subplot(pos, projection=proj, **kwargs)

    # makes maps round
    stereo_maps = (
        crs.Stereographic,
        crs.NorthPolarStereo,
        crs.SouthPolarStereo,
    )
    if isinstance(ax.projection, stereo_maps) & round:

        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.475
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)

        ax.set_boundary(circle, transform=ax.transAxes)

    # adds features
    if coast_res == "110m":
        ax.add_feature(feature.LAND, zorder=4, color=land_color)
    else:
        ax.add_feature(
            feature.NaturalEarthFeature(
                "physical", "land", coast_res, facecolor=land_color
            )
        )

    ax.coastlines(
        resolution=coast_res, color="black", linewidth=rcMaps["coast_lw"], zorder=5
    )
    ax.outline_patch.set_lw(rcMaps["coast_lw"])
    ax.outline_patch.set_zorder(5)

    return {"ax": ax, "transform": crs.PlateCarree()}


def fill_lon_gap(xds):

    lon = xds.dims[-1]
    if xds[lon].min() < -10:
        x = np.arange(-180.5, 180)
    else:
        x = np.arange(0.5, 361)

    xds = xds.sel(**{lon: x}, method="nearest").assign_coords(**{lon: x})
    return xds


@xr.register_dataarray_accessor("map")
class Mapping(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._lon_name = self._obj.dims[-1]

    @wraps(map_subplot)
    def __call__(self, **kwargs):
        """Plot 2D data on a map. See map.pcolormesh for all call arguments"""
        return self.pcolormesh(**kwargs)

    def _plot(self, plot_func="pcolormesh", **kwargs):
        da = self._obj.astype(float)
        da = da.squeeze()
        if np.ndim(da) != 2:
            raise ValueError("Can only plot 2D arrays with maps")

        plot_kwargs = ["levels", "cmap", "vmin", "vmax"]
        da_props = {k: da.attrs[k] for k in plot_kwargs if k in da.attrs}

        da = da.assign_coords(lon=lambda x: x[self._lon_name] % 360).sortby(
            self._lon_name
        )
        da = fill_lon_gap(da)

        self._get_cbar_kwargs(kwargs)
        map_kwargs = self._get_map_kwargs(kwargs)
        props = dict(robust=rcMaps["robust"], **map_subplot(**map_kwargs))
        props.update(da_props)
        props.update(kwargs)
        img = getattr(da.plot, plot_func)(**props)

        if hasattr(img, "ax"):
            img.axes = img.ax

        self.axes = img.axes
        img.set_title = self._text
        img.set_extent = self._set_extent
        img.figure = self.axes.get_figure()

        if "pretty_name" in da.attrs and hasattr(img, "colorbar"):
            img.colorbar.set_label(da.attrs["pretty_name"])

        return img

    @wraps(map_subplot)
    def contourf(self, **kwargs):
        return self._plot(**kwargs, plot_func="contourf")

    @wraps(map_subplot)
    def pcolormesh(self, **kwargs):
        return self._plot(**kwargs, plot_func="pcolormesh")

    @wraps(map_subplot)
    def contour(self, **kwargs):
        return self._plot(**kwargs, plot_func="contour")

    def _text(
        self, s, x=90, y=50, ha="center", va="center", weight="bold", size=12, **props
    ):
        """
        Write a title to the map, rather than above the map.
        Will remove any axes titles. These can be returned with img.axes.set_title

        Parameters
        ----------
        s : str
            the text that will be the title
        x : float [90]
            the longitude location of the text
        y : float [50]
            the latitude location of the text

        For the remaining parameters, see plt.text

        Returns
        -------
        plt.text object
        """
        from cartopy import crs

        kwargs = dict(transform=crs.PlateCarree(), zorder=30)
        kwargs.update(**props, **dict(ha=ha, va=va, weight=weight, size=size))
        self.axes.set_title("")
        text = self.axes.text(x, y, s, **kwargs)
        return text

    def _set_extent(self, extent):
        """
        A wrapper for setting the extent in map projections

        set_extent is broken for global non-cylindrical projections in cartopy

        Parameters
        ----------
        extent : list
            the [left, right, bottom, top] extents in degrees

        Returns
        -------
        plt.line : a line object that extends the figure to the outer limits
            unless img.axes.set_extent is used
        """
        from cartopy.crs import PlateCarree

        x0, x1, y0, y1 = extent
        return self.axes.plot(
            [x0, x1], [y0, y1], lw=0, zorder=10, transform=PlateCarree()
        )

    @staticmethod
    def _get_cbar_kwargs(kwargs):
        cbar_defaults = {
            k.split(".")[1]: v for k, v in rcMaps.items() if k.startswith("colorbar")
        }
        cbar_opts = cbar_defaults
        if kwargs.get("add_colorbar", True):
            if "cbar_kwargs" in kwargs:
                cbar_opts.update(kwargs["cbar_kwargs"])
            kwargs["cbar_kwargs"] = cbar_opts
        return kwargs

    @staticmethod
    def _get_map_kwargs(kwargs):
        possible_kwargs = "pos", "land_color", "proj", "round", "coast_res"
        map_kwargs = {k: v for k, v in kwargs.items() if k in possible_kwargs}
        for k in map_kwargs:
            kwargs.pop(k)
        return map_kwargs
