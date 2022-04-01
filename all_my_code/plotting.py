import xarray as xr
from matplotlib import pylab as plt


def pimp_plot(ax=None, **grid_kwargs):
    """
    Runs the following changes on a plot:
    - no x-label (assuming year)
    - horizontal gridlines will be made grey with width 0.5
    - remove right and top axis lines
    - make remaining axis lines gray

    Parameters
    ----------
    ax : plt.Axes
        the axes object that will be changed
    grid_kwargs : dict (or keyword arguements)
        grids are ax.hlines drawn for all ytick locations. Gives more
        flexibility with keyword arguments that are passed to ax.hlines

    Returns
    -------
    ax : plt.Axes
        changed axes
    """

    if ax is None:
        ax = plt.gca()

    ax.set_xlabel("")

    grd_kwds = dict(lw=0.5, color="lightgrey")
    grd_kwds.update(grid_kwargs)

    xlim = ax.get_xlim()
    ax.hlines(ax.get_yticks(), *xlim, **grd_kwds)

    ax.xaxis.set_tick_params(color="#aaaaaa", which="both")
    ax.yaxis.set_tick_params(color="#aaaaaa", which="both")
    [ax.spines[s].set_visible(False) for s in ["right", "top"]]
    [ax.spines[s].set_color("#CCCCCC") for s in ax.spines]
    [ax.spines[s].set_linewidth(1) for s in ax.spines]
    ax.set_xlim(*xlim)

    return ax


def save_figures_to_pdf(fig_list, pdf_name, **savefig_kwargs):
    """
    Saves a list of figure objects to a pdf with multiple pages.

    Parameters
    ----------
    fig_list : list
        list of figure objects
    pdf_name : str
        path to save pdf to.
    savefig_kwargs : key-value pairs passed to ``Figure.savefig``

    Returns
    -------
    None
    """
    import matplotlib.backends.backend_pdf
    from matplotlib import pyplot as plt

    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)

    kwargs = dict(dpi=120)
    kwargs.update(savefig_kwargs)

    for fig in fig_list:  # will open an empty extra figure :(
        pdf.savefig(fig.number, **kwargs)

    pdf.close()
    plt.close("all")


def close_lon_gap(xda):
    lon = xda.dims[-1]
    halflon = xda[lon].size / 2

    if xda[:, [0, -1]].isnull().all():
        xda = (
            xda.roll(**{lon: halflon}, roll_coords=False)
            .interpolate_na("lon", limit=6)
            .roll(**{lon: -halflon}, roll_coords=False)
        )
    return xda


@xr.register_dataarray_accessor("plot_map")
class CartopyMap(object):
    """
    Plot the given 2D array on a cartopy axes (assuming that Lat and Lon exist)
    The default projection is PlateCarree, but can be:
        cartopy.crs.<ProjectionName>()

    If the projection is Stereographic the plot will be round unless
    the keyword arguement `round` is set False.

    If you would like to create a figure with multiple subplots
    you can pass an axes object to the function with keyword argument `ax,
    BUT then you need to specify the projection when you create the axes:
        plt.axes([x0, y0, w, h], projection=cartopy.crs.<ProjectionName>())

    Additional keywords can be given to the function as you would to
    the xr.DataArray.plot function. The only difference is that `robust`
    is set to True by default.

    The function returns a GeoAxes object to which features can be added with:
        ax.add_feature(feature.<FeatureName>, **kwargs)
    By default, LAND and COASTLINE are added, but can be removed by
    setting default_features=False
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, ax=None, proj=None, round=True, land_color='w', default_features=True, **kwargs):
        return self._cartopy(
            ax=ax, proj=proj, round=round, default_features=default_features, land_color=land_color, **kwargs
        )

    @staticmethod
    def _close_lon_gap(xda):
        lon = xda.dims[-1]
        halflon = xda[lon].size / 2

        if xda[:, [0, -1]].isnull().all():
            xda = (
                xda.roll(**{lon: halflon}, roll_coords=False)
                .interpolate_na("lon", limit=6)
                .roll(**{lon: -halflon}, roll_coords=False)
            )
        return xda

    def _cartopy(self, ax=None, proj=None, round=True, default_features=True, land_color='w', **kwargs):
        import matplotlib.pyplot as plt
        from cartopy import feature, crs

        xda = self._obj.squeeze()

        cntr_lon = getattr(kwargs, "central_longitude", -155)
        assert xda.ndim == 2, "The array must be two dimensional"

        if ax is None:
            tighten = True
            proj = crs.PlateCarree(cntr_lon) if proj is None else proj
            fig, ax = plt.subplots(
                1, 1, figsize=[11, 4], dpi=120, subplot_kw={"projection": proj}
            )
        else:
            tighten = False

        # makes maps round
        stereo_maps = (
            crs.Stereographic,
            crs.NorthPolarStereo,
            crs.SouthPolarStereo,
        )

        proj = ax.projection
        if isinstance(proj, stereo_maps) & round:
            import matplotlib.path as mpath
            import numpy as np

            theta = np.linspace(0, 2 * np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)

            ax.set_boundary(circle, transform=ax.transAxes)

        lat = xda[xda.dims[-2]].values
        lon = xda[xda.dims[-1]].values

        top = min(abs(lat - 70)) < 10
        bot = min(abs(lat + 75)) < 10
        left = min(abs(lon + 180)) < 10
        right = min(abs(lon - 180)) < 10

        global_domain = bot & top & left & right
        if global_domain:
            ax.plot(
                [-180, 180, 0, 0],
                [0, 0, -90, 90],
                lw=0,
                marker=".",
                ms=0.1,
                transform=crs.PlateCarree(),
            )

        # adds features
        if default_features:
            ax.add_feature(feature.LAND, color=land_color, zorder=4)
            ax.add_feature(feature.COASTLINE, lw=0.5, zorder=4)

        if "robust" not in kwargs:
            kwargs["robust"] = True
        if ("cbar_kwargs" not in kwargs) & kwargs.get("add_colorbar", True):
            kwargs["cbar_kwargs"] = {"pad": 0.02}

        axm = xda.plot(ax=ax, transform=crs.PlateCarree(), **kwargs)
        if kwargs.get("add_colorbar", True):
            ax.colorbar = axm.colorbar
        if tighten:
            fig.tight_layout()
            
        ax.outline_patch.set_zorder(5)
        ax.outline_patch.set_lw(0.5)

        return ax


def animate_xda(
    ax, xda, draw_func=None, dim="time", sname="./animation.mp4", fps=6, **plot_kwargs
):
    """
    NOTE: this function is still in development. Would be
    nice to have this as a xarray accessor
    Animate a DataArray along the first dimension.

    Parameters
    ----------
    ax : pyplot.Axes
        The axes object into which you'd like to animate
    xda : xr.DataArray
        the given dimension of the array will be iterated
        over given that the remaining axes is 2D
    draw_func : callable
        function that plots a single time slice of the data
        where a list of changed objects are returned
    dim : str
        the dimension over which the image will be iterated
    sname : str
        name to save the animation to
    fps : int
        frames per second
    plot_kwargs : {}
        keyword pairs that will be passed to the plot function
    """
    from matplotlib import animation, pyplot

    def draw(f):
        print(".", end="")
        return (xda.isel({dim: f}).plot(ax=ax, **plot_kwargs),)

    def animate(frame):
        if draw_func is None:
            return draw(frame)
        elif callable(draw_func):
            return draw_func(frame)

    fig = ax.get_figure()
    nframes = xda[dim].size
    anim = animation.FuncAnimation(
        fig, animate, frames=nframes, blit=True, repeat=False, interval=1000 / fps,
    )

    anim.save(sname, writer=animation.FFMpegWriter(fps=fps))

    pyplot.close(fig)

