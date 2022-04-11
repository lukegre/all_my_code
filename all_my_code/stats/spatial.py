from ..utils import make_xarray_accessor as _make_xarray_accessor
import xarray as xr
import numpy as np
from functools import wraps as _wraps


def avg_area_weighted(da, dims=['lat', 'lon']):
    """
    Calculates the area weighted average for a data array

    Assumes latitude and longitude are named "lat", "lon" respectively

    Parameters
    ----------
    da : xr.DataArray
        The data array to average

    Returns
    -------
    xr.DataArray
        The weighted average
    """

    area = da.spatial.get_area()
    return da.weighted(area).mean(dims)


def aggregate_region(xda, region_mask=None, region_names=None, weights='area', func='mean'):
    """
    Average a data array over a region mask with area weighting

    Parameters
    ----------
    xda : xr.DataArray
        The data array to aggregate that matches the shape of region_mask
        Assumes that latitudes are named lat, and longitudes are named lon
    region_mask : xr.DataArray
        A mask array with the same shape as xda that defines the regions
        to aggregate over
    weights : xr.DataArray, [area]
        A data array with the same shape as the mask that is used to weight
        the aggregation. If None, all values are weighted equally. If 'area',
        then the area will be calculated from region_mask.
    func : str [mean]
        The function to use for aggregation. Must be a method of a 
        weighted_mean object.

    Returns
    -------
    xr.DataArray
        The aggregated data array
    """
    from warnings import warn

    assert isinstance(region_mask, xr.DataArray), "region_mask must be a DataArray"
    assert region_mask.dtype == np.int_, "region_mask must be an integer array"
    if region_mask.ndim > 2:
        warn("Your mask array has more than two dimensions. This might take some time")

    # first make sure that weights is a string before we do the string compare
    if isinstance(weights, str):
        # areas will be weighted be area 
        if (weights == 'area'):
            weights = region_mask.spatial.get_area()

    groups = xda.groupby(region_mask)

    if region_names is not None:
        assert len(groups) == len(region_names), "Number of groups does not match number of region names"

    regional = []
    for r, da in groups:
        da = da.unstack()
        if weights is not None:
            da = da.weighted(weights)
        func_ = getattr(da, func, None)
        if func_ is not None:
            da = func_(['lat', 'lon'])
        else:
            raise ValueError(
                f"{func} is not available as an aggregation function for weighted arrays. "
                f"Try setting weights to None"
                )
        da = da.assign_coords(region=r)
        regional += da,
        
    regional = xr.concat(regional, 'region')

    if region_names is not None:
        regional = regional.assign_coords(region=region_names)

    return regional
    

def pca_decomp(
    xda, n_components=10, return_plots=False, return_pca=False, **pca_kwargs,
):
    """
    Apply a principle component decomposition to a dataset with
    time, lat, lon axes.

    Should perhaps use the Eof package in Python
    """
    from sklearn.decomposition import PCA

    def unnan(arr):
        t, y, x = arr.shape
        flat = arr.reshape(t, -1)
        mask = ~np.isnan(flat).any(0)
        return flat[:, mask], mask

    def renan(arr, mask, shape=None):
        out = np.ndarray([min(arr.shape), mask.size]) * np.NaN
        if np.argmin(arr.shape) == 1:
            arr = arr.T
        out[:, mask] = arr
        out = out
        if shape:
            out = out.reshape(*shape)
        return out

    t, y, x = xda.dims

    assert t.lower() in [
        "time",
        "date",
        "tmnth",
        "days",
    ], "DataArray needs to have time as first dimension"
    assert (
        y.lower() in "ylatitude"
    ), "DataArray needs to have latitude as second dimension"
    assert (
        x.lower() in "xlongitude"
    ), "DataArray needs to have longitude as third dimension"

    coords = {d: xda[d].values for d in xda.dims}
    coords.update({"n_components": np.arange(n_components)})

    pca = PCA(n_components=n_components, **pca_kwargs)

    v, m = unnan(xda.values)

    trans = pca.fit_transform(v.T)
    trans_3D = renan(trans, m, shape=[n_components, coords[y].size, coords[x].size])

    xds = xr.Dataset(attrs={"name": xda.name})
    dims = ["n_components", "lat", "lon"]
    props = dict(coords={k: coords[k] for k in dims}, dims=dims)
    xds["transformed"] = xr.DataArray(trans_3D, **props)

    dims = ["n_components", "time"]
    props = dict(coords={k: coords[k] for k in dims}, dims=dims)
    xds["principle_components"] = xr.DataArray(pca.components_, **props)

    dims = ["time"]
    props = dict(coords={k: coords[k] for k in dims}, dims=dims)
    xds["mean_"] = xr.DataArray(pca.mean_, **props)

    dims = ["n_components"]
    props = dict(coords={k: coords[k] for k in dims}, dims=dims)
    xds["variance_explained"] = xr.DataArray(pca.explained_variance_ratio_, **props)

    if return_plots and return_pca:
        fig = _pca_plot(xds)
        return xds, pca, fig
    elif return_plots:
        fig = _pca_plot(xds)
        return xds, fig
    elif return_pca:
        return xds, pca
    else:
        return xds


def _pca_plot(xds_pca):
    from matplotlib import pyplot as plt

    n = xds_pca.n_components.size
    fig = plt.figure(figsize=[15, n * 3.2], dpi=120)
    shape = n, 5
    ax = []

    for i in range(shape[0]):
        ax += (
            [
                plt.subplot2grid(shape, [i, 0], colspan=3, fig=fig),
                plt.subplot2grid(
                    shape, [i, 3], colspan=2, fig=fig, facecolor="#AAAAAA"
                ),
            ],
        )

    t = xds_pca.principle_components.dims[-1]
    y, x = xds_pca.transformed.dims[1:]
    for i in xds_pca.n_components.values:
        pt = xds_pca[t].values
        px = xds_pca[x].values
        py = xds_pca[y].values
        pz = xds_pca.transformed[i].to_masked_array()

        var = xds_pca.variance_explained[i].values * 100
        lim = np.nanpercentile(abs(pz.filled(np.nan)), 99)

        a0 = ax[i][0]
        a1 = ax[i][1]

        a0.plot(pt, xds_pca.principle_components[i].values)
        a0.axhline(0, color="k")
        a0.set_ylabel("Component {}\n({:.2f}%)".format(i + 1, var), fontsize=12)

        img = a1.pcolormesh(
            px, py, pz, vmin=-lim, rasterized=True, vmax=lim, cmap=plt.cm.RdBu_r,
        )
        plt.colorbar(img, ax=a1)
        img.colorbar.set_label("Transformed units")

        if i != (shape[0] - 1):
            a0.set_xticklabels([])
            a1.set_xticklabels([])
        else:
            pass

    title = (
        "Principle Component Analysis (PCA) "
        "for {} showing the first {} components"
    )
    fig.suptitle(
        title.format(xds_pca.name, n),
        y=1.01,
        x=0.5,
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()

    return fig


def earth_radius(lat):
    """Calculate the radius of the earth for a given latitude

    Args:
        lat (array, float): latitude value (-90 : 90)

    Returns:
        array: radius in metres
    """
    from numpy import cos, deg2rad, sin

    lat = deg2rad(lat)
    a = 6378137
    b = 6356752
    r = (
        ((a ** 2 * cos(lat)) ** 2 + (b ** 2 * sin(lat)) ** 2)
        / ((a * cos(lat)) ** 2 + (b * sin(lat)) ** 2)
    ) ** 0.5

    return r


def area_grid(lat, lon, return_dataarray=False):
    """Calculate the area of each grid cell for given lats and lons

    Args:
        lat (array): latitudes in decimal degrees of length N
        lon (array): longitudes in decimal degrees of length M
        return_dataarray (bool, optional): if True returns xr.DataArray, else array

    Returns:
        array, xr.DataArray: area of each grid cell in meters

    References:
        https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import cos, deg2rad, gradient, meshgrid

    ylat, xlon = meshgrid(lat, lon)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=1))
    dlon = deg2rad(gradient(xlon, axis=0))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    if not return_dataarray:
        return area
    else:
        from xarray import DataArray

        xda = DataArray(
            area.T,
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
            attrs=dict(
                long_name="Area per pixel",
                units="m^2",
                description=(
                    "Area per pixel as calculated by pySeaFlux. The non-"
                    "spherical shape of Earth is taken into account."
                ),
            ),
        )

        return xda


def get_area(da, lat_name="lat", lon_name="lon"):

    x = da[lon_name].values
    y = da[lat_name].values

    out = area_grid(y, x, return_dataarray=True)

    name = getattr(da, 'name', 'xr.DataArray')
    description = out.attrs.get('description', '')
    description += f' Area calculated from the latitude and longitude coordinates of {name}'

    out.attrs['description'] = description
    
    return out


_make_xarray_accessor("spatial", [avg_area_weighted, aggregate_region, get_area])
