import xarray as xr
from ..utils import append_attr
from tempfile import gettempdir


def lon_180W_180E(ds, lon_name="lon"):
    """
    Regrid the data to [-180 : 180] from [0 : 360]
    """
    from numpy import isclose

    lon = ds[lon_name].values
    lon180 = (lon - 180) % 360 - 180
    if isclose(lon, lon180).all():
        return ds
    ds = append_attr(ds, "regridded to [-180 : 180] from [0 : 360]")
    return ds.assign_coords(**{lon_name: lon180}).sortby(lon_name)


def lon_0E_360E(ds, lon_name="lon"):
    """
    Regrid the data to [0 : 360] from [-180 : 180]
    """
    from numpy import isclose

    lon = ds[lon_name].values
    lon360 = lon % 360
    if isclose(lon, lon360).all():
        return ds
    ds = ds.assign_coords(**{lon_name: lon360}).sortby(lon_name)
    ds = append_attr(ds, "regridded to [0 : 360] from [-180 : 180]")
    return ds


def coord_05_offset(ds, center=0.5, coord_name="lon"):
    """
    Interpolate data to grid centers.
    Only works for 1deg data

    Parameters
    ----------
    ds: xr.Dataset
        the dataset with a coordinate variable variable
    center: float
        the desired center point of the grid points between 0 - 1
    coord_name: str [lon]
        the name of the coordinate

    Returns
    -------
    xr.Dataset: interpolated onto the new grid with the new
        coord being the old coord + center
    """

    def has_coords(ds, checklist=["time", "lat", "lon"]):
        """
        Check that data has coordinates
        """
        matches = {key: (key in ds.coords) for key in checklist}
        if all(matches.values()):
            return 1
        else:
            return 0

    center = center - (center // 1)
    if has_coords(ds):
        coord = ds[coord_name].values
        mod = coord - (coord // 1)
        # use the modulus to determine if grid centers are correct
        if any(mod != center):
            ds = ds.interp({coord_name: coord + center})
            ds = ds.sel(lat=slice(-90, 90))

    return ds


def regrid(
    ds,
    weights_path=gettempdir(),
    res=1,
    like=None,
    mask=None,
    keep_attrs=True,
    verbose=True,
    recommendation="raise",
    overwrite_weights=False,
    **kwargs,
):
    """
    Regrid data using xesmf

    Weights can be reused making this method extremely fast for regridding large amounts
    of data. Weights are automatically saved to disk.

    Parameters
    ----------
    ds: xr.Dataset
    weights_path: path-like str
        Can be one of three options
        1. path to the directory where the weights will be saved (default names)
        2. path to a file - will be created if it does not exist
        The default file name is the a similar format to the xesmf default
        name + a hash based on the lat and lon values of both datasets.
        The default directory is a temporary directory - warning - this will
        persist unless deleted. recommended that you change the dir.
    res: float
        resolution of the grid to interpolate to
    like: xr.Dataset
        dataset to use as a template for the new grid - ignores res if given
    **kwargs:
        passed to xesmf.Regridder. The method can be overwridden by passing
        method='<other method>'. extrap_method is set to nearest_s2d by default
    """

    def get_latlon_str(ds):
        import numpy as np

        coords_list = [np.array(ds.coords[k]) for k in ("lat", "lon")]
        coords = np.concatenate(coords_list)
        coords_str = str(coords)
        return coords_str

    def make_hash(string, hash_length=6):
        from hashlib import sha1

        hash = sha1(string.encode("UTF-8")).hexdigest()[:hash_length]
        return hash

    def make_default_filename(method, ds_in, ds_out):
        # e.g. bilinear_400x600_300x400.nc
        iy = ds_in.lat.size
        ix = ds_in.lon.size
        oy = ds_out.lat.size
        ox = ds_out.lon.size

        hash = make_hash(get_latlon_str(ds_in) + get_latlon_str(ds_out))
        filename = f"xesmf-weights_{method}_in{iy}x{ix}_out{oy}x{ox}_{hash}.nc"
        return filename

    def weights_to_netcdf(regridder, filename):
        """Save weights to disk as a netCDF file."""
        w = regridder.weights.data
        dim = "n_s"
        ds = xr.Dataset(
            {
                "S": (dim, w.data),
                "col": (dim, w.coords[1, :] + 1),
                "row": (dim, w.coords[0, :] + 1),
            }
        )
        encoding = {k: {"zlib": True, "complevel": 1} for k in ds.data_vars}
        ds.to_netcdf(filename, encoding=encoding)

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    import os
    import xesmf as xe
    import xarray as xr
    from ..files.utils import is_path_exists_or_creatable
    from warnings import filterwarnings

    filterwarnings("ignore", category=UserWarning, module="xesmf")

    assert "lat" in ds.coords, "Data must have lat coordinate"
    assert "lon" in ds.coords, "Data must have lon coordinate"

    if (like is None) and (mask is None):
        like = xe.util.grid_global(res, res, cf=True)
    elif mask is not None:
        assert isinstance(mask, xr.DataArray), "mask must be an xr.DataArray"
        like = mask.astype(int).to_dataset(name="mask")
        assert "mask" in ds.data_vars, "Data must have mask variable if `mask` is given"
    elif like is not None and mask is not None:
        raise ValueError("`like` and `mask` cannot both be given")

    _is_interp_best(ds.lat, ds.lon, like.lat, like.lon, recommendation)

    method = kwargs.pop("method", "bilinear")

    m = "bilinear and conservative interpolation"
    vprint(f"xesmf will be used for {m}")

    # THIS SECTION IS TO DEAL WITH SAVING THE WEIGHTS
    # if the given path is not a file, then create a default filename
    path_valid = is_path_exists_or_creatable(weights_path)
    file_exist = os.path.isfile(weights_path)
    if not path_valid:
        raise ValueError(
            f"{weights_path} is not path-like. Must be "
            "a creatable or existing file or directory"
        )
    elif path_valid and not file_exist:
        default_sname = make_default_filename(method, ds, like)
        weights_path = os.path.join(weights_path, default_sname)
        file_exist = os.path.isfile(weights_path)
    else:
        # path is valid and file exists
        # this is just so that I know I'm not missing any options
        pass
    weights_path = os.path.abspath(os.path.expanduser(weights_path))

    if file_exist and overwrite_weights:
        vprint(f"Overwriting weights file: {weights_path}")
        os.remove(weights_path)
        file_exist = False
        kwargs["weights"] = None
    elif file_exist:
        vprint(f"Loading weights from file {weights_path}")
        kwargs["weights"] = xr.open_dataset(weights_path)
    else:
        vprint(f"Creating weights (could take some time) and saving to {weights_path}")
        kwargs["weights"] = None

    props = dict(extrap_method="nearest_s2d")
    props.update(**kwargs)

    try:
        regridder = xe.Regridder(ds, like, method, **props)
    except ValueError:
        raise ValueError(
            "invalid entry in coordinates array. "
            "Weights file may not match the desired resolution"
        )

    if not file_exist:
        weights_to_netcdf(regridder, weights_path)

    interpolated = regridder(ds)
    new_attrs = interpolated.attrs

    if keep_attrs:
        interpolated.attrs = {}
        interpolated = interpolated.assign_attrs(**ds.attrs)
        if isinstance(ds, xr.Dataset):
            for k in interpolated.data_vars:
                interpolated[k] = interpolated[k].assign_attrs(**ds[k].attrs)

    interpolated = interpolated.assign_attrs(regrid_weights=weights_path, **new_attrs)
    interpolated = append_attr(interpolated, f"regridded with xesmf using {method}")

    return interpolated


def interp(ds, res=1, like=None, method="linear", recommendation="warn", **kwargs):
    """
    Interpolate and fill the longitude gap in a dataset

    Parameters
    ----------
    ds: xr.Dataset
    lon_name: str
        name of the longitude coordinate
    roll_by: int
        number of grid points to roll the data by - must be more than the gap
        if interpolating from low to high resolution this will be a problem
    **kwargs:
        passed to xr.interp
    """
    if ("lat" in kwargs) or ("lon" in kwargs):
        like = xr.DataArray(
            dims=["lat", "lon"],
            coords={"lat": kwargs.pop("lat"), "lon": kwargs.pop("lon")},
        )
    elif like is None:
        like = _make_like_array(res)

    assert ("lat" in like.coords) and (
        "lon" in like.coords
    ), "'like' must have lat and lon coordinates"

    _is_interp_best(ds.lat, ds.lon, like.lat, like.lon, recommendation)

    props = dict(**kwargs)
    props.update(method=method)
    roll_by = int(like.lon.size // 3)
    interpolated = (
        ds.interp_like(like, **props)
        .roll(**{"lon": roll_by}, roll_coords=False)
        .interpolate_na("lon", limit=int(roll_by / 2))
        .roll(**{"lon": -roll_by}, roll_coords=False)
    )

    interpolated = append_attr(
        interpolated,
        f"interpolated to {res}deg resolution using {method} interpolation",
    )

    return interpolated


def coarsen(ds, res_out=1.0):
    """
    Coarsen a dataset to a given resolution
    Will return an error if coarsening is not suitable

    Parameters
    ----------
    ds: xr.Dataset
    res_out: float
        desired resolution

    Returns
    -------
    xr.Dataset
    """

    from ..utils import append_attr
    import numpy as np

    res_in = np.around(float(ds.lat.diff("lat").mean()), 4)
    res_out = np.around(res_out, 4)
    ratio = res_out / res_in
    if abs(ratio - np.round(ratio)) > 0.05:
        raise ValueError(
            f"The input resolution ({res_in}) and "
            f"output resolution ({res_out}) are not "
            "divisible to an intiger"
        )
    coarsen_step = np.int32(np.round(ratio))

    coord_func = lambda x, **kwargs: np.round(np.mean(x, **kwargs), 3)

    ds = append_attr(
        ds, f"coarsened resolution from {res_in:.3g}deg to {res_out:.3g}deg"
    )
    coarse = ds.coarsen(lat=coarsen_step, lon=coarsen_step, coord_func=coord_func)

    return coarse


def _create_time_bnds(time_left):
    import numpy as np

    t = time_left.values
    dt = np.nanmedian(np.diff(t))
    t = np.concatenate([t, [t[-1] + dt]])

    time_bnds = xr.DataArray(
        np.c_[t[:-1], t[1:]],
        dims=[time_left.name, "bnds"],
        coords={time_left.name: time_left},
        attrs={
            "description": (
                "time bands. note that time dimension " "is left aligned to the band"
            )
        },
    )
    return time_bnds


def resample(ds, func="mean", **kwargs):
    """
    Resample time resolution and add a time_bnds coordinate

    Parameters
    ----------
    ds: xr.Dataset
    time_res: str
        time resolution in the format of '<int>D'
        where int is the number of days
    func: str
        function to apply to the data

    Returns
    -------
    ds: xr.Dataset
    """
    from ..utils import append_attr

    ds_res = ds.resample(**kwargs)
    ds_out = getattr(ds_res, func)(keep_attrs=True)
    dim = list(kwargs)[0]
    res = kwargs[dim]
    if isinstance(ds, xr.DataArray):
        ds_out = append_attr(ds_out, f"resampled {dim} to {res} using `{func}`")
    elif isinstance(ds, xr.Dataset):
        ds_out["time_bands"] = _create_time_bnds(ds_out.time)
        ds_out = ds_out.append_attrs(
            history=f"resampled {dim} to {res} using `{func}` and added time_bands"
        )
    return ds_out


def _is_interp_best(iy, ix, oy, ox, recommendation="warn"):
    from warnings import warn

    idx = ix.diff("lon", 1).median().values
    idy = iy.diff("lat", 1).median().values
    odx = ox.diff("lon", 1).median().values
    ody = oy.diff("lat", 1).median().values
    ratio_x = odx / idx
    ratio_y = ody / idy
    if (ratio_x > 2) | (ratio_y > 2):
        message = (
            "The output grid is less than half the resolution of the input grid. "
            "Interpolation may not be the best approach. "
            f"Consider using da.coarsen(lat={ratio_y:.0f}, lon={ratio_x:.0f}).mean()"
        )
        if recommendation == "warn":
            warn(message)
        elif recommendation == "raise":
            raise ValueError(message)
        elif recommendation == "ignore":
            pass


def _make_like_array(resolution):
    import xarray as xr
    import numpy as np

    r = resolution
    grids = xr.DataArray(
        dims=["lat", "lon"],
        coords={
            "lat": np.arange(-90 + r / 2, 90, r),
            "lon": np.arange(-180 + r / 2, 180, r),
        },
    )

    return grids


def estimate_grid_spacing(coord):
    """
    Estimate the grid spacing from a coordinate

    Parameters
    ----------
    coord: xr.DataArray
        coordinate

    Returns
    -------
    float
        grid spacing
    """
    import numpy as np

    delta_x = np.diff(coord.values)
    delta_x_u = np.unique(delta_x)
    if len(delta_x_u) > 1:
        return np.median(delta_x)
    elif len(delta_x_u) == 1:
        return delta_x_u[0]
