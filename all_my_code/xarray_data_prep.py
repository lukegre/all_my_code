"""
Used to process netCDF files to a standard format.
All functions take in an xarray.Dataset and return an xarray.Dataset
All functions in this preprocessing module should add metadata to the
xarray object under `history`
"""
from functools import wraps

from pkg_resources import DistributionNotFound, get_distribution

import xarray as xr
from astropy import convolution as conv

try:
    __version__ = get_distribution("ocean_data_tools").version
except DistributionNotFound:
    __version__ = ""
del get_distribution, DistributionNotFound


def apply_process_pipeline(xds, *funcs):
    """
    Applies a list of functions to an xarray.Dataset object.
    Functions must accept a Dataset and return a Dataset
    """
    for func in funcs:
        xds = func(xds)
    return xds


class processor:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        docs = func.__doc__
        self.msg = docs.strip().split("\n")[0] if isinstance(docs, str) else ""

    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            return self._add_history(self.func(*args, **kwargs))

        self.kwargs = kwargs
        return self.__caller__

    def __caller__(self, xds):
        return self._add_history(self.func(xds, **self.kwargs))

    def _add_history(self, xds):
        from pandas import Timestamp

        version = ".{__version__}" if __version__ else ""
        now = Timestamp.today().strftime("%Y-%m-%dT%H:%M")
        prefix = f"[OceanDataTools{version}@{now}] "
        msg = prefix + self.msg
        if "history" not in xds.attrs:
            xds.attrs["history"] = msg
        elif xds.attrs["history"] == "":
            xds.attrs["history"] = msg
        else:
            hist = xds.attrs["history"].split(";")
            hist = [h.strip() for h in hist]
            xds.attrs["history"] = "; ".join(hist + [msg])

        return xds


@processor
def rename_to_timelatlon(xds, **kwargs):
    """
    Rename time, lat, lon to standard names

    Parameters
    ----------
    xds : xr.Dataset or xr.DataArray
        Dataset that will be renamed so that dimensions are standard
    kwargs : key-value pairs
        key is the name you would like to change the dimension to
        value is a list or string of possible names you'd like to change

    Returns
    -------
    xds : same as input, but with changes made
    """

    def make_list(val):
        if isinstance(val, str):
            return [val]
        elif isinstance(val, [list, tuple]):
            return list(val)
        else:
            raise TypeError(f"{val} must be str, list or tuple")

    coords = dict(
        time=["mtime"],
        lat=["latitude", "lats", "yt_ocean", "ylat"],
        lon=["longitude", "lons", "xt_ocean", "longs", "long", "xlon"],
        depth=[],
    )
    for key in kwargs:
        if key in coords:
            coords[key] += make_list(kwargs[key])

    rename_dict = {}
    for key in xds.coords.keys():
        if key in coords["time"]:
            rename_dict[key] = "time"
        if key in coords["lat"]:
            rename_dict[key] = "lat"
        if key in coords["lon"]:
            rename_dict[key] = "lon"
        if key in coords["depth"]:
            rename_dict[key] = "depth"

    xds = xds.rename(rename_dict)

    return xds


@processor
def center_coords_at_0(xds, lon_offset=0, lat_offset=0):
    """
    Change longitude from 0:360 to -180:180

    Parameters
    ----------
    xds : xr.Dataset or xr.DataArray

    Returns
    -------
    xds : same as input type
        lon will be from -180 to 180 and lats will also be increasing
    offset_kwargs: keyword-value pairs
        the keyword matches the dimension name that you would like to
        add an offset to and the value is the offset magnitude
    """
    import numpy as np

    assert "lat" in xds.dims, "lat is not in Dataset/DataArray"
    assert "lon" in xds.dims, "lon is not in Dataset/DataArray"

    def strictly_increasing(L):
        return all([x < y for x, y in zip(L, L[1:])])

    x = xds["lon"].values + lon_offset
    y = xds["lat"].values + lat_offset
    x = lon_shift(x)
    xds = xds.assign_coords(lon=x, lat=y)

    if not strictly_increasing(x):
        sort_idx = np.argsort(x)
        xds = xds.isel(**{"lon": sort_idx})
        xds = xds.assign_coords(lon=x[sort_idx])

    if not strictly_increasing(y):
        xds = xds.isel(**{"lat": slice(None, None, -1)})

    return xds


@processor
def time_month_day(xds, dayofmonth=1):
    """
    Shift time day of month

    Parameters
    ----------
    xds : xr.Dataset or xr.DataArray
        Dataset that will be renamed so that dimensions are standard
    dayofmonth : int
        the day of the month you'd like to shift the days to. Defaults
        to the 1st.

    Returns
    -------
    xds : same as input, but with changes made
    """
    import numpy as np

    time = xds.time.values.astype("datetime64[ns]")
    timem = time.astype("datetime64[M]").astype("datetime64[ns]")
    timeu = np.unique(timem)
    assert timeu.size == timem.size

    timed = timem + np.timedelta64(dayofmonth - 1, "D")
    xds = xds.assign_coords(time=timed)

    return xds


@processor
def shallowest(xda, depth_dim=None):
    """
    Surface data selected

    Parameters
    ----------
    xds : xr.Dataset or xr.DataArray
        Dataset that will be renamed so that dimensions are standard
    depth_dim : str | None
        if not defined, function will look for variable that has `dept`
        in the name, or where units have `meters`, otherwise returns
        unchanged array

    Returns
    -------
    xds : same as input, but with changes made
    """

    if depth_dim is None:
        for dim in xda.dims:
            depth_dim = None
            if hasattr(xda[dim], "units"):
                units = xda[dim].units
                if "meters" in units:
                    depth_dim = dim
                    break
            if "dept" in dim:
                depth_dim = dim
                break
        if depth_dim is None:
            return xda
    else:
        assert isinstance(depth_dim, str), "depth_dim must be string"
        assert depth_dim in xda.dims, f"{depth_dim} is not in DataArray"

    xda = xda.sel(**{depth_dim: 0}, method="nearest").drop(depth_dim)
    return xda


@processor
def interpolate_1deg(xds, method="linear"):
    from numpy import arange

    attrs = xds.attrs
    xds = (
        xds.interp(lat=arange(-89.5, 90), lon=arange(-179.5, 180), method=method)
        # filling gaps due to interpolation along 180deg
        .roll(lon=180, roll_coords=False)
        .interpolate_na(dim="lon", limit=3)
        .roll(lon=-180, roll_coords=False)
    )

    xds.attrs = attrs

    return xds


@processor
def resample_time(xds, freq="1D"):
    attrs = xds.attrs

    xds = xds.resample(time=freq, keep_attrs=True).mean("time", keep_attrs=True)

    xds.attrs.update(attrs)

    return xds


@processor
def convolve(xda, kernel=None, fill_nans=False, verbose=True):
    import warnings

    warnings.filterwarnings("ignore", ".*A contiguous region of NaN values.*")

    def _convlve_timestep(xda, kernel, preserve_nan):
        convolved = xda.copy()
        convolved.values = conv.convolve(
            xda.values, kernel, preserve_nan=preserve_nan, boundary="wrap"
        )
        return convolved

    ndims = len(xda.dims)
    preserve_nan = not fill_nans

    if kernel is None:
        kernel = conv.Gaussian2DKernel(x_stddev=2)
    elif isinstance(kernel, list):
        if len(kernel) == 2:
            kernel_size = kernel
            for i, ks in enumerate(kernel_size):
                kernel_size[i] += 0 if (ks % 2) else 1
            kernel = conv.kernels.Box2DKernel(max(kernel_size))
            kernel._array = kernel._array[: kernel_size[0], : kernel_size[1]]
        else:
            raise UserWarning(
                "If you pass a list to `kernel`, it needs to have a length of 2"
            )
    elif kernel.__class__.__base__ == conv.core.Kernel2D:
        kernel = kernel
    else:
        raise UserWarning(
            "kernel needs to be a list or astropy.kernels.Kernel2D base type"
        )

    if ndims == 2:
        convolved = _convlve_timestep(xda, kernel, preserve_nan)
    elif ndims == 3:
        convolved = []
        for t in range(xda.shape[0]):
            convolved += (_convlve_timestep(xda[t], kernel, preserve_nan),)
        convolved = xr.concat(convolved, dim=xda.dims[0])

    kern_size = kernel.shape
    convolved.attrs["description"] = (
        "same as `{}` but with {}x{}deg (lon x lat) smoothing using "
        "astropy.convolution.convolve"
    ).format(xda.name, kern_size[0], kern_size[1])
    return convolved


def lon_shift(lon):
    return ((lon + 180) % 360) - 180


_func_registry_both = [
    center_coords_at_0,
    interpolate_1deg,
    rename_to_timelatlon,
    resample_time,
    shallowest,
    time_month_day,
]


@xr.register_dataset_accessor("prep")
@xr.register_dataarray_accessor("prep")
class PreperationBoth(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

        for func in _func_registry_both:
            setattr(self, func.func.__name__, self._make_accessor_func(func))

    def _make_accessor_func(self, func):
        @wraps(func)
        def run_func(*args, **kwargs):
            return func(self._obj, *args, **kwargs)

        return run_func
