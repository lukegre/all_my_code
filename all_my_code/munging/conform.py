import xarray as xr


def apply_process_pipeline(ds, *funcs):
    """
    Applies a list of functions to an xarray.Dataset object.
    Functions must accept a Dataset and return a Dataset
    """
    for func in funcs:
        try:
            ds = func(ds)
        except:
            pass
    return ds


def transpose_dims(ds, default=["time", "depth", "lat", "lon"], other_dims_before=True):
    """
    Transpose dimensions to [time, depth, lat, lon].
    Can specify if remaining dimensions should be ordered before
    or after the default dimensions.
    """
    old_order = list(ds.dims)
    dims = set(old_order)
    default = [d for d in default if d in dims]
    default_set = set(default)
    other = dims - default_set

    if other_dims_before:
        new_order = list(other) + list(default)
    else:
        new_order = list(default) + list(other)

    matching = all([a == b for a, b in zip(ds.dims, new_order)])
    if not matching:
        ds = ds.transpose(*new_order)

    return ds


def correct_coord_names(
    ds,
    match_dict=dict(
        time=["month", "time", "t"],
        depth=["depth", "z", "lev", "z_t", "z_l"],
        lat=["lat", "latitude", "y"],
        lon=["lon", "longitude", "x"],
    ),
):
    """
    Rename coordinates to [time, lat, lon, depth] with fuzzy matching

    Parameters
    ----------
    ds: xr.Dataset
    match_dict: dict
        A dictionary where the keys are the desired coordinate/dimension names
        The values are the nearest guesses for possible names. Note these do
        not have to match the possible names perfectly.

    Returns
    -------
    xr.Dataset: with renamed coordinates that match the keys from match_dict
    """
    from .name_matching import guess_coords_from_column_names

    coord_keys = list(set(list(ds.coords) + list(ds.dims)))
    coord_renames = guess_coords_from_column_names(coord_keys, match_dict=match_dict)

    if any(coord_renames):
        ds = ds.rename(coord_renames)

    return ds


def time_center_monthly(ds, center_day=15, time_name="time"):
    """
    Date centered on a given date (default 15th)

    Data must be monthly for this function to work
    """
    from pandas import Timedelta as timedelta
    from .date_utils import datetime64ns_to_lower_order_datetime

    time = datetime64ns_to_lower_order_datetime(ds[time_name].values)

    if "[M]" not in str(time.dtype):
        raise ValueError("data time variable is not monthly")

    delta_days = timedelta(f"{center_day - 1}D")

    ds = ds.assign_coords(time=time.astype("datetime64[D]") + delta_days)

    return ds


def drop_0d_coords(da):
    """
    Removes single-dimensional elements from a multi-dimensional data array.

    Parameters
    ----------
    da : xarray.DataArray
        The original multi-dimensional data array.

    Returns
    -------
    da_dropped_coords : xarray.DataArray
        The resulting data array after all single-dimensional coordinates have
        been removed.
    """

    da = da.squeeze()
    coords_to_drop = [c for c in da.coords if len(da[c].shape) == 0]
    da_dropped_coords = da.drop(coords_to_drop)
    return da_dropped_coords


def rename_vars_snake_case(ds):
    """
    Rename variables to snake_case from CamelCase
    """
    from ..utils import camel_to_snake

    if isinstance(ds, xr.Dataset):
        ds = ds.rename({k: camel_to_snake(k) for k in ds.data_vars})
    elif isinstance(ds, xr.DataArray):
        name = getattr(ds, "name", "")
        ds = ds.rename(camel_to_snake(name))

    return ds
