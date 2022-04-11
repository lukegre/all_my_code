import xarray as xr
from functools import wraps as _wraps
from ..utils import add_docs_line1_to_attribute_history, get_unwrapped


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


@add_docs_line1_to_attribute_history
def lon_180W_180E(ds, lon_name='lon'):
    """
    Regrid the data to [-180 : 180] from [0 : 360]
    """
    names = set(list(ds.coords) + list(ds.dims))
    if lon_name not in names:
        return ds
    
    lon180 = (ds[lon_name] - 180) % 360 - 180
    return ds.assign_coords(**{lon_name: lon180}).sortby(lon_name)


@add_docs_line1_to_attribute_history
def lon_0E_360E(ds, lon_name='lon'):
    """
    Regrid the data to [0 : 360] from [-180 : 180] 
    """
    names = set(list(ds.coords) + list(ds.dims))
    if lon_name not in names:
        return ds
    
    lon360 = ds[lon_name].values % 360
    ds = ds.assign_coords(**{lon_name: lon360}).sortby(lon_name)
    return ds
    
    
@add_docs_line1_to_attribute_history  
def coord_05_offset(ds, center=0.5, coord_name='lon'):
    """
    Interpolate data if the grid centers are offset.
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

    def has_coords(ds, checklist=['time', 'lat', 'lon']):
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
            
    return ds
    
    
@add_docs_line1_to_attribute_history
def transpose_dims(ds, default=['time', 'depth', 'lat', 'lon'], other_dims_before=True):
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
    
    matching = all([a==b for a,b in zip(ds.dims, new_order)])
    if not matching:
        ds = ds.transpose(*new_order)
    
    return ds


@add_docs_line1_to_attribute_history
def correct_coord_names(
    ds, 
    match_dict=dict(
        time=["month", "time", "t"],
        depth=["depth", "z", "lev", "z_t", "z_l"],
        lat=["lat", "latitude", "y"], 
        lon=["lon", "longitude", "x"])
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
    from . name_matching import guess_coords_from_column_names
    
    coord_keys = list(set(list(ds.coords) + list(ds.dims)))
    coord_renames = guess_coords_from_column_names(coord_keys, match_dict=match_dict)
    
    if any(coord_renames):
        ds = ds.rename(coord_renames)
    
    return ds


@add_docs_line1_to_attribute_history
def interpolate_1deg(xds, method="linear"):
    """
    interpolate the data to 1 degree resolution [-89.5 : 89.5] x [-179.5 : 179.5]
    """
    from warnings import warn
    from numpy import arange

    if xds.lon.max() > 180:
        warn("Longitude range is from 0 to 360, interpolate_1deg only works for -180 to 180")

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


@add_docs_line1_to_attribute_history
def time_center_monthly(ds, center_day=15, time_name='time'):
    """
    Date centered on a given date (default 15th)
    
    Data must be monthly for this function to work
    """
    from pandas import Timedelta as timedelta
    from . date_utils import datetime64ns_to_lower_order_datetime
    
    time = datetime64ns_to_lower_order_datetime(ds[time_name].values)
    
    if "[M]" not in str(time.dtype):
        raise ValueError("data time variable is not monthly")
    
    delta_days = timedelta(f'{center_day - 1}D')
    
    ds = ds.assign_coords(time=time.astype('datetime64[D]') + delta_days)
    
    return ds


@add_docs_line1_to_attribute_history
def drop_0d_coords(da):
    """drop the dimensions that are single dimension"""
    coords_to_drop = [c for c in da.coords if len(da[c].shape) == 0]
    da_dropped_coords = da.drop(coords_to_drop)
    return da_dropped_coords


@add_docs_line1_to_attribute_history
def rename_vars_snake_case(ds):
    """
    Rename variables to snake_case from CamelCase
    """
    from ..utils import camel_to_snake
    if isinstance(ds, xr.Dataset):
        ds = ds.rename({k: camel_to_snake(k) for k in ds.data_vars})
    elif isinstance(ds, xr.DataArray):
        name = getattr(ds, 'name', "")
        ds = ds.rename(camel_to_snake(name))

    return ds


_func_registry = [
    lon_0E_360E,
    lon_180W_180E,
    interpolate_1deg,
    coord_05_offset,
    transpose_dims,
    correct_coord_names,
    rename_vars_snake_case,
    time_center_monthly,
    drop_0d_coords,
]


_default_conform = [
    correct_coord_names,
    transpose_dims,
    lon_180W_180E,
    time_center_monthly,
    rename_vars_snake_case,
]


@xr.register_dataset_accessor("conform")
@xr.register_dataarray_accessor("conform")
class DataConform(object):
    """
    A class to conform a dataset/dataarray to a the desired conventions

    Modules (subfunctions) can be used to conform the dataset/dataarray 
    individually, or you can call this function to apply a set of standard 
    functions. 
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

        for func in _func_registry:
            setattr(self, get_unwrapped(func).__name__, self._make_accessor_func(func))

    def _make_accessor_func(self, func):
        @_wraps(get_unwrapped(func))
        def run_func(*args, **kwargs):
            return func(self._obj, *args, **kwargs)

        return run_func

    def __call__(
        self, 
        coord_names=True,
        time_centered=False,
        squeeze=True,
        transpose=True,
        lon_180W=True,
        standardize_var_names=False,
    ):
        da = self._obj

        funclist = []
        if coord_names:
            funclist.append(correct_coord_names)
        if time_centered:
            funclist.append(time_center_monthly)
        if squeeze:
            funclist.append(drop_0d_coords)
        if transpose:
            funclist.append(transpose_dims)
        if lon_180W:
            funclist.append(lon_180W_180E)
        if standardize_var_names:
            funclist.append(rename_vars_snake_case)

        out = apply_process_pipeline(da, *funclist)

        return out

