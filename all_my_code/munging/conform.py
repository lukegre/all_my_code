from pkg_resources import DistributionNotFound, get_distribution
import xarray as xr
from functools import wraps as _wraps

try:
    __version__ = get_distribution("ocean_data_tools").version
except DistributionNotFound:
    __version__ = ""
del get_distribution, DistributionNotFound


def apply_process_pipeline(ds, *funcs):
    """
    Applies a list of functions to an xarray.Dataset object.
    Functions must accept a Dataset and return a Dataset
    """
    for func in funcs:
        ds = func(ds)
    return ds


class processor:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        docs = func.__doc__
        self.msg = docs.strip().split("\n")[0] if isinstance(docs, str) else ""

    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            try:
                out = self._add_history(self.func(*args, **kwargs))
                return out
            except Exception as e:
                raise e
                return args[0]

        self.kwargs = kwargs
        return self.__caller__

    def __caller__(self, ds):
        return self._add_history(self.func(ds, **self.kwargs))

    def _add_history(self, ds, key='history'):
        from pandas import Timestamp

        version = ".{__version__}" if __version__ else ""
        
        now = Timestamp.today().strftime("%y%m%d")
        prefix = f"(ODT{version}@{now}) "
        msg = prefix + self.msg
        
        hist = ds.attrs.get(key, '')
        if hist != '':
            hist = hist.split(";")
            hist = [h.strip() for h in hist]
            msg = "; ".join(hist + [msg])
            
        ds = ds.assign_attrs({key: msg})

        return ds


@processor
def lon_180W_180E(ds, lon_name='lon'):
    """
    Regrid the data to [-180 : 180] from [0 : 360]
    """
    names = set(list(ds.coords) + list(ds.dims))
    if lon_name not in names:
        return ds
    
    lon180 = (ds[lon_name] - 180) % 360 - 180
    return ds.assign_coords(**{lon_name: lon180}).sortby(lon_name)


@processor
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
    
    
@processor  
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
    
    
@processor
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


@processor
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


_func_registry_both = [
    lon_0E_360E,
    lon_180W_180E,
    coord_05_offset,
    transpose_dims,
    correct_coord_names,
]

@xr.register_dataset_accessor("prep")
@xr.register_dataarray_accessor("prep")
class PreperationBoth(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

        for func in _func_registry_both:
            setattr(self, func.func.__name__, self._make_accessor_func(func))

    def _make_accessor_func(self, func):
        @_wraps(func)
        def run_func(*args, **kwargs):
            return func(self._obj, *args, **kwargs)

        return run_func
