import xarray as xr
from functools import wraps as _wraps
from ..utils import add_docs_line1_to_attribute_history, get_unwrapped
from tempfile import gettempdir


def lon_180W_180E(ds, lon_name='lon'):
    """
    Regrid the data to [-180 : 180] from [0 : 360]
    """
    from numpy import isclose

    lon = ds[lon_name].values
    lon180 = (lon - 180) % 360 - 180
    if isclose(lon, lon180).all():
        return ds
    return ds.assign_coords(**{lon_name: lon180}).sortby(lon_name)


def lon_0E_360E(ds, lon_name='lon'):
    """
    Regrid the data to [0 : 360] from [-180 : 180] 
    """
    from numpy import isclose
    
    lon = ds[lon_name].values
    lon360 = lon % 360
    if isclose(lon, lon360).all():
        return ds
    ds = ds.assign_coords(**{lon_name: lon360}).sortby(lon_name)
    return ds
    
      
def coord_05_offset(ds, center=0.5, coord_name='lon'):
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
            ds = ds.sel(lat=slice(-90, 90))
            
    return ds


def regrid(ds, weights_path=gettempdir(), res=1, like=None, keep_attrs=True, verbose=True, **kwargs):
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
        coords_list = [np.array(ds.coords[k]) for k in ('lat', 'lon')]
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
        filename = f'xesmf-weights_{method}_in{iy}x{ix}_out{oy}x{ox}_{hash}.nc'
        return filename

    def weights_to_netcdf(regridder, filename):
        """Save weights to disk as a netCDF file."""
        w = regridder.weights.data
        dim = 'n_s'
        ds = xr.Dataset(
            {'S': (dim, w.data), 'col': (dim, w.coords[1, :] + 1), 'row': (dim, w.coords[0, :] + 1)}
        )
        encoding = {k: {'zlib': True, 'complevel': 1} for k in ds.data_vars}
        ds.to_netcdf(filename, encoding=encoding)
    
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    import os
    import xesmf as xe
    import xarray as xr
    from ..files.utils import is_path_exists_or_creatable

    assert 'lat' in ds.coords, 'Data must have lat coordinate'
    assert 'lon' in ds.coords, 'Data must have lon coordinate'

    if like is None:
        like = xe.util.grid_global(res, res, cf=True)
    
    _is_interp_best(ds.lat, ds.lon, like.lat, like.lon)

    method = kwargs.pop('method', 'bilinear')

    m = 'bilinear and conservative interpolation'
    vprint(f'xesmf will be used for {m}')

    # THIS SECTION IS TO DEAL WITH SAVING THE WEIGHTS 
    # if the given path is not a file, then create a default filename
    path_valid = is_path_exists_or_creatable(weights_path)
    file_exist = os.path.isfile(weights_path)
    if not path_valid:
        raise ValueError(
            f'{weights_path} is not path-like. Must be '
            'a creatable or existing file or directory')
    elif path_valid and not file_exist:
        default_sname = make_default_filename(method, ds, like)
        weights_path = os.path.join(weights_path, default_sname)
        file_exist = os.path.isfile(weights_path)
    else:
        # path is valid and file exists
        # this is just so that I know I'm not missing any options
        pass
    weights_path = os.path.abspath(os.path.expanduser(weights_path))

    if file_exist:
        vprint(f'Loading weights from file {weights_path}')
        kwargs['weights'] = xr.open_dataset(weights_path)
    else:
        vprint(f'Creating weights (could take some time) and saving to {weights_path}')
        kwargs['weights'] = None

    props = dict(extrap_method='nearest_s2d')
    props.update(**kwargs)

    try:  
        regridder = xe.Regridder(ds, like, method, **props)
    except ValueError as e:
        raise ValueError(
            'invalid entry in coordinates array. '
            'Weights file may not match the desired resolution')

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

    return interpolated


def interp(ds, res=1, like=None, roll_by=10, method='linear', **kwargs):
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
    import numpy as np

    if ('lat' in kwargs) or ('lon' in kwargs):
        like = xr.DataArray(
            dims=['lat', 'lon'],
            coords={'lat': kwargs.pop('lat'), 'lon': kwargs.pop('lon')})
    elif like is None:
        like = xr.DataArray(
            dims=['lat', 'lon'],
            coords={
                'lat': np.arange(-90 + res / 2, 90, res), 
                'lon': np.arange(-180 + res / 2, 180, res)})
    
    assert ('lat' in like.coords) and ('lon' in like.coords), "'like' must have lat and lon coordinates"

    _is_interp_best(ds.lat, ds.lon, like.lat, like.lon)

    props = dict(**kwargs)
    props.update(method=method)
    interpolated = (
        ds
        .interp_like(like, **props)
        .roll(**{'lon': roll_by}, roll_coords=False)
        .interpolate_na('lon', limit=int(roll_by / 2))
        .roll(**{'lon': -roll_by}, roll_coords=False))

    return interpolated


def _is_interp_best(iy, ix, oy, ox):
    from warnings import warn

    idx = ix.diff('lon', 1).median().values
    idy = iy.diff('lat', 1).median().values
    odx = ox.diff('lon', 1).median().values
    ody = oy.diff('lat', 1).median().values
    ratio_x = odx / idx
    ratio_y = ody / idy
    if (ratio_x > 2) | (ratio_y > 2):
        warn(
            "The output grid is less than half the resolution of the input grid. "
            "Interpolation may not be the best approach. "
            f"Consider using da.coarsen(lat={ratio_y:.0f}, lon={ratio_x:.0f}).mean()")
